# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class AloraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["lora_A", "lora_B", "lora_router"]

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.lora_router = nn.ModuleDict({})

        # Mark the weight as unmerged. In this adapter we can't merge weights back into the base model.
        self.merged = False
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A[adapter_name] = nn.Parameter(torch.empty((num_experts, self.in_features, r)))
            self.lora_B[adapter_name] = nn.Parameter(torch.empty((num_experts, r, self.out_features)))
            self.lora_router[adapter_name] = nn.Sequential(
                nn.Linear(self.in_features, 128),
                nn.ReLU(),
                nn.Linear(128, num_experts)
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)


    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])

    def scale_layer(self, scale_factor: float) -> None:
        if scale_factor != 1:
            for active_adapter in self.active_adapters:
                alpha = self.lora_alpha[active_adapter]
                r = self.r[active_adapter]
                self.scaling[active_adapter] = (alpha / r) * scale_factor

    def unscale_layer(self) -> None:
        for active_adapter in self.active_adapters:
            alpha = self.lora_alpha[active_adapter]
            r = self.r[active_adapter]
            self.scaling[active_adapter] = alpha / r


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Linear, AloraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_experts: int = 1,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        AloraLayer.__init__(self, in_features=in_features, out_features=out_features, num_experts=num_experts)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts)
        self.set_adapter(adapter_name)


    # Does this makes sense?
    def get_delta_weight(self, adapter) -> torch.Tensor:
        return (
            transpose(
                self.lora_B[adapter] @ self.lora_A[adapter],
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     previous_dtype = x.dtype
    #     result = self._linear(x)

    #     # The molora model only supports one active adapter at a time
    #     active_adapter = self.active_adapters[0]
    #     if active_adapter in self.lora_A.keys():
    #         lora_A = self.lora_A[active_adapter] # Shape: [num_experts, in_features, r]
    #         lora_B = self.lora_B[active_adapter] # Shape: [num_experts, r, out_features]
    #         lora_router = self.lora_router[active_adapter]
    #         dropout = self.lora_dropout[active_adapter]
    #         scaling = self.scaling[active_adapter]

    #         # Compute expert_weights using the routing layer
    #         logits = lora_router(x)
    #         expert_weights = F.softmax(logits, dim=-1)
    #         # Expand expert_weights across sequence length
    #         expert_weights = expert_weights.unsqueeze(1).expand(-1, x.shape[1], -1)

    #         # Using einsum for lora_A matrix multiplication
    #         x_transformed = torch.einsum('bsi,eij->bsej', x, lora_A)
    #         x_transformed = dropout(x_transformed)

    #         # Using einsum for lora_B matrix multiplication
    #         weighted_output = torch.einsum('bsej,ejk,bse->bsk', x_transformed, lora_B, expert_weights)
    #         result += weighted_output * scaling

    #     result = result.to(previous_dtype)
    #     return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self._linear(x)

        # The molora model only supports one active adapter at a time
        active_adapter = self.active_adapters[0]
        if active_adapter in self.lora_A.keys():
            lora_A = self.lora_A[active_adapter] # Shape: [num_experts, in_features, r]
            lora_B = self.lora_B[active_adapter] # Shape: [num_experts, r, out_features]
            lora_router = self.lora_router[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # Compute expert_weights using the routing layer
            logits = lora_router(x)
            expert_weights = F.softmax(logits, dim=-1)

            # Compute ax using einsum
            ax = torch.einsum('bsi,eij->bsej', x, lora_A)
            ax = dropout(ax)

            # Compute bax using einsum
            bax = torch.einsum('bsej,ejk->bske', ax, lora_B)

            # Combine using router probabilities
            lora_output = torch.einsum('...e,...ek->...k', expert_weights, bax) * scaling

            # Add the output of the original linear layer
            result += lora_output

        result = result.to(previous_dtype)
        return result


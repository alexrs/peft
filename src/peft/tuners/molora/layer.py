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

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class Router(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super(Router, self).__init__()
        self.linear = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        router_logits = self.linear(x)
        router_probs = F.softmax(router_logits, dim=-1)
        return router_probs


class MoLoraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["lora_A", "lora_B"]

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        # Mark the weight as unmerged
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
            self.lora_A[adapter_name] = nn.Parameter(torch.randn(num_experts, self.in_features, r) * 2e-2)
            self.lora_B[adapter_name] = nn.Parameter(torch.zeros(num_experts, r, self.out_features))
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
            # TODO: is this the correct initialization?
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


class Linear(nn.Linear, MoLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        num_experts: int = 1,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        super(nn.Linear, self).__init__()
        MoLoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_router = Router(in_features, num_experts)

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts)
        self.set_adapter(adapter_name)


    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)


    def forward(self, x):
        previous_dtype = x.dtype

        #  Apply the forward pass of the original linear layer
        result = self._linear(x)
        # we only support a single active adapter
        active_adapter = self.active_adapters[0]

        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x = x.to(self.lora_A[active_adapter])

        # Routing mechanism
        router_probs = self.lora_router(x)
        x = dropout(x)

        # MoLoRa mechanism
        ax = torch.einsum("bsd,edr->bser", x, self.lora_A[active_adapter])
        bax = torch.einsum("bser,erd->bsed", ax, self.lora_B[active_adapter])
        lora_output = torch.einsum("...e,...ed->...d", router_probs, bax) * scaling

        # Add the output of the original linear layer
        result += lora_output
        result = result.to(previous_dtype)
        return lora_output

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


class SelfAttentionRouter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, bax):
        queries = self.query(x)
        keys = self.key(bax)
        values = self.value(x)

        scores = torch.matmul(queries.unsqueeze(2), keys.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
        attention = self.softmax(scores)
        # weighted = torch.matmul(attention, values.unsqueeze(-2))

        # Ensure attention scores and values have the correct dimensions
        attention = attention.squeeze(-2)  # Change shape to (batch_size, seq_len, seq_len)
        values = values.transpose(1, 2)  # Change shape to (batch_size, output_dim, seq_len)

        # Perform batch matrix multiplication
        weighted = torch.bmm(attention, values)  # Resulting shape: (batch_size, seq_len, output_dim)

        print(weighted.shape)
        print(weighted)
        return weighted


# class SelfAttentionRouter(nn.Module):
#     def __init__(self, input_dim, num_experts):
#         super(SelfAttentionRouter, self).__init__()
#         self.input_dim = input_dim
#         self.num_experts = num_experts

#         # Define linear layers for query for x
#         self.query = nn.Linear(input_dim, num_experts)

#         # Define linear layers for key and value for bax
#         self.key = nn.Linear(input_dim, num_experts)
#         self.value = nn.Linear(input_dim, num_experts)

#     def forward(self, x, bax):
#         # Compute query for x
#         qx = self.query(x)  # Shape: (batch_size, seq_len_x, num_experts)

#         # Compute key and value for bax
#         kbax = self.key(bax)  # Shape: (batch_size, seq_len_x, num_experts, num_experts)
#         vbax = self.value(bax)  # Shape: (batch_size, seq_len_x, num_experts, num_experts)

#         # Reshape and transpose for batched matrix multiplication
#         qx = qx.unsqueeze(2)  # Shape: (batch_size, seq_len_x, 1, num_experts)
#         kbax = kbax.transpose(-2, -3)  # Shape: (batch_size, seq_len_x, num_experts, num_experts)

#         # Calculate attention scores
#         attention_scores = torch.matmul(qx, kbax) / (self.num_experts ** 0.5)  # Shape: (batch_size, seq_len_x, 1, num_experts)

#         # Normalize attention scores
#         attention_probs = F.softmax(attention_scores, dim=-1)

#         # Compute weighted sum of values
#         expert_weights = torch.einsum('bsie,bsej->bsij', attention_probs, vbax)  # Shape: (batch_size, seq_len_x, 1, num_experts)

#         # Squeeze to remove the singleton dimension
#         expert_weights = expert_weights.squeeze(-2)

#         return expert_weights



class MoloraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["lora_A", "lora_B", "lora_router"]

    def __init__(self, in_features: int, out_features: int, num_experts: int, top_k: float, top_p: float, self_attn_router: bool, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.lora_router = nn.ModuleDict({})

        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.self_attn_router = self_attn_router
        if top_k > 0:
            self.top_k = top_k
        else:
            self.top_k = num_experts
        self.top_p = top_p
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

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

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts, self_attn_router):
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
            if self_attn_router:
                self.lora_router[adapter_name] = SelfAttentionRouter(self.in_features, num_experts)
            else:
                self.lora_router[adapter_name] = nn.Linear(self.in_features, num_experts)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, self_attn_router)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)


    def reset_lora_parameters(self, adapter_name, self_attn_router=False):
        if adapter_name in self.lora_A.keys():
            # initialize each expert using kaiming_uniform_
            for i in range(self.lora_A[adapter_name].shape[0]):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][i])
                if self_attn_router:
                    nn.init.kaiming_uniform_(self.lora_router[adapter_name].query.weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.lora_router[adapter_name].key.weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.lora_router[adapter_name].value.weight, a=math.sqrt(5))
                else:
                    nn.init.kaiming_uniform_(self.lora_router[adapter_name].weight, a=math.sqrt(5))


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


    def top_p_routing(self, logits, top_p = 1.0):
        # From: https://github.com/lucidrains/DALLE-pytorch/issues/318
        # Look at https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317 as well
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits


    def top_k_routing(self, logits, top_k):
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Linear, MoloraLayer):
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
        top_k: int = 0,
        top_p: float = 0.0,
        self_attn_router: bool = False,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        MoloraLayer.__init__(self, in_features=in_features, out_features=out_features, num_experts=num_experts, top_k=top_k, top_p=top_p, self_attn_router=self_attn_router, **kwargs)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts)
        self.set_adapter(adapter_name)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights by averaging all the experts.
        TODO: Should we average or just add?

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
            )
        active_adapter = self.active_adapters[0]

        if active_adapter in self.lora_A.keys():
            for expert in range(self.num_experts):
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter, expert) / self.num_experts

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The expert {expert} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter, expert) / self.num_experts
            self.merged_adapters.append(active_adapter)


    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        active_adapter = self.merged_adapters.pop()
        for expert in range(self.num_experts):
            if active_adapter in self.lora_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter, expert) * self.num_experts

    # Does this makes sense?
    def get_delta_weight(self, adapter, expert) -> torch.Tensor:
        return (
            transpose(
                self.lora_B[adapter][expert] @ self.lora_A[adapter][expert],
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self._linear(x)

        # The molora model only supports one active adapter at a time
        active_adapter = self.active_adapters[0]
        if active_adapter in self.lora_A.keys():
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_router = self.lora_router[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # Compute ax using einsum
            ax = torch.einsum("bsd,edr->bser", x, lora_A)
            ax = dropout(ax)
            # Compute bax using einsum
            bax = torch.einsum("bser,erd->bsed", ax, lora_B)

            if self.self_attn_router:
                expert_weights = lora_router(x, bax)
            else:
                if self.num_experts > 1:
                    # Compute expert_weights using the routing layer
                    logits = lora_router(x)

                    # Top-k routing
                    if self.top_k < self.num_experts:
                        logits = self.top_k_routing(logits, self.top_k)

                    # Top-p routing
                    if self.top_p > 0.0:
                        logits = self.top_p_routing(logits, self.top_p)

                    expert_weights = F.softmax(logits, dim=-1)
                else:
                    # initialize expert_weights to 1 as we only have one expert
                    expert_weights = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)

            # Combine using router probabilities
            output = torch.einsum("...e,...ed->...d", expert_weights, bax) * scaling

            result += output

        result = result.to(previous_dtype)
        return result

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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class SelfAttentionRouter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 8, use_value: bool = False):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(output_dim, hidden_dim)
        if use_value:
            self.value = nn.Linear(output_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / (self.hidden_dim ** 0.5)
        self.use_value = use_value

    def l2_normalize(self, x, axis: int = -1, eps: float = 1e-6):
        m = torch.rsqrt(torch.square(x).sum(dim=axis, keepdim=True) + eps)
        return x * m

    def forward(self, x, bax):
        # x: [batch_size, seq_len, input_dim]
        # bax: [batch_size, seq_len, num_experts, output_dim]
        x = self.l2_normalize(x, axis=-1)
        bax = self.l2_normalize(bax, axis=-1)

        queries = self.query(x)  # [batch_size, seq_len, hidden_dim]
        keys = self.key(bax)  # [batch_size, seq_len, num_experts, hidden_dim]
        if self.use_value:
            values = self.value(bax)  # [batch_size, seq_len, num_experts, output_dim]

        # Transpose for attention computation
        keys_transposed = keys.transpose(-2, -1)  # [batch_size, seq_len, hidden_dim, num_experts]

        # Compute scaled dot product attention
        scores = torch.einsum('bsh,bshn->bsn', queries, keys_transposed) * self.scale
        attention = F.softmax(scores, dim=-1)  # [batch_size, seq_len, num_experts]

        # print(f"attention: {attention}")
        # Apply attention scores to values
        if self.use_value:
            weighted = torch.einsum('bsn,bsne->bse', attention, values)  # [batch_size, seq_len, output_dim]
            return weighted
        else:
            return attention


class DotProductRouter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, bax):
        # x: [batch_size, seq_len, input_dim]
        # bax: [batch_size, seq_len, num_experts, output_dim]
        input_dim = x.shape[-1]
        scale = 1.0 / (input_dim ** 0.5)

        # Compute scaled dot product between x and each expert in bax
        # Fix: Ensure the dimensions of x and bax are compatible for einsum
        x_expanded = x.unsqueeze(-2)  # [batch_size, seq_len, 1, input_dim]
        # Ensure bax is viewed correctly to align with x_expanded for dot product
        bax = bax.view(bax.shape[0], bax.shape[1], bax.shape[2], -1)  # [batch_size, seq_len, num_experts, ?]
        # Fix: Adjust einsum subscripts to account for dot product across input_dim
        scores = torch.einsum('bsid,bsed->bse', x_expanded, bax) * scale
        attention = F.softmax(scores, dim=-1)  # [batch_size, seq_len, num_experts]
        return attention


# class SelfAttentionRouter(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 4):
#         super().__init__()
#         self.query = nn.Linear(input_dim, hidden_dim)
#         self.key = nn.Linear(output_dim, hidden_dim)
#         self.value = nn.Linear(output_dim, output_dim)
#         self.hidden_dim = hidden_dim
#         self.scale = 1.0 / (self.hidden_dim ** 0.5)

#     def forward(self, x, bax):
#         # x: [batch_size, seq_len, input_dim]
#         # bax: [batch_size, seq_len, num_experts, output_dim]
#         queries = self.query(x)  # [batch_size, seq_len, hidden_dim]
#         keys = self.key(bax)  # [batch_size, seq_len, num_experts, hidden_dim]
#         values = self.value(bax)  # [batch_size, seq_len, num_experts, output_dim]

#         # Transpose for attention computation
#         keys_transposed = keys.transpose(-2, -1)  # [batch_size, seq_len, hidden_dim, num_experts]

#         # Compute scaled dot product attention
#         scores = torch.einsum('bsh,bshn->bsn', queries, keys_transposed) * self.scale
#         attention = F.softmax(scores, dim=-1)  # [batch_size, seq_len, num_experts]

#         # Apply attention scores to values
#         weighted = torch.einsum('bsn,bsne->bse', attention, values)  # [batch_size, seq_len, output_dim]

#         return weighted


class MoloraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "router_dropout")

    def __init__(
        self,
        base_layer: nn.Module,
        num_experts: int,
        top_k: float,
        top_p: float,
        self_attn_router: bool,
        random_routing: bool,
        uniform_routing: bool,
        **kwargs
    ) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.lora_router = nn.ModuleDict({})
        self.router_dropout = nn.ModuleDict({})

        self._disable_adapters = False
        self.merged_adapters = []
        self.num_experts = num_experts
        self.self_attn_router = self_attn_router
        self.random_routing = random_routing
        self.uniform_routing = uniform_routing
        if top_k > 0:
            self.top_k = top_k
        else:
            self.top_k = num_experts
        self.top_p = top_p
        self.kwargs = kwargs

        base_layer = self.get_base_layer()

        in_features, out_features = base_layer.in_features, base_layer.out_features

        self.in_features = in_features
        self.out_features = out_features


    def update_layer(
            self,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            num_experts,
            self_attn_router,
            self_attn_hidden_dim,
            self_attn_use_value,
            router_dropout,
        ):
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
                self.lora_router[adapter_name] = SelfAttentionRouter(self.in_features, self.out_features, self_attn_hidden_dim, self_attn_use_value)
            else:
                self.lora_router[adapter_name] = nn.Linear(self.in_features, num_experts)

            if router_dropout > 0.0:
                router_dropout_layer = nn.Dropout(p=router_dropout)
            else:
                router_dropout_layer = nn.Identity()

            self.router_dropout.update(nn.ModuleDict({adapter_name: router_dropout_layer}))

            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, self_attn_router)

        weight = getattr(self.get_base_layer(), "weight", None)
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


class Linear(nn.Module, MoloraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_experts: int = 1,
        top_k: int = 0,
        top_p: float = 0.0,
        self_attn_router: bool = False,
        self_attn_hidden_dim: int = 4,
        self_attn_use_value: bool = False,
        random_routing: bool = False,
        uniform_routing: bool = False,
        router_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        super().__init__()
        MoloraLayer.__init__(
            self,
            base_layer,
            num_experts=num_experts,
            top_k=top_k,
            top_p=top_p,
            self_attn_router=self_attn_router,
            random_routing=random_routing,
            uniform_routing=uniform_routing,
            router_dropout=router_dropout,
            **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            num_experts,
            self_attn_router,
            self_attn_hidden_dim,
            self_attn_use_value,
            router_dropout,
        )
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

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)

        # The molora model only supports one active adapter at a time
        active_adapter = self.active_adapters[0]
        if active_adapter in self.lora_A.keys():
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_router = self.lora_router[active_adapter]
            router_dropout = self.router_dropout[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # Compute ax using einsum
            ax = torch.einsum("bsd,edr->bser", x, lora_A)
            ax = dropout(ax)
            # Compute bax using einsum
            bax = torch.einsum("bser,erd->bsed", ax, lora_B)

            if self.self_attn_router:
                # assume we do not use value
                # output = lora_router(x, bax)
                expert_weights = lora_router(x, bax)
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)
                # output = torch.einsum('bsn,bsne->bse', attention, bax)  # [batch_size, seq_len, output_dim]

            elif self.random_routing:
                expert_weights = torch.rand(x.size(0), x.size(1), self.num_experts, device=x.device, dtype=x.dtype)
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            elif self.uniform_routing:
                expert_weights = torch.ones(x.size(0), x.size(1), self.num_experts, device=x.device, dtype=x.dtype) / self.num_experts
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            else: # linear routing
                if self.num_experts > 1:
                    # Compute expert_weights using the routing layer
                    logits = lora_router(x)

                    # Top-k routing
                    if self.top_k < self.num_experts:
                        logits = self.top_k_routing(logits, self.top_k)

                    # Top-p routing
                    if self.top_p > 0.0:
                        logits = self.top_p_routing(logits, self.top_p)

                    expert_weights = router_dropout(F.softmax(logits, dim=-1))
                else:
                    # initialize expert_weights to 1 as we only have one expert
                    expert_weights = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)

                # Combine using router probabilities
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            result += output * scaling

        result = result.to(previous_dtype)
        return result

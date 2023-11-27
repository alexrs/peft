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


import bitsandbytes as bnb
import torch
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

from .layer import MoloraLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, MoloraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
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
                )

            init_lora_weights = kwargs.pop("init_lora_weights", True)
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

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter] @ self.lora_A[adapter],
                    False,
                )
                * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            result = self.base_layer(x, *args, **kwargs)

            # The molora model only supports one active adapter at a time
            active_adapter = self.active_adapters[0]
            if active_adapter in self.lora_A.keys():
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_router = self.lora_router[active_adapter]
                router_dropout = self.lora_dropout[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

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
                # output = torch.einsum('bsn,bsne->bse', attention, bax)  # [batch_si

            elif self.random_routing:
                expert_weights = torch.rand(x.size(0), x.size(1), self.num_experts, device=x.device, dtype=x.dtype)
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            elif self.uniform_routing:
                expert_weights = torch.ones(x.size(0), x.size(1), self.num_experts, device=x.device, dtype=x.dtype) / self.num_experts
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

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

                    expert_weights = router_dropout(F.softmax(logits, dim=-1))
                else:
                    # initialize expert_weights to 1 as we only have one expert
                    expert_weights = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
                # Combine using router probabilities
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            if requires_conversion:
                output = output.to(expected_dtype)
            result += output * scaling

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "molora." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, MoloraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
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
            )

            # Freezing the pre-trained weight matrix
            # self.get_base_layer().weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
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

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter] @ self.lora_A[adapter],
                    False,
                )
                * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            result = self.base_layer(x, *args, **kwargs)
            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            # The molora model only supports one active adapter at a time
            active_adapter = self.active_adapters[0]
            if active_adapter in self.lora_A.keys():
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_router = self.lora_router[active_adapter]
                router_dropout = self.lora_dropout[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

            # Compute ax using einsum
            ax = torch.einsum("bsd,edr->bser", x, lora_A)
            ax = dropout(ax)
            # Compute bax using einsum
            bax = torch.einsum("bser,erd->bsed", ax, lora_B)

            if self.self_attn_router:
                # assume we do not use value
                output = lora_router(x, bax)
                # expert_weights = lora_router(x, bax)
                # output = torch.einsum("...e,...ed->...d", expert_weights, bax)
                # output = torch.einsum('bsn,bsne->bse', attention, bax)  # [batch_si

            elif self.random_routing:
                expert_weights = torch.rand(x.size(0), x.size(1), self.num_experts, device=x.device, dtype=x.dtype)
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            elif self.uniform_routing:
                expert_weights = torch.ones(x.size(0), x.size(1), self.num_experts, device=x.device, dtype=x.dtype) / self.num_experts
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

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

                    expert_weights = router_dropout(F.softmax(logits, dim=-1))
                else:
                    # initialize expert_weights to 1 as we only have one expert
                    expert_weights = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)

                # Combine using router probabilities
                output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            if requires_conversion:
                output = output.to(expected_dtype)

            result += output * scaling

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "molora." + rep
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

import warnings

import bitsandbytes as bnb
import torch
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

from .layer import MoloraLayer


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, MoloraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            num_experts: int = 1,
            **kwargs,
        ) -> None:
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            MoloraLayer.__init__(self, in_features=in_features, out_features=out_features, num_experts=num_experts)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts)
            self.set_adapter(adapter_name)


        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter] @ self.lora_A[adapter],
                    False,
                )
                * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = super().forward(x)

            # The molora model only supports one active adapter at a time
            active_adapter = self.active_adapters[0]
            if active_adapter in self.lora_A.keys():
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_router = self.lora_router[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

                # Compute expert_weights using the routing layer
                logits = lora_router(x)
                expert_weights = F.softmax(logits, dim=-1)

                # Compute ax using einsum
                ax = torch.einsum('bsi,eij->bsej', x, lora_A)
                ax = dropout(ax)

                # Compute bax using einsum
                bax = torch.einsum('bsej,ejk->bske', ax, lora_B)

                # Combine using router probabilities
                output = torch.einsum('...e,...ek->...k', expert_weights, bax)

                if requires_conversion:
                    output = output.to(expected_dtype)
                result += output * scaling

            return result


if is_bnb_4bit_available():

    class Linear4bit(bnb.nn.Linear4bit, MoloraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            num_experts: int = 1,
            **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            MoloraLayer.__init__(self, in_features=in_features, out_features=out_features, num_experts=num_experts)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts)
            self.set_adapter(adapter_name)

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter] @ self.lora_A[adapter],
                    False,
                )
                * self.scaling[adapter]
            )


        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = super().forward(x)
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
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

                # Compute expert_weights using the routing layer
                logits = lora_router(x)
                expert_weights = F.softmax(logits, dim=-1)

                # Compute ax using einsum
                ax = torch.einsum('bsi,eij->bsej', x, lora_A)
                ax = dropout(ax)

                # Compute bax using einsum
                bax = torch.einsum('bsej,ejk->bske', ax, lora_B)

                # Combine using router probabilities
                output = torch.einsum('...e,...ek->...k', expert_weights, bax)

                if requires_conversion:
                    output = output.to(expected_dtype)
                result += output * scaling

            return result

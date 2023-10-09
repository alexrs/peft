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

import torch
import torch.nn.functional as F

from peft.tuners.molora.layer import MoloraLayer


class QuantLinear(torch.nn.Module, MoloraLayer):
    def __init__(
        self,
        adapter_name,
        quant_linear_module,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        num_experts: int = 1,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        MoloraLayer.__init__(
            self,
            in_features=quant_linear_module.infeatures,
            out_features=quant_linear_module.outfeatures,
            num_experts=num_experts,
        )
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, num_experts=num_experts)
        self.set_adapter(adapter_name)

    def forward(self, x: torch.Tensor):
        # note: logic differs from default Linear because merging is not supported
        result = self.quant_linear_module(x)

        if self.disable_adapters:
            return result

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
                x = x.to(lora_A.dtype)

            # Compute expert_weights using the routing layer
            logits = lora_router(x)
            expert_weights = F.softmax(logits, dim=-1)

            # Compute ax using einsum
            ax = torch.einsum("bsd,edr->bser", x, lora_A)
            ax = dropout(ax)
            # Compute bax using einsum
            bax = torch.einsum("bser,erd->bsed", ax, lora_B)
            # Combine using router probabilities
            output = torch.einsum("...e,...ed->...d", expert_weights, bax)

            if requires_conversion:
                output = output.to(expected_dtype)
            result += output * scaling

        return result

    # TODO: Check if it is better as suggested by users https://github.com/PanQiWei/AutoGPTQ/pull/102
    # def reset_lora_parameters(self, adapter_name):
    #     if adapter_name in self.lora_A.keys():
    #         torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
    #         torch.nn.init.zeros_(self.lora_B[adapter_name].weight)

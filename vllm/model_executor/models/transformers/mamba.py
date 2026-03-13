# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
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
"""Transformers modeling backend mixin for Mixture of Experts (MoE) models."""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config.utils import getattr_iter
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from .utils import log_replacement

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@CustomOp.register("transformers_mamba_mixer")
class TransformersMambaMixer(MambaMixer2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if hidden_states.ndim == 3:
            # Flatten to 2D
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        return super().forward(hidden_states, mup_vector=None)

class MambaMixerMixin(IsHybrid):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"A_log": "A"},
    )

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super(IsHybrid, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def recursive_replace(self):
        """Initialize the Mamba mixer layers."""
        text_config = self.text_config

        hidden_size = text_config.hidden_size
        ssm_state_size = text_config.ssm_state_size
        conv_kernel = text_config.conv_kernel
        mamba_num_heads = text_config.mamba_num_heads
        mamba_head_dim = text_config.mamba_head_dim
        use_conv_bias = text_config.use_conv_bias
        use_bias = text_config.use_bias
        n_groups = text_config.n_groups
        layer_norm_epsilon = text_config.layer_norm_epsilon
        mamba_hidden_act = text_config.mamba_hidden_act

        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                if child_name == "mixer" and module.block_type == "mamba":
                    # This is a mamba NemotronHBlock
                    nemotron_h_block = module
                    # Mamba mixer layer
                    mamba_mixer = TransformersMambaMixer(
                        hidden_size=hidden_size,
                        ssm_state_size=ssm_state_size,
                        conv_kernel_size=conv_kernel,
                        intermediate_size=mamba_num_heads * mamba_head_dim,
                        use_conv_bias=use_conv_bias,
                        use_bias=use_bias,
                        n_groups=n_groups,
                        num_heads=mamba_num_heads,
                        head_dim=mamba_head_dim,
                        rms_norm_eps=layer_norm_epsilon,
                        activation=mamba_hidden_act,
                        model_config=self.model_config,
                        cache_config=self.cache_config,
                        quant_config=self.quant_config,
                        prefix=qual_name,
                    )
                    nemotron_h_block.mixer = mamba_mixer
                    log_replacement(qual_name, child_module, mamba_mixer)
                else:
                    _recursive_replace(child_module, prefix=qual_name)

        _recursive_replace(self.model, prefix="model")

        super().recursive_replace()

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        text_config = hf_config.get_text_config()
        intermediate_size = text_config.mamba_num_heads * text_config.mamba_head_dim

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=text_config.n_groups,
            num_heads=text_config.mamba_num_heads,
            head_dim=text_config.mamba_head_dim,
            state_size=text_config.ssm_state_size,
            conv_kernel=text_config.conv_kernel,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

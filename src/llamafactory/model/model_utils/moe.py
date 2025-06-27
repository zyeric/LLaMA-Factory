# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Union

from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.misc import check_version
import torch
import torch.nn.functional as F
from grouped_gemm.ops import gmm


if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


def _set_z3_leaf_modules(model: "PreTrainedModel", leaf_modules: list[Union["nn.Module", str]]) -> None:
    check_version("deepspeed>=0.13.0")
    from deepspeed.utils import set_z3_leaf_modules  # type: ignore

    set_z3_leaf_modules(model, leaf_modules)


def add_z3_leaf_module(model: "PreTrainedModel") -> None:
    r"""Set module as a leaf module to skip partitioning in deepspeed zero3."""
    if not is_deepspeed_zero3_enabled():
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type == "dbrx":
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN

        _set_z3_leaf_modules(model, [DbrxFFN])

    if model_type == "deepseek_v2":
        # deepseek v2 uses custom code
        _set_z3_leaf_modules(model, ["DeepseekV2MoE"])

    if model_type == "deepseek_v3" or model_type == "kimi_vl":
        # deepseek v3 and kimi vl use custom code
        _set_z3_leaf_modules(model, ["DeepseekV3MoE"])

    if model_type == "granitemoe":
        from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

        _set_z3_leaf_modules(model, [GraniteMoeMoE])

    if model_type == "jamba":
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock

        _set_z3_leaf_modules(model, [JambaSparseMoeBlock])

    if model_type == "jetmoe":
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE

        _set_z3_leaf_modules(model, [JetMoeMoA, JetMoeMoE])

    if model_type == "llama4":
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        _set_z3_leaf_modules(model, [Llama4TextMoe])

    if model_type == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        _set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "olmoe":
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

        _set_z3_leaf_modules(model, [OlmoeSparseMoeBlock])

    if model_type == "phimoe":
        from transformers.models.phimoe.modeling_phimoe import PhimoeSparseMoeBlock

        _set_z3_leaf_modules(model, [PhimoeSparseMoeBlock])

    if model_type == "qwen2_moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])

    if model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])


def configure_moe(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.moe_aux_loss_coef:
        return

    model_type = getattr(config, "model_type", None)
    if model_type in [
        "dbrx",
        "granitemoe",
        "jamba",
        "jetmoe",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "output_router_logits", True)

    if model_type in ["granitemoe", "jamba", "llama4", "mixtral", "olmoe", "phimoe", "qwen2_moe", "qwen3_moe"]:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

    elif model_type == "deepseek":
        setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

    elif model_type == "jetmoe":
        setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each
       token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of
                                [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the
                                        capacity factor, should equal the number of tokens not
                                        dropped. By default, set to None, meaning no tokens are
                                        dropped.
        padded_mode (bool, optional): If True, indicating the indices are padded to
                                      [num_expert, capacity] to denote selected tokens per expert.
                                      Defaults to False.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        indices = indices.unsqueeze(1)

    topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    moe_gather_indices = (sorted_indices // topk).unsqueeze(1).expand(-1, tokens.size(-1))
    permuted_tokens = moe_gather.apply(tokens, moe_gather_indices)

    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the
    tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): 2D tensor [num_tokens*topk, hidden]. The tensor of permuted
                                        tokens to be unpermuted.
        sorted_indices (torch.Tensor): 1D tensor [num_tokens*topk]. The tensor of sorted indices
                                       used to unpermute the tokens.
        probs (torch.Tensor, optional): 2D tensor [num_tokens, topk]. The tensor of probabilities
                                        corresponding to the permuted tokens. If provided,
                                        the unpermuted tokens will be merged with their respective
                                        probabilities.
        padded_mode (bool, optional): If True, indicating the indices are padded to
                                      [num_expert, capacity] to denote selected tokens per expert.
                                      Defaults to False.
        restore_shape (torch.Size, optional): The input shape before permutation, only used in
                                              padding mode. Defaults to None.

    Returns:
        torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
    """
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(
        0
    ), f"Got {sorted_indices.numel()} != {permuted_tokens.size(0)}."
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        assert probs.dim() == 2, f"Expected 2D tensor for probs, got {probs.dim()} dims."
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    output_size = [num_unpermuted_tokens, permuted_tokens.shape[-1]]
    moe_scatter_indices = sorted_indices.unsqueeze(1).expand(-1, permuted_tokens.size(-1))
    unpermuted_tokens = moe_scatter.apply(permuted_tokens, moe_scatter_indices, output_size)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def permute_with_padded_tokens(tokens, indices):
    """Permute the tokens based on the indices, only used in padding mode.
       The input indices shape is [num_expert, capacity], it indicates which tokens were selected
       by each expert separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected
                                tokens for each expert.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

    return permuted_tokens, indices


def unpermute_with_padded_tokens(
    permuted_tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermutes a padded permuted tokens based on sorted indices and merges the tokens with their
    corresponding probabilities.

    This function takes a tensor of permuted tokens and reorders them according to the provided
    indices. It also combines the tokens with their associated probabilities.

    Parameters:
        permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected
                                tokens for each expert.
        probs (torch.Tensor): A tensor with the same shape as indices, containing probabilities
                              corresponding to each token.
        restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.

    Returns:
        torch.Tensor: A tensor of unpermuted tokens, merged with their probabilities.

    """
    # Ensure permuted_tokens is 2D
    assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."

    # Reshape and expand probabilities and indices to match permuted_tokens
    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    assert (
        permuted_tokens.shape == indices.shape
    ), "Shape mismatch between permuted_tokens and indices."

    # Combine tokens with their probabilities
    combined_output = probs * permuted_tokens

    # Prepare a tensor of zeros with the desired output shape
    empty_tokens = torch.zeros(
        restore_shape, dtype=combined_output.dtype, device=combined_output.device
    )

    # Scatter the combined tokens back to their original positions
    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)

    return unpermuted_tokens


class moe_gather(torch.autograd.Function):
    """Gather the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_):
        """Gather the input tensor based on the map tensor."""
        ctx.input_size = input_.size()
        ctx.map = map_
        return torch.gather(input_, 0, map_)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter the grad_output tensor based on the map tensor."""
        input_size = ctx.input_size
        map_ = ctx.map

        output = torch.zeros(
            input_size, dtype=grad_output.dtype, device=torch.cuda.current_device()
        )
        output.scatter_add_(0, map_, grad_output)
        return output, None, None


class moe_scatter(torch.autograd.Function):
    """Scatter the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        """Scatter the input tensor based on the map tensor."""
        ctx.map = map_

        if output_size is not None:
            output = torch.zeros(output_size, dtype=input_.dtype, device=input_.device)
        else:
            output = torch.zeros_like(input_)

        output.scatter_add_(0, map_, input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gather the grad_output tensor based on the map tensor."""
        map_ = ctx.map
        grad_input = torch.gather(grad_output, 0, map_)
        return grad_input, None, None, None


def nnscaler_moe_gmm(
    hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor,
    gate_projs: torch.Tensor, up_projs: torch.Tensor, down_projs: torch.Tensor,
    n_routed_experts: int, local_expert_start: int, local_expert_end: int):
    
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])

    with torch.no_grad():
        local_mask = (topk_idx >= local_expert_start) & (topk_idx < local_expert_end)
        local_idx = topk_idx.masked_select(local_mask)

    local_prob = topk_weight.masked_select(local_mask)
    local_prob = local_prob.view(-1, 1)
    local_map = local_mask.nonzero()[:, 0]
    local_map = local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
    local_hidden_states = moe_gather.apply(hidden_states, local_map)

    with torch.no_grad():
        tokens_per_expert = torch.histc(local_idx, bins=local_expert_end - local_expert_start, min=local_expert_start, max=local_expert_end - 1)
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

    permuted_inputs, row_id_map = permute(local_hidden_states, local_idx)

    fc1_output = gmm(permuted_inputs, gate_projs, tokens_per_expert, trans_b=True)
    fc2_output = gmm(permuted_inputs, up_projs, tokens_per_expert, trans_b=True)
    intermediate_parallel = torch.nn.functional.silu(fc1_output) * fc2_output
    expert_outs = gmm(intermediate_parallel, down_projs, tokens_per_expert, trans_b=True)

    y = unpermute(expert_outs, row_id_map)
    y = y * local_prob
    y = moe_scatter.apply(y, local_map, hidden_states.shape)

    y = y.to(hidden_states.dtype).view(*orig_shape)

    return y


def moe_forward(self, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    gate_projs = torch.stack([expert.gate_proj.weight for expert in self.experts], dim=0)
    up_projs = torch.stack([expert.up_proj.weight for expert in self.experts], dim=0)
    down_projs = torch.stack([expert.down_proj.weight for expert in self.experts], dim=0)

    final_hidden_states = nnscaler_moe_gmm(
        hidden_states,
        selected_experts,
        routing_weights,
        gate_projs,
        up_projs,
        down_projs,
        8,
        0,
        128
    )
    return final_hidden_states, router_logits

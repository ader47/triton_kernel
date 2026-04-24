import random

import torch
import torch_npu

import triton
import triton.language as tl
torch.set_printoptions(threshold=torch.inf)


def get_cache_miss_topk_indices(
        req_ids_tensor: torch.Tensor,  # [num_reqs], int64, used to distinguish topk_idx from different requests
        topk_indices_old: torch.Tensor,  # [num_reqs, topk(2048)], int64
        topk_indices_new: torch.Tensor,  # [num_reqs, topk(2048)], int32
):
    """
    remove the cache hit (already in topk_indices_old) idx from topk_indices_new,
    only keep the cache miss part for following npu/cpu loading.
    for example,
    old: [1, 2, 3, 4],
    new: [1, 3, 5, 7],
    ret: [-1, 5, -1, 7],
    """

    def get_set_diff_mask(a: torch.tensor, b: torch.tensor) -> torch.Tensor:
        # only consider a.shape == b.shape == [bs, topk]
        assert a.shape == b.shape
        assert a.ndim == 2
        comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1)  # [bs, topk, topk]
        intersect_mask = comparison_mask.any(-1)  # [bs, topk]
        return ~intersect_mask

    # to distinguish tokens of different reqs, add a req_ids_offset
    # maybe betther to use torch.bitwise_left_shift, but seems not supported on npu
    req_ids_offset = (req_ids_tensor * (1 << 16)).unsqueeze(-1)
    topk_indices_new = torch.where(topk_indices_new >= 0, topk_indices_new + req_ids_offset, -1)

    # tokens in new but not in old, which is cache miss and need to load
    cache_miss_token_mask = get_set_diff_mask(topk_indices_new, topk_indices_old) & (topk_indices_new >= 0)
    # tokens in old but not in new, which is useless now
    available_slot_mask = get_set_diff_mask(topk_indices_old, topk_indices_new) & (topk_indices_old >= 0)

    num_tokens_to_load = cache_miss_token_mask.sum(dim=1)
    num_available_slot = available_slot_mask.sum(dim=1)
    num_shortage_slot = num_tokens_to_load - num_available_slot
    # this part is needed while seq_len < 2k, num_shortage_slot > 0,
    # so there are multiple empty slots (idx == -1) in old topk_idx,
    # we also pick these empty slots to store cache miss tokens.
    num_shortage_slot = num_shortage_slot.unsqueeze(1)
    empty_slot_mask = topk_indices_old == -1
    empty_slot_cumsum = torch.cumsum(empty_slot_mask, dim=1)
    selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
    available_slot_mask = torch.where(selected_empty_slot_mask, True, available_slot_mask)

    topk_indices_to_load_flattened = topk_indices_new[cache_miss_token_mask]
    topk_indices_new.fill_(-1)
    topk_indices_new[available_slot_mask] = topk_indices_to_load_flattened

    # update history topk_indices for next step usage
    topk_indices_old[...] = torch.where(available_slot_mask, topk_indices_new, topk_indices_old)

    # recover topk_indices (remove req offset)
    topk_indices_new = torch.where(topk_indices_new >= 0, topk_indices_new - req_ids_offset, -1)

    return topk_indices_new.to(torch.int32)


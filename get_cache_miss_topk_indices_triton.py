import random

import torch
import torch_npu

import triton
import triton.language as tl


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


@triton.jit
def get_cache_miss_topk_kernel1(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        out_ptr,
        num_reqs,
        topk: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid)
    req_offset = req_id * 65536
    # Triton 的指针算术是 按元素偏移 的，不是按字节偏移
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    # 只处理当前 req 的 topk 个 token
    mask = cols < topk

    old = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)

    # add req offset to new, ignore -1

    new_with_offset = tl.where(new >= 0, new + req_offset, -1)
    # ---- cache miss mask: new not in old ----
    new_b = tl.broadcast_to(new_with_offset[:, None], (BLOCK, BLOCK))
    old_b = tl.broadcast_to(old[None, :], (BLOCK, BLOCK))

    cmp_new_old = (
                new_b == old_b)  # [BLOCK, BLOCK]

    # replace tl.any(cmp_new_old, axis=1) → sum > 0
    has_match_in_old = tl.sum(cmp_new_old.to(tl.int32),
                              axis=1) > 0  # [BLOCK]
    miss_mask = (~has_match_in_old) & (
                new_with_offset >= 0)  # [BLOCK]

    # ---- available slot mask: old not in new ----
    old_b2 = tl.broadcast_to(old[:, None], (BLOCK, BLOCK))
    new_b2 = tl.broadcast_to(new_with_offset[None, :], (BLOCK, BLOCK))
    cmp_old_new = (
                old_b2 == new_b2)  # [BLOCK, BLOCK]

    # replace tl.any(cmp_old_new, axis=1) → sum > 0
    has_match_in_new = tl.sum(cmp_old_new.to(tl.int32),
                              axis=1) > 0  # [BLOCK]
    avail_mask = (~has_match_in_new) & (
                old >= 0)  # [BLOCK]

    # ---- shortage: fill empty slots in old ----
    num_tokens_to_load = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_available_slot = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    empty_mask = (old == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage_slot) & empty_mask
    avail_mask = avail_mask | selected_empty

    # ---- compact miss tokens into available slots ----
    # 在新的矩阵上，看哪些值没有命中
    miss_vals = tl.where(miss_mask, new_with_offset,
                         0)  # [BLOCK]
    # rank of each available slot (0-based)
    avail_rank = tl.cumsum(avail_mask.to(tl.int32),
                           axis=0) - 1  # [BLOCK]
    # rank of each miss token (0-based)
    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1  # [BLOCK]

    # for each available slot i, pick miss_vals[avail_rank[i]]
    # broadcast: avail_rank[:, None] == miss_rank[None, :]  -> [BLOCK, BLOCK]
    avail_rank_b = tl.broadcast_to(avail_rank[:, None], (BLOCK, BLOCK))

    miss_rank_b = tl.broadcast_to(miss_rank[None, :], (BLOCK, BLOCK))
    miss_vals_b = tl.broadcast_to(miss_vals[None, :], (BLOCK, BLOCK))
    rank_match = (avail_rank_b == miss_rank_b) & (
                miss_rank_b >= 0)  # [BLOCK, BLOCK]
    # for each row i, at most one column j matches → gather
    matched_vals = tl.sum(
        tl.where(rank_match, miss_vals_b, tl.zeros((BLOCK, BLOCK), tl.int32)),
        axis=1
    )  # [BLOCK]
    out = tl.where(avail_mask, matched_vals, tl.full((BLOCK,), -1, tl.int32))
    # ---- remove req offset and store ----
    out = tl.where(out >= 0, out - req_offset, tl.full((BLOCK,), -1, tl.int32))
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)

    # ---- update old in-place ----
    # avail_mask 位置写入 new_with_offset（用 matched_vals + offset 反推不方便，
    # 直接用带 offset 的 out 版本）
    new_val_with_offset = tl.where(out >= 0, out + req_offset, tl.full((BLOCK,), -1, tl.int32))
    updated_old = tl.where(avail_mask, new_val_with_offset, old)
    tl.store(old_ptr + row_off + cols, updated_old, mask=mask)
@triton.jit
def get_cache_miss_topk_kernel(
    req_ids_ptr,
    old_ptr,
    new_ptr,
    out_ptr,
    debug_ptr,
    num_reqs,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid)
    req_offset = req_id * 65536
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_with_offset = tl.where(new >= 0, new + req_offset, -1)
    if pid == 0:
        tl.store(debug_ptr + 5 * topk + cols, new_with_offset.to(tl.int32), mask=mask)
    # ---- sub-blocked miss_mask: new not in old ----
    miss_count = tl.zeros((BLOCK,), tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        old_chunk = tl.load(
            old_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int32)
        old_b = tl.broadcast_to(old_chunk[None, :], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_with_offset[:, None], (BLOCK, SUB_BLOCK))
        cmp = new_b == old_b
        miss_count += tl.sum(cmp.to(tl.int32), axis=1)
    miss_mask = (miss_count == 0) & (new_with_offset >= 0)

    # ---- sub-blocked avail_mask: old not in new ----
    avail_count = tl.zeros((BLOCK,), tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        new_chunk = tl.load(
            new_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int32)
        new_chunk_off = tl.where(new_chunk >= 0, new_chunk + req_offset, -1)
        old_b = tl.broadcast_to(old[:, None], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_chunk_off[None, :], (BLOCK, SUB_BLOCK))
        cmp = old_b == new_b
        avail_count += tl.sum(cmp.to(tl.int32), axis=1)
    avail_mask = (avail_count == 0) & (old >= 0)

    # ---- shortage: fill empty slots ----
    num_tokens_to_load = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_available_slot = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    empty_mask = old == -1
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage_slot) & empty_mask
    avail_mask = avail_mask | selected_empty

    # ---- compact: scatter miss values into available slots ----
    miss_vals = tl.where(miss_mask, new_with_offset, 0)
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    if pid == 0:
        tl.store(debug_ptr + 0 * topk + cols, miss_mask.to(tl.int32), mask=mask)
        tl.store(debug_ptr + 1 * topk + cols, avail_mask.to(tl.int32), mask=mask)
        tl.store(debug_ptr + 2 * topk + cols, miss_rank, mask=mask)
        tl.store(debug_ptr + 3 * topk + cols, avail_rank, mask=mask)
        tl.store(debug_ptr + 4 * topk + cols, miss_vals, mask=mask)

    # print("find success")

    # Gather-then-scatter: split by SUB_BLOCK chunks of target rank
    # Phase 1 (gather): for each target rank r in [sb_start, sb_start+SUB_BLOCK),
    #            find miss_vals where miss_rank == r
    # Phase 2 (scatter): for each available slot where avail_rank == r,
    #            write the gathered value
    out = tl.full((BLOCK,), -1, tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        target_ranks = sb_start + tl.arange(0, SUB_BLOCK)

        # Phase 1: gather - [BLOCK, SUB_BLOCK]
        mr_b = tl.broadcast_to(miss_rank[:, None], (BLOCK, SUB_BLOCK))
        tr_b = tl.broadcast_to(target_ranks[None, :], (BLOCK, SUB_BLOCK))
        mv_b = tl.broadcast_to(miss_vals[:, None], (BLOCK, SUB_BLOCK))
        mm_b = tl.broadcast_to(miss_mask[:, None], (BLOCK, SUB_BLOCK))

        miss_match = (mr_b == tr_b) & mm_b
        gathered = tl.sum(
            tl.where(miss_match, mv_b, tl.zeros((BLOCK, SUB_BLOCK), tl.int32)),
            axis=0,
        )

        # Phase 2: scatter - [BLOCK, SUB_BLOCK]
        ar_b = tl.broadcast_to(avail_rank[:, None], (BLOCK, SUB_BLOCK))
        am_b = tl.broadcast_to(avail_mask[:, None], (BLOCK, SUB_BLOCK))
        valid_rank = tr_b < num_miss

        slot_match = (ar_b == tr_b) & am_b & valid_rank
        result = tl.sum(
            tl.where(
                slot_match,
                gathered[None, :],
                tl.zeros((BLOCK, SUB_BLOCK), tl.int32),
            ),
            axis=1,
        )
        has_match = tl.sum(slot_match.to(tl.int32), axis=1) > 0
        out = tl.where(has_match, result, out)

    # ---- remove req offset and store ----
    out = tl.where(out >= 0, out - req_offset, tl.full((BLOCK,), -1, tl.int32))
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)

    # ---- update old in-place ----
    new_val_with_offset = tl.where(
        out >= 0, out + req_offset, tl.full((BLOCK,), -1, tl.int32)
    )
    updated_old = tl.where(avail_mask, new_val_with_offset, old)
    tl.store(old_ptr + row_off + cols, updated_old, mask=mask)

def get_cache_miss_topk_indices_triton(
        req_ids_tensor: torch.Tensor,
        topk_indices_old: torch.Tensor,
        topk_indices_new: torch.Tensor,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    out = torch.empty_like(topk_indices_new, dtype=torch.int32)
    grid = (num_reqs,)
    # 为什么需要 2 的幂？ Triton 的 tl.arange(0, BLOCK) 和 tl.broadcast_to 等 API 要求 BLOCK 是编译期常量且为 2 的幂，
    # 这样 GPU/NPU 可以高效地按 warp（32 线程）分配工作。
    BLOCK = triton.next_power_of_2(topk)
    sub_block = 1
    SUB_BLOCK = min(triton.next_power_of_2(sub_block), BLOCK)
    debug = torch.zeros((6, topk), dtype=torch.int32)

    get_cache_miss_topk_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        out,
        debug,
        num_reqs,
        topk=topk,
        BLOCK=BLOCK,
        SUB_BLOCK=SUB_BLOCK,
    )
    print("miss_mask :", debug[0].cpu().numpy())
    print("avail_mask:", debug[1].cpu().numpy())
    print("miss_rank :", debug[2].cpu().numpy())
    print("avail_rank:", debug[3].cpu().numpy())
    print("miss_vals :", debug[4].cpu().numpy())
    print("out       :", debug[5].cpu().numpy())
    return out


def demo():
    # 超参数
    num_reqs = 4  # 请求数量
    topk = 16  # 每个请求保留的 top‑k 索引数

    # 请求 ID（int32），这里使用 npu 设备
    device = 'npu:15'
    req_ids = torch.arange(num_reqs, dtype=torch.int32, device=device)

    # 旧的 top‑k 索引（int32），模拟缓存中已有的 token 索引（不带 req_offset）
    old_indices = torch.randint(0, 1000, (num_reqs, topk), dtype=torch.int32, device=device)
    # 随机将一些位置设为 -1（空槽）
    old_indices[torch.rand(num_reqs, topk, device=device) > 0.7] = -1
    # 添加offset 到old indices中
    req_offsets = (req_ids * 65536).unsqueeze(-1)
    old_indices = torch.where(old_indices >= 0, old_indices + req_offsets, -1)
    # 新的 top‑k 索引（int32），当前请求要写入的 token 索引（不带 req_offset）
    new_indices = torch.randint(0, 1000, (num_reqs, topk), dtype=torch.int32, device=device)
    # 新索引中也可能包含 -1（表示无效 token）
    new_indices[torch.rand(num_reqs, topk, device=device) > 0.8] = -1

    print("old_indices (before):")
    print(old_indices)
    print("\nnew_indices:")
    print(new_indices)


    # 调用 Triton kernel
    out = get_cache_miss_topk_indices_triton(req_ids, old_indices, new_indices)

    print("\nout (indices that can be loaded from cache miss, in original order):")
    print(out)

    # 注意：old_indices 会被 kernel 原地更新（因为 kernel 内部对 old_ptr 执行了 store）
    print("\nold_indices (after update):")
    print(old_indices)


# if __name__ == "__main__":
demo()

exit()
device = 'npu'

# req_ids_tensor = torch.tensor([1, 2, 3, 5], dtype=torch.int64, device=device)
# topk_indices_old = torch.tensor(
#     [[4 + 1 * 65536, 2 + 1 * 65536, 5 + 1 * 65536, 6 + 1 * 65536],
#      [4 + 2 * 65536, 2 + 2 * 65536, 1 + 2 * 65536, -1],
#      [2 + 3 * 65536, 7 + 3 * 65536, -1, -1],
#      [1 + 4 * 65536, 2 + 4 * 65536, 3 + 4 * 65536, 4 + 4 * 65536]],
#     dtype=torch.int64,
#     device=device,
# )
# topk_indices_new = torch.tensor(
#     [[1, 2, 3, 4],
#      [1, 2, 3, 4],
#      [1, 2, 3, -1],
#      [1, 2, 3, 4]],
#     dtype=torch.int32,
#     device=device,
# )
# gold = torch.tensor(
#     [[-1, -1,  1,  3],
#      [-1, -1, -1,  3],
#      [-1,  1,  3, -1],
#      [ 1,  2,  3,  4]],
#     dtype=torch.int32,
#     device=device,
# )

req_ids_tensor = torch.tensor([0], dtype=torch.int64, device=device)
token_indices = random.sample(list(range(8 * 1024)), 2048);
token_indices.sort()
topk_indices_old = torch.tensor(token_indices, dtype=torch.int64, device=device).unsqueeze(0)

token_indices = random.sample(list(range(8 * 1024)), 2048);
token_indices.sort()
topk_indices_new = torch.tensor(token_indices, dtype=torch.int32, device=device).unsqueeze(0)

print(f'>>>>> topk_idx old = {topk_indices_old}')
print(f'>>>>> topk_idx new = {topk_indices_new}')
gold = get_cache_miss_topk_indices(req_ids_tensor, topk_indices_old.clone(), topk_indices_new.clone())
print(f'>>>>> gold = {gold}')
ret = get_cache_miss_topk_indices_triton(req_ids_tensor, topk_indices_old.clone(), topk_indices_new.clone())
print(f'>>>>> ret = {ret}')
print(f'>>>>> equal = {torch.equal(ret, gold)}')

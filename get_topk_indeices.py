import torch
import torch_npu

import triton
import triton.language as tl

@triton.jit
def mark_cache_tokens_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        num_reqs,
        stamp,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int32)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    stamp_i32 = stamp.to(tl.int32)
    stamp_vals = tl.full((BLOCK,), 0, tl.int32) + stamp_i32
    tl.store(old_marker_ptr + marker_off + old_token, stamp_vals, mask=old_valid)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    tl.store(new_marker_ptr + marker_off + new_token, stamp_vals, mask=new_valid)


@triton.jit
def compact_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        stamp,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int32)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_with_offset = new_token + req_offset

    old_hit = tl.load(old_marker_ptr + marker_off + new_token, mask=new_valid, other=0)
    new_hit = tl.load(new_marker_ptr + marker_off + old_token, mask=old_valid, other=0)

    stamp_i32 = stamp.to(tl.int32)
    miss_mask = new_valid & (old_hit != stamp_i32)
    avail_mask = old_valid & (new_hit != stamp_i32)

    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_avail = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage = num_miss - num_avail

    empty_mask = mask & (old_with_offset == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    num_slots = tl.sum(avail_mask.to(tl.int32), axis=0)

    tl.store(slot_scratch_ptr + row_off + avail_rank, cols, mask=avail_mask)
    tl.store(miss_scratch_ptr + row_off + miss_rank, new_with_offset, mask=miss_mask)
    tl.store(miss_count_ptr + pid, num_miss)
    tl.store(slot_count_ptr + pid, num_slots)


@triton.jit
def apply_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        out_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int32)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    tl.store(out_ptr + row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)

    num_miss = tl.load(miss_count_ptr + pid).to(tl.int32)
    num_slots = tl.load(slot_count_ptr + pid).to(tl.int32)
    update_mask = cols < num_slots
    miss_mask = cols < num_miss
    slots = tl.load(slot_scratch_ptr + row_off + cols, mask=update_mask, other=0).to(tl.int32)
    miss_with_offset = tl.load(miss_scratch_ptr + row_off + cols, mask=miss_mask, other=-1).to(tl.int32)
    miss_token = miss_with_offset - req_offset

    tl.store(old_ptr + row_off + slots, miss_with_offset, mask=update_mask)
    tl.store(out_ptr + row_off + slots, miss_token, mask=miss_mask)


@triton.jit
def get_cache_miss_topk_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        out_ptr,
        num_reqs,
        topk: tl.constexpr,
        BLOCK: tl.constexpr,
        SUB_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int32)
    req_offset = req_id * 65536
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_with_offset = tl.where(new >= 0, new + req_offset, -1)
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

    # Gather-then-scatter: split by SUB_BLOCK chunks of target rank
    # Phase 1 (gather): for each target rank r in [sb_start, sb_start+SUB_BLOCK),
    #            find miss_vals where miss_rank == r
    # Phase 2 (scatter): for each available slot where avail_rank == r,
    #            write the gathered value
    out_with_offset = tl.full((BLOCK,), -1, tl.int32)
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
        out_with_offset = tl.where(has_match, result, out_with_offset)

    # ---- update old in-place ----
    updated_old = tl.where(avail_mask, out_with_offset, old)
    tl.store(old_ptr + row_off + cols, updated_old, mask=mask)

    # ---- remove req offset and store ----
    out = tl.where(out_with_offset >= 0, out_with_offset - req_offset, tl.full((BLOCK,), -1, tl.int32))
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)


def get_cache_miss_topk_indices_triton_bitmap(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    token_limit: int = 65536,
    old_marker: torch.Tensor | None = None,
    new_marker: torch.Tensor | None = None,
    slot_scratch: torch.Tensor | None = None,
    miss_scratch: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    slot_count: torch.Tensor | None = None,
    stamp: int = 1,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    out = torch.empty_like(topk_indices_new, dtype=torch.int32)
    if old_marker is None:
        old_marker = torch.zeros(
            (num_reqs, token_limit),
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if new_marker is None:
        new_marker = torch.zeros(
            (num_reqs, token_limit),
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if slot_scratch is None:
        slot_scratch = torch.empty_like(topk_indices_new, dtype=torch.int32)
    if miss_scratch is None:
        miss_scratch = torch.empty_like(topk_indices_new, dtype=torch.int32)
    if miss_count is None:
        miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=topk_indices_new.device)
    if slot_count is None:
        slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=topk_indices_new.device)

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    mark_cache_tokens_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        old_marker,
        new_marker,
        num_reqs,
        stamp,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    compact_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        old_marker,
        new_marker,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        stamp,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    apply_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        out,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    return out


def get_cache_miss_topk_indices_triton_exact(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    out = torch.empty_like(topk_indices_new, dtype=torch.int32)

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    get_cache_miss_topk_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        out,
        num_reqs,
        topk=topk,
        BLOCK=BLOCK,
        SUB_BLOCK=1
    )
    return out


def get_cache_miss_topk_indices_triton(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    **kwargs,
):
    return get_cache_miss_topk_indices_triton_bitmap(
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        **kwargs,
    )


def prepare_cache_miss_scratch(owner, num_reqs: int, topk: int, device, token_limit: int = 65536):
    marker_shape = (num_reqs, token_limit)
    scratch_shape = (num_reqs, topk)
    needs_alloc = (
        getattr(owner, "_cache_miss_token_limit", None) != token_limit
        or getattr(owner, "_cache_miss_device", None) != device
        or getattr(owner, "_cache_miss_marker_shape", (0, 0))[0] < num_reqs
        or getattr(owner, "_cache_miss_scratch_shape", (0, 0))[0] < num_reqs
        or getattr(owner, "_cache_miss_scratch_shape", (0, 0))[1] < topk
    )

    if needs_alloc:
        owner._cache_miss_old_marker = torch.zeros(marker_shape, dtype=torch.int32, device=device)
        owner._cache_miss_new_marker = torch.zeros(marker_shape, dtype=torch.int32, device=device)
        owner._cache_miss_slot_scratch = torch.empty(scratch_shape, dtype=torch.int32, device=device)
        owner._cache_miss_miss_scratch = torch.empty(scratch_shape, dtype=torch.int32, device=device)
        owner._cache_miss_miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
        owner._cache_miss_slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
        owner._cache_miss_token_limit = token_limit
        owner._cache_miss_device = device
        owner._cache_miss_marker_shape = marker_shape
        owner._cache_miss_scratch_shape = scratch_shape
        owner._cache_miss_stamp = 0

    owner._cache_miss_stamp += 1
    if owner._cache_miss_stamp >= 2_000_000_000:
        owner._cache_miss_old_marker.zero_()
        owner._cache_miss_new_marker.zero_()
        owner._cache_miss_stamp = 1

    return {
        "token_limit": token_limit,
        "stamp": owner._cache_miss_stamp,
        "old_marker": owner._cache_miss_old_marker[:num_reqs],
        "new_marker": owner._cache_miss_new_marker[:num_reqs],
        "slot_scratch": owner._cache_miss_slot_scratch[:num_reqs, :topk],
        "miss_scratch": owner._cache_miss_miss_scratch[:num_reqs, :topk],
        "miss_count": owner._cache_miss_miss_count[:num_reqs],
        "slot_count": owner._cache_miss_slot_count[:num_reqs],
    }


def _get_topk_buffer(
    self,
    topk_indices: torch.Tensor,       # [num_tokens, 1, max_seq_len]
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    attn_metadata: M,               
    layer_name: str,
    block_table: torch.Tensor,        # [num_tokens, max_blocks]
    seq_len_kv: torch.Tensor,
):
    forward_context: ForwardContext = get_forward_context()
    num_reqs = topk_indices.shape[0]
    topk_buffer_k = kv_cache[3][:num_reqs]
    topk_buffer_v = kv_cache[4][:num_reqs]
    topk_indices = topk_indices.squeeze(1) # TODO maybe consider dim1 (head_num?)

    # cache reuse
    # num_tokens_ori = (topk_indices >= 0).sum().item()
    # topk_indices = self.get_cache_miss_topk_indices(
    #     attn_metadata.req_ids_tensor[:num_reqs],
    #     self.last_step_topk_indices[:num_reqs],
    #     topk_indices,
    # )

    t1 = time.time()
    cache_miss_scratch = prepare_cache_miss_scratch(
        self,
        num_reqs,
        topk_indices.shape[1],
        topk_indices.device,
    )
    topk_indices = get_cache_miss_topk_indices_triton(
        attn_metadata.req_ids_tensor[:num_reqs],
        self.last_step_topk_indices[:num_reqs],
        topk_indices,
        **cache_miss_scratch,
    )
    t2 = time.time()
    print(f">>>>>>>>>>> get_cache_miss_topk_indices_triton {(t2-t1)*1000:.2f}ms")
    num_tokens_cache_miss = (topk_indices >= 0).sum().item()

    # common
    t1 = time.time()
    valid_mask = topk_indices >= 0
    num_offloaded_blocks = attn_metadata.num_offloaded_blocks[:num_reqs].unsqueeze(1)
    offload_thresholds = num_offloaded_blocks * self.block_size
    npu_mask = (topk_indices >= offload_thresholds) & valid_mask
    cpu_mask = (topk_indices < offload_thresholds) & valid_mask
    t2 = time.time()
    print(f">>>>>>>>>>> time 2 {(t2-t1)*1000:.2f}ms")

    # num_tokens_npu = npu_mask.sum().item()
    # num_tokens_cpu = cpu_mask.sum().item()

    # load npu
    t1 = time.time()
    block_indices = torch.clamp(topk_indices // self.block_size, min=0)
    block_ids = torch.gather(block_table, 1, block_indices)
    offsets_in_block = topk_indices % self.block_size
    npu_mask = npu_mask.unsqueeze(-1).unsqueeze(-1)
    topk_buffer_k[...] = torch.where(npu_mask, kv_cache[0][block_ids, offsets_in_block], topk_buffer_k)
    topk_buffer_v[...] = torch.where(npu_mask, kv_cache[1][block_ids, offsets_in_block], topk_buffer_v)
    t2 = time.time()
    print(f">>>>>>>>>>> time 3 {(t2-t1)*1000:.2f}ms")
    # load cpu
    cpu_token_indices = torch.where(cpu_mask, topk_indices, -1)
    # maybe_load_kv_token_wise_graph(layer_name, num_reqs, cpu_token_indices, cpu_mask, forward_context.capturing)

    # generate new block_table & indices
    t1 = time.time()
    topk_buffer_k = topk_buffer_k.reshape([-1, self.block_size, 1, 512])
    topk_buffer_v = topk_buffer_v.reshape([-1, self.block_size, 1, 64])
    sparse_block_table = self.sparse_block_table[:num_reqs]
    sparse_seq_len_kv = torch.clamp(seq_len_kv, max=2048)
    sparse_topk_indices = self.sparse_topk_indices[:num_reqs]
    sparse_topk_indices = torch.where(sparse_topk_indices < sparse_seq_len_kv.unsqueeze(1), sparse_topk_indices, -1)
    sparse_topk_indices = sparse_topk_indices.unsqueeze(1)
    t2 = time.time()
    print(f">>>>>>>>>>> time 4 {(t2-t1)*1000:.2f}ms")
    return (topk_buffer_k, topk_buffer_v), sparse_topk_indices, sparse_block_table, sparse_seq_len_kv

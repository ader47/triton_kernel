def build_test_cases():
    """
    构造覆盖各种边界场景的测试数据：
    1. 完全无交集：old 和 new 完全不同 → 全部 miss
    2. 完全相同：old 和 new 一样 → 全部 hit，输出全 -1
    3. 部分交集：有 hit 有 miss
    4. old 有空槽(-1)：miss token 可以填入空槽
    5. new 有 -1：无效 token 不参与比较
    6. 多 req：不同 req_id 的 token 不会混淆
    7. 大数据：topk=2048 验证分块正确性
    """
    device = "npu"
    cases = {}

    # ---- Case 1: 完全无交集 ----
    # old=[0,1,2,3], new=[4,5,6,7] → 全部 miss，输出 [4,5,6,7]
    cases["no_overlap"] = {
        "req_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "old": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        "new": torch.tensor([[4, 5, 6, 7]], dtype=torch.int32, device=device),
        "desc": "完全无交集：全部 miss",
    }

    # ---- Case 2: 完全相同 ----
    # old=[0,1,2,3], new=[0,1,2,3] → 全部 hit，输出全 -1
    cases["full_overlap"] = {
        "req_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "old": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        "new": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        "desc": "完全相同：全部 hit",
    }

    # ---- Case 3: 部分交集 ----
    # old=[0,1,2,3], new=[1,2,4,5] → miss={4,5}, avail={0,3}, 输出 [-1,-1,4,5]
    cases["partial_overlap"] = {
        "req_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "old": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        "new": torch.tensor([[1, 2, 4, 5]], dtype=torch.int32, device=device),
        "desc": "部分交集：有 hit 有 miss",
    }

    # ---- Case 4: old 有空槽 ----
    # old=[0,-1,2,-1], new=[3,4,0,5] → miss={3,4,5}, avail={-1,-1,2}+empty={-1,-1}
    #   miss_vals = [3,4,5] (去掉 hit 的 0)
    #   avail slots = 位置1(-1), 位置3(-1), 位置2(old=2不在new中)
    #   输出: [-1, 3, 5, 4] (位置0 hit, 位置1空槽填3, 位置2 avail填5, 位置3空槽填4)
    cases["old_has_empty_slots"] = {
        "req_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "old": torch.tensor([[0, -1, 2, -1]], dtype=torch.int32, device=device),
        "new": torch.tensor([[3, 4, 0, 5]], dtype=torch.int32, device=device),
        "desc": "old 有空槽：miss token 填入空槽",
    }

    # ---- Case 5: new 有 -1 ----
    # old=[0,1,2,3], new=[1,-1,4,-1] → miss={4}, avail={0,2,3}
    #   输出: [-1, -1, 4, -1] (只有位置2是avail且对应miss_rank=0)
    cases["new_has_invalid"] = {
        "req_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "old": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        "new": torch.tensor([[1, -1, 4, -1]], dtype=torch.int32, device=device),
        "desc": "new 有 -1：无效 token 不参与比较",
    }

    # ---- Case 6: 多 req，不同 req_id ----
    # req0: old=[0,1], new=[2,3] → miss={2,3}, avail={0,1}
    # req1: old=[0,1], new=[0,2] → miss={2}, avail={1}
    # req_id 不同，token 0 在 req0 和 req1 中不会混淆
    cases["multi_req"] = {
        "req_ids": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "old": torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device),
        "new": torch.tensor([[2, 3], [0, 2]], dtype=torch.int32, device=device),
        "desc": "多 req：不同 req_id 的 token 不混淆",
    }

    # ---- Case 7: old 全空 ----
    # old=[-1,-1,-1,-1], new=[0,1,2,3] → 全部 miss，全部填入空槽
    cases["old_all_empty"] = {
        "req_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "old": torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32, device=device),
        "new": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        "desc": "old 全空：全部 miss 填入空槽",
    }

    # ---- Case 8: new 全 -1 ----
    # old=[0,1,2,3], new=[-1,-1,-1,-1] → 无 miss，输出全 -1
    cases["new_all_invalid"] = {
        "req_ids": torch.tensor([0], dtype=torch.int32, device=device),
        "old": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device),
        "new": torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32, device=device),
        "desc": "new 全 -1：无 miss",
    }

    # ---- Case 9: 大数据 topk=2048 ----
    num_reqs = 2
    topk = 2048
    token_range = 8192
    req_ids_big = torch.arange(num_reqs, dtype=torch.int32, device=device)

    old_big = torch.zeros((num_reqs, topk), dtype=torch.int32, device=device)
    new_big = torch.zeros((num_reqs, topk), dtype=torch.int32, device=device)
    for r in range(num_reqs):
        # old: 随机 token，约 20% 空槽
        old_tokens = random.sample(range(token_range), topk)
        old_row = torch.tensor(old_tokens, dtype=torch.int32)
        empty_mask = torch.rand(topk) > 0.8
        old_row[empty_mask] = -1
        old_big[r] = old_row

        # new: 随机 token，约 10% -1，和 old 有部分重叠
        new_tokens = random.sample(range(token_range), topk)
        new_row = torch.tensor(new_tokens, dtype=torch.int32)
        invalid_mask = torch.rand(topk) > 0.9
        new_row[invalid_mask] = -1
        new_big[r] = new_row

    cases["large_topk_2048"] = {
        "req_ids": req_ids_big,
        "old": old_big,
        "new": new_big,
        "desc": f"大数据：num_reqs={num_reqs}, topk={topk}",
    }

    return cases


def run_all_tests():
    cases = build_test_cases()
    passed = 0
    failed = 0

    for name, data in cases.items():
        req_ids = data["req_ids"]
        old = data["old"].clone()
        new = data["new"].clone()
        desc = data["desc"]

        # 给 old 加 req_offset（和 demo 一样）
        req_offsets = (req_ids * 65536).unsqueeze(-1)
        old_with_offset = torch.where(old >= 0, old + req_offsets, -1)

        # PyTorch 参考实现
        gold = get_cache_miss_topk_indices(
            req_ids.clone(), old_with_offset.clone(), new.clone()
        )

        # Triton 实现
        try:
            ret = get_cache_miss_topk_indices_triton(
                req_ids.clone(), old_with_offset.clone(), new.clone()
            )
            match = torch.equal(ret, gold)
        except Exception as e:
            match = False
            ret = f"ERROR: {e}"

        status = "✅ PASS" if match else "❌ FAIL"
        print(f"{status} [{name}] {desc}")

        if not match:
            failed += 1
            print(f"  gold: {gold}")
            print(f"  ret:  {ret}")
        else:
            passed += 1

    print(f"\n结果: {passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0


run_all_tests()
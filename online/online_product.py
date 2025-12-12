#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列生成 + 最终整数求解（多少套）版本
⚠ 单位全部使用 mm（原材料长度 y_mm）
"""

import math
import json
from collections import Counter
import sys

try:
    from pulp import (
        LpProblem, LpMinimize, LpVariable, lpSum, LpContinuous, LpInteger, PULP_CBC_CMD
    )
except Exception:
    print("需要安装 pulp: pip install pulp")
    raise


# ====================================================================================
#   Column Generation Optimizer
# ====================================================================================

class ColumnGenSetsOptimizer:
    def __init__(self, y_mm, a_head, b_tail, c_loss, specs, available_bars,
                 max_cg_iters=200, tol=1e-6):
        """
        参数单位均为 mm
        y_mm          母材原始长度 (mm)
        a_head        去头 (mm)
        b_tail        去尾 (mm)
        c_loss        每刀损耗 (mm)
        specs         [(length_mm, count_per_set), ...]
        available_bars 可用原材料根数
        """

        self.y_mm = int(y_mm)
        self.a = int(a_head)
        self.b = int(b_tail)
        self.c = int(c_loss)

        # 规格：长度降序
        self.specs = [(int(l), int(q)) for l, q in specs]
        self.specs.sort(key=lambda t: t[0], reverse=True)

        self.lengths = [s[0] for s in self.specs]
        self.per_set = [s[1] for s in self.specs]
        self.n = len(self.specs)

        self.available = int(available_bars)

        # 有效长度 L
        self.L = self.y_mm - self.a - self.b
        if self.L <= 0:
            raise ValueError("有效长度 L <= 0，请检查参数")

        # 背包容量模式
        self.cap = self.L + self.c
        self.max_per = [(self.L + self.c) // (l + self.c) if l > 0 else 0
                        for l in self.lengths]

        self.max_cg_iters = max_cg_iters
        self.tol = tol

    # ----------------------------------------------------
    # 初始列
    # ----------------------------------------------------
    def initial_patterns(self):
        pats = []

        # 单规格最大重复
        for i, l in enumerate(self.lengths):
            k = self.max_per[i]
            if k > 0:
                vec = [0]*self.n
                vec[i] = k
                pats.append(tuple(vec))

        # 贪心
        for start in range(self.n):
            vec = [0]*self.n
            rem = self.cap
            for i in range(start, self.n):
                w = self.lengths[i] + self.c
                if w <= 0:
                    continue
                k = rem // w
                if k > 0:
                    vec[i] = int(min(k, self.max_per[i]))
                    rem -= vec[i] * w
            if sum(vec) > 0:
                pats.append(tuple(vec))

        uniq, seen = [], set()
        for p in pats:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    # ----------------------------------------------------
    # DP 求 knapsack column
    # ----------------------------------------------------
    def knapsack_dp(self, duals):
        cap = self.cap
        n = self.n
        weights = [l + self.c for l in self.lengths]

        items = []
        for i in range(n):
            mi = self.max_per[i]
            if mi <= 0:
                continue
            v = duals[i]
            w = weights[i]

            # 二进制拆分
            k = 1
            rem = mi
            while rem > 0:
                take = min(k, rem)
                items.append((w * take, v * take, i, take))
                rem -= take
                k *= 2

        dp = [-1e100] * (cap + 1)
        dp[0] = 0.0
        parent = [None] * (cap + 1)
        item_used = [None] * (cap + 1)

        for idx, (wt, val, iidx, mult) in enumerate(items):
            wt = int(wt)
            for cur in range(cap, wt - 1, -1):
                prev = cur - wt
                if dp[prev] + val > dp[cur] + 1e-12:
                    dp[cur] = dp[prev] + val
                    parent[cur] = prev
                    item_used[cur] = idx

        best_w = max(range(cap + 1), key=lambda w: dp[w])
        if dp[best_w] < -1e50:
            return [0]*n, 0.0

        counts = [0]*n
        cur = best_w
        while cur != 0 and parent[cur] is not None:
            idx = item_used[cur]
            it = items[idx]
            counts[it[2]] += it[3]
            cur = parent[cur]

        return counts, dp[best_w]

    # ----------------------------------------------------
    # LP 主问题
    # ----------------------------------------------------
    def solve_master_lp(self, patterns, targets):
        m = len(patterns)
        prob = LpProblem("Master_LP", LpMinimize)

        x = [LpVariable(f"x_{j}", lowBound=0, cat=LpContinuous) for j in range(m)]
        prob += lpSum(x)

        for i in range(self.n):
            prob += lpSum(x[j] * patterns[j][i] for j in range(m)) >= targets[i], f"dem_{i}"

        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        if prob.status != 1:
            return None, None, None

        xvals = [v.varValue for v in x]
        duals = [prob.constraints[f"dem_{i}"].pi for i in range(self.n)]
        return sum(xvals), xvals, duals

    # ----------------------------------------------------
    # 整数主问题
    # ----------------------------------------------------
    def solve_master_int(self, patterns, targets):
        m = len(patterns)
        prob = LpProblem("Master_INT", LpMinimize)

        x = [LpVariable(f"x_{j}", lowBound=0, cat=LpInteger) for j in range(m)]
        prob += lpSum(x)

        for i in range(self.n):
            prob += lpSum(x[j] * patterns[j][i] for j in range(m)) >= targets[i]

        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        if prob.status != 1:
            return math.inf, {}

        usage = {patterns[j]: int(x[j].varValue) for j in range(m) if int(x[j].varValue) > 0}
        return sum(usage.values()), usage

    # ----------------------------------------------------
    # 主流程：二分求最大完整套数
    # ----------------------------------------------------
    def maximize_sets(self):
        # 快速上界
        hi_list = []
        for i in range(self.n):
            if self.per_set[i] > 0:
                hi_list.append((self.available * self.max_per[i]) // self.per_set[i])
        hi = min(hi_list) if hi_list else 0

        denom = sum(l * q for l, q in zip(self.lengths, self.per_set))
        if denom > 0:
            hi2 = (self.available * self.L) // denom
            hi = min(hi, hi2)

        lo, best, best_usage = 0, 0, {}

        print(f"🔍 最大完整套数二分查找范围: {lo} → {hi}")

        while lo <= hi:
            mid = (lo + hi) // 2
            targets = [mid * q for q in self.per_set]

            print(f"  ▸ 检查套数 {mid} ...", end="")

            feasible, bars_needed, usage = self._feasible_by_column_generation(targets)

            if feasible and bars_needed <= self.available:
                print(f" ✔ 可行，需要 {bars_needed} 根")
                best = mid
                best_usage = usage
                lo = mid + 1
            else:
                print(f" ✘ 不可行（需 {bars_needed} 根）")
                hi = mid - 1

        return best, best_usage

    # ----------------------------------------------------
    # 列生成（单次 feasibility check）
    # ----------------------------------------------------
    def _feasible_by_column_generation(self, targets):
        patterns = [tuple(p) for p in self.initial_patterns()]
        iters = 0

        while True:
            iters += 1
            if iters > self.max_cg_iters:
                print("（警告：列生成达到迭代上限）", end="")
                break

            lp_obj, xvals, duals = self.solve_master_lp(patterns, targets)
            if lp_obj is None:
                return False, math.inf, {}

            counts, val = self.knapsack_dp(duals)
            reduced_cost = val - 1.0

            if reduced_cost <= self.tol:
                break

            new_pat = tuple(min(counts[i], self.max_per[i]) for i in range(self.n))
            if sum(new_pat) == 0:
                break
            if new_pat not in patterns:
                patterns.append(new_pat)
            else:
                break

        bars_needed, usage = self.solve_master_int(patterns, targets)
        return bars_needed <= self.available, bars_needed, usage

    # ----------------------------------------------------
    # 根据 usage 展开为每根材料的切割列表（detail）
    # ----------------------------------------------------
    def expand_usage_to_bins(self, usage):
        bins = []
        for p, cnt in usage.items():
            for _ in range(cnt):
                lst = []
                for i, k in enumerate(p):
                    lst += [self.lengths[i]] * k
                bins.append(sorted(lst, reverse=True))
        return bins

    # ====================================================================================
    #   整理输出 JSON
    # ====================================================================================

    def solve_and_report(self):
        print("════════════════════════════════════════════")
        print("  ✨ 列生成 + 整数规划：最大套数求解开始")
        print("════════════════════════════════════════════")
        print(f"原材料长度 = {self.y_mm} mm，有效长度 = {self.L} mm，刀损 = {self.c} mm")
        print(f"可用母材数量 = {self.available}")
        print(f"规格列表 = {self.specs}")

        # 1. 主求解
        best_sets, usage = self.maximize_sets()

        # 2. 展开 bins
        bins = self.expand_usage_to_bins(usage)
        total_bars = len(bins)

        # 3. 统计
        total_used = sum(sum(b) for b in bins)
        total_cut_loss = sum(max(0, len(b)-1) * self.c for b in bins)
        total_waste = sum(self.L - sum(b) - max(0, len(b)-1)*self.c for b in bins)
        total_head_tail = (self.a + self.b) * total_bars
        utilization = (total_used / (total_bars * self.L) * 100) if total_bars else 0

        print("──────────────────────────────")
        print(f"最大套数 = {best_sets}")
        print(f"最优使用母材 = {total_bars} 根")
        print(f"利用率 = {utilization:.2f}%")
        print("──────────────────────────────")

        # 4. summary_plan
        summary_plan = []
        pattern_id = 1
        for p, cnt in usage.items():
            cutting_list = []
            for i, k in enumerate(p):
                cutting_list += [self.lengths[i]] * k
            cutting_list.sort(reverse=True)

            total_len = sum(cutting_list)
            cut_times = max(0, len(cutting_list)-1)
            pattern_cut_loss = cut_times * self.c
            waste = self.L - total_len - pattern_cut_loss
            util = (total_len / self.L * 100) if self.L > 0 else 0

            summary_plan.append({
                "pattern_id": pattern_id,
                "count": cnt,
                "cutting_list": cutting_list,
                "total_length": total_len,
                "cutting_times": cut_times,
                "cutting_loss": pattern_cut_loss,
                "waste": waste,
                "utilization": round(util, 2)
            })

            pattern_id += 1

        # 5. detail_plan
        detail_plan = []
        idx = 1
        for b in bins:
            total_len = sum(b)
            cut_times = max(0, len(b)-1)
            cut_loss = cut_times * self.c
            waste = self.L - total_len - cut_loss
            util = (total_len / self.L * 100) if self.L > 0 else 0

            detail_plan.append({
                "bar_number": idx,
                "original_length": self.y_mm,
                "head_cut": self.a,
                "tail_cut": self.b,
                "effective_length": self.L,
                "cutting_list": b,
                "total_used": total_len,
                "cutting_times": cut_times,
                "cutting_loss": cut_loss,
                "waste": waste,
                "utilization": round(util, 2)
            })
            idx += 1

        # 6. demand_verification
        produced = Counter()
        for b in bins:
            for item in b:
                produced[item] += 1

        verify_list = []
        all_ok = True
        for (l, q) in self.specs:
            actual = produced.get(l, 0)
            demand = best_sets * q
            ok = actual >= demand
            verify_list.append({
                "width": l,
                "demand": demand,
                "actual": actual,
                "satisfied": ok
            })
            if not ok:
                all_ok = False

        # ----------------------------------------------------
        # 最终 JSON
        # ----------------------------------------------------
        result = {
            "success": True,
            "message": "",
            "parameters": {
                "original_length": self.y_mm,
                "effective_length": self.L,
                "head_cut": self.a,
                "tail_cut": self.b,
                "cutting_loss": self.c,
                "demands": self.specs
            },
            "statistics": {
                "total_bars": total_bars,
                "utilization": round(utilization, 2),
                "total_waste": total_waste,
                "total_cutting_loss": total_cut_loss,
                "total_head_tail_cut": total_head_tail,
                "total_complete_sets": best_sets
            },
            "summary_plan": summary_plan,
            "detail_plan": detail_plan,
            "demand_verification": {
                "all_satisfied": all_ok,
                "details": verify_list
            }
        }

        return result


# ====================================================================================
# 示例运行
# ====================================================================================

if __name__ == "__main__":
    # 示例参数
    y = 6000
    a = 0
    b = 0
    c = 2
    specs = [         
        [143, 104],   # 每套需要 1430mm 零件 1 根
        [123, 202],   # 每套需要 1230mm 零件 2 根
        [1145, 17],   # 每套需要 1145mm 零件 1 根
        [1092, 21],    # 每套需要 1092mm 零件 2 根
        [92, 210],    # 每套需要 1092mm 零件 2 根
        [192, 21]    # 每套需要 1092mm 零件 2 根
    ]
    x_bars = 100

    opt = ColumnGenSetsOptimizer(y, a, b, c, specs, x_bars)
    result = opt.solve_and_report()

    print("\n== JSON 结果已生成（未全部打印）==\n")
    with open("sets_cut_result.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
    print("保存到 sets_cut_result.json")

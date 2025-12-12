#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列生成 + 最终整数求解（针对“多少套”问题）实现
保存为 cut_sets_column_generation.py 并运行
依赖: pip install pulp
"""

import math
import json
from collections import Counter
import sys

try:
    from pulp import (
        LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD, LpContinuous, LpInteger
    )
except Exception:
    print("需要安装 pulp: pip install pulp")
    raise

class ColumnGenSetsOptimizer:
    def __init__(self, y_meters, a_head, b_tail, c_loss, specs, available_bars,
                 max_cg_iters=200, tol=1e-6):
        self.y_meters = float(y_meters)
        self.a = int(a_head)
        self.b = int(b_tail)
        self.c = int(c_loss)
        self.specs = [(int(l), int(q)) for l, q in specs]
        self.specs.sort(key=lambda t: t[0], reverse=True)
        self.lengths = [s[0] for s in self.specs]
        self.per_set = [s[1] for s in self.specs]
        self.n = len(self.specs)
        self.available = int(available_bars)
        self.L = int(round(self.y_meters * 1000)) - self.a - self.b
        if self.L <= 0:
            raise ValueError("有效长度 L <= 0，请检查参数")
        # capacity transform: use capacity = L + c, item weight = length + c
        self.cap = self.L + self.c
        # max pieces of type i per bar
        self.max_per = [(self.L + self.c) // (l + self.c) if l>0 else 0 for l in self.lengths]

        self.max_cg_iters = max_cg_iters
        self.tol = tol

    # 初始列：每种单规格的最大重复 + 一些贪心组合
    def initial_patterns(self):
        pats = []
        # single-type max repeats
        for i, l in enumerate(self.lengths):
            k = self.max_per[i]
            if k > 0:
                vec = [0]*self.n
                vec[i] = k
                pats.append(tuple(vec))
        # 贪心：按长度降序填充
        for start in range(self.n):
            vec = [0]*self.n
            rem = self.cap
            for i in range(start, self.n):
                w = self.lengths[i] + self.c
                if w <= 0: continue
                k = rem // w
                if k > 0:
                    vec[i] = int(min(k, self.max_per[i]))
                    rem -= vec[i] * w
            if sum(vec) > 0:
                pats.append(tuple(vec))
        # 去重
        uniq = []
        seen = set()
        for p in pats:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    # 将 pattern tuple(counts) -> 真实长度 list
    def pattern_to_list(self, p):
        lst = []
        for i, c in enumerate(p):
            lst += [self.lengths[i]] * c
        return lst

    # solve knapsack subproblem (integer) to maximize sum(dual[i] * count_i)
    # subject to sum((length_i + c) * count_i) <= cap and 0<=count_i<=max_per[i]
    # returns best_count_vector and best_value
    def knapsack_dp(self, duals):
        cap = self.cap
        n = self.n
        weights = [l + self.c for l in self.lengths]
        # DP: dp[w] = best value; keep parent for reconstruct
        # But we also need to enforce per-item max counts; implement bounded knapsack by binary-splitting each item
        # Build items list of (weight, value, orig_index, multiplicity_unit)
        items = []
        for i in range(n):
            mi = self.max_per[i]
            if mi <= 0: continue
            v = duals[i]
            w = weights[i]
            # binary split
            k = 1
            rem = mi
            while rem > 0:
                take = min(k, rem)
                items.append((w * take, v * take, i, take))
                rem -= take
                k *= 2

        # dp arrays
        dp = [-1e100] * (cap + 1)
        dp[0] = 0.0
        parent = [None] * (cap + 1)  # store (prev_w, item_idx)
        item_used = [None] * (cap + 1)
        for idx, (wt, val, iidx, mult) in enumerate(items):
            wt = int(wt)
            # traverse backwards
            for cur in range(cap, wt - 1, -1):
                prev = cur - wt
                if dp[prev] + val > dp[cur] + 1e-12:
                    dp[cur] = dp[prev] + val
                    parent[cur] = prev
                    item_used[cur] = idx
        # find best weight
        best_w = max(range(cap + 1), key=lambda w: dp[w])
        best_val = dp[best_w]
        if best_val < -1e50:
            return [0]*n, 0.0
        # reconstruct counts
        counts = [0]*n
        cur = best_w
        while cur != 0 and parent[cur] is not None:
            idx = item_used[cur]
            it = items[idx]
            counts[it[2]] += it[3]
            cur = parent[cur]
        return counts, best_val

    # Solve LP master given current patterns (columns)
    def solve_master_lp(self, patterns, targets):
        # patterns: list of tuples length n
        m = len(patterns)
        prob = LpProblem("Master_LP", LpMinimize)
        x = [LpVariable(f"x_{j}", lowBound=0, cat=LpContinuous) for j in range(m)]
        prob += lpSum(x)
        # constraints for each type
        cons = []
        for i in range(self.n):
            prob += lpSum(x[j] * patterns[j][i] for j in range(m)) >= targets[i], f"dem_{i}"
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        if prob.status != 1:
            return None, None, None  # infeasible or failed
        xvals = [v.varValue for v in x]
        # extract duals
        duals = []
        for i in range(self.n):
            cname = f"dem_{i}"
            # pulp stores constraints in prob.constraints
            pi = prob.constraints[cname].pi if hasattr(prob.constraints[cname], 'pi') else prob.constraints[cname].pi
            duals.append(pi)
        obj = sum(xvals)
        return obj, xvals, duals

    # Solve integer master (min bars) with given patterns
    def solve_master_int(self, patterns, targets):
        m = len(patterns)
        prob = LpProblem("Master_INT", LpMinimize)
        x = [LpVariable(f"x_{j}", lowBound=0, cat=LpInteger) for j in range(m)]
        prob += lpSum(x)
        for i in range(self.n):
            prob += lpSum(x[j] * patterns[j][i] for j in range(m)) >= targets[i], f"dem_{i}"
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        if prob.status != 1:
            return math.inf, {}
        usage = {patterns[j]: int(x[j].varValue) for j in range(m) if int(x[j].varValue) > 0}
        total_bars = sum(usage.values())
        return total_bars, usage

    # The main: maximize complete sets via binary search + column generation per feasibility test
    def maximize_sets(self):
        # compute upper bound quickly
        upper_candidates = []
        for i in range(self.n):
            if self.per_set[i] > 0:
                upper_candidates.append((self.available * self.max_per[i]) // self.per_set[i])
        if upper_candidates:
            hi = min(upper_candidates)
        else:
            hi = 0
        denom = sum(l*q for l,q in zip(self.lengths, self.per_set))
        if denom > 0:
            hi2 = (self.available * self.L) // denom
            hi = min(hi, hi2)
        hi = max(hi, 0)
        lo = 0
        best = 0
        best_usage = {}
        # binary search
        while lo <= hi:
            mid = (lo + hi) // 2
            targets = [mid * q for q in self.per_set]
            print(f"  ↳ 检验套数 {mid} ... ", end='', flush=True)
            feasible, bars_needed, usage = self._feasible_by_column_generation(targets)
            if feasible and bars_needed <= self.available:
                print(f"可行，需要 {bars_needed} 根")
                best = mid
                best_usage = usage
                lo = mid + 1
            else:
                print(f"不可行 (需要 {bars_needed} 根)" if bars_needed<math.inf else "不可行 (无解)")
                hi = mid - 1
        # final best usage
        return best, best_usage

    def _feasible_by_column_generation(self, targets):
        # initialize patterns
        patterns = list(self.initial_patterns())
        # ensure patterns are tuples length n
        patterns = [tuple(p) for p in patterns]
        # limit iterations
        iters = 0
        while True:
            iters += 1
            if iters > self.max_cg_iters:
                print("（列生成迭代到达上限）", end=' ')
                break
            # solve master LP
            lp_obj, xvals, duals = self.solve_master_lp(patterns, targets)
            if lp_obj is None:
                return False, math.inf, {}
            # compute reduced cost subproblem: maximize sum(dual[i]*count_i) - 1
            counts, val = self.knapsack_dp(duals)
            reduced_cost = val - 1.0
            # debug
            # print(f" iter {iters} lp_obj {lp_obj:.4f} sub_val {val:.6f} reduced_cost {reduced_cost:.6f}")
            if reduced_cost <= self.tol:
                # LP optimal wrt available columns
                break
            # else add new pattern counts (but ensure bounded by max_per)
            new_pat = tuple(min(counts[i], self.max_per[i]) for i in range(self.n))
            if sum(new_pat) == 0:
                break
            if new_pat not in patterns:
                patterns.append(new_pat)
            else:
                # cannot improve
                break
        # now solve integer master with current patterns
        bars_needed, usage = self.solve_master_int(patterns, targets)
        feasible = (bars_needed <= self.available)
        return feasible, bars_needed, usage

    # helper to expand usage into bins and stats
    def expand_usage_to_bins(self, usage):
        bins = []
        for p, cnt in usage.items():
            for _ in range(cnt):
                lst = []
                for i,k in enumerate(p):
                    lst += [self.lengths[i]] * k
                bins.append(sorted(lst, reverse=True))
        return bins

    def solve_and_report(self):
        print("="*80)
        print("使用 列生成 + 最终整数求解（比全枚举更快）")
        print(f"母材原长 {self.y_meters} m -> 有效 L = {self.L} mm，刀损 c = {self.c} mm")
        print(f"规格: {self.specs}")
        print(f"可用母材: {self.available}")
        # binary search maximize
        best_sets, usage = self.maximize_sets()
        # expand
        bins = self.expand_usage_to_bins(usage)
        # stats
        total_bars = len(bins)
        total_used = sum(sum(b) for b in bins)
        total_loss = sum((len(b)-1) * self.c for b in bins)
        total_waste = sum(max(0, self.L - sum(b) - max(0, len(b)-1)*self.c) for b in bins)
        util = (total_used / (total_bars * self.L) * 100) if total_bars>0 else 0
        print("="*80)
        print(f"结果: 最大完整套数 = {best_sets}")
        print(f"最优使用母材数（近似/最终整数解）= {total_bars}")
        print(f"利用率 = {round(util,2)}%  切损总和 = {total_loss} mm  总浪费 = {total_waste} mm")
        print("模式使用：")
        for i,(p,cnt) in enumerate(usage.items(),1):
            parts = []
            for t_idx, k in enumerate(p):
                if k>0:
                    parts.append(f"{self.lengths[t_idx]}×{k}")
            print(f" 模式{i}: {' + '.join(parts)}  用 {cnt} 次")

        # verification
        produced = Counter()
        for b in bins:
            for item in b: produced[item] += 1
        verify_details = []
        all_ok = True
        for idx,l in enumerate(self.lengths):
            actual = produced.get(l,0)
            req = best_sets * self.per_set[idx]
            ok = actual >= req
            if not ok: all_ok = False
            verify_details.append({"length_mm": l, "actual_total": actual, "required_total_for_sets": req, "satisfied": ok})

        result = {
            "success": True,
            "parameters": {
                "mother_length_m": self.y_meters,
                "effective_length_mm": self.L,
                "head_cut_mm": self.a,
                "tail_cut_mm": self.b,
                "cut_loss_mm": self.c,
                "specs": [{"length_mm": l, "count_per_set": q} for l,q in self.specs],
                "available_bars": self.available
            },
            "total_complete_sets": best_sets,
            "statistics": {
                "total_bars_used": total_bars,
                "total_used_mm": total_used,
                "total_cut_loss_mm": total_loss,
                "total_waste_mm": total_waste,
                "utilization_percent": round(util,2)
            },
            "patterns_usage": [
                {"pattern_counts": {str(self.lengths[i]): p[i] for i in range(self.n)}, "count": cnt}
                for p,cnt in usage.items()
            ],
            "bins": [{"bar_number": i+1, "cutting_list": b} for i,b in enumerate(bins)],
            "verification": {"all_satisfied": all_ok, "details": verify_details}
        }
        return result

# Example usage:
if __name__ == "__main__":
    # 你原来的示例参数
    y = 6.0
    a = 0
    b = 0
    c = 2
    # specs = [
    #     (1596, 1),
    #     (851, 1),
    #     (123, 1),
    #     (65, 1),
    #     (50, 1)
    # ]
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
    res = opt.solve_and_report()
    print("\n== JSON 输出 ==\n")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open("sets_cut_columngen_result.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n结果已保存到 sets_cut_columngen_result.json")

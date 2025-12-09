import itertools
from typing import List, Tuple, Dict
from collections import Counter
import json
import sys

try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("错误: 未安装 pulp 库")
    print("安装方法: pip install pulp")
    exit(1)


class SteelCuttingOptimizer:
    """钢材下料优化器 - 用改进的模式生成替换原有暴力枚举，但保持原有输出格式与接口。"""

    def __init__(self, L: int, demands: List[List[int]], loss_mm: int = 5, 
                 head_cut: int = 0, tail_cut: int = 0,
                 max_combo_widths: int = 4, keep_top_patterns: int = 2000,
                 enum_limit: int = 500000):
        """
        参数说明：除了原有参数，增加可选参数以控制模式生成规模：
         - max_combo_widths: 每个模式允许组合的不同宽度种类数（默认4）
         - keep_top_patterns: 最终保留的高利用率模式数上限
         - enum_limit: 枚举组合时的总枚举上限（防止爆炸）
        """
        self.L_original = L  # 原始长度
        self.head_cut = head_cut
        self.tail_cut = tail_cut
        self.L = L - head_cut - tail_cut  # 有效长度
        self.demands = demands
        self.loss_mm = loss_mm
        self.max_combo_widths = max(1, min(max_combo_widths, 6))
        self.keep_top_patterns = max(50, keep_top_patterns)
        self.enum_limit = enum_limit

    def _max_count_for_width(self, width: int) -> int:
        if width <= 0:
            return 0
        # 推导 k*w + (k-1)*loss <= L -> k <= floor((L + loss) / (w + loss))
        return (self.L + self.loss_mm) // (width + self.loss_mm)

    def check_pattern(self, pattern: List[int]) -> bool:
        """检查切割模式是否有效（保持原有语义）"""
        if not pattern:
            return False
        total_length = sum(pattern)
        cuts = len([x for x in pattern if x > 0]) - 1
        total_with_loss = total_length + cuts * self.loss_mm
        return total_with_loss <= self.L

    def generate_all_patterns(self) -> List[Tuple[int, ...]]:
        """受控生成可行切割模式：单规格、受限多规格组合、贪心填充、按利用率筛选。

        目的：用更少的模式数量逼近最优解，同时避免原始暴力枚举导致的组合爆炸和长时间计算。
        返回值格式和原始函数一致（list of tuples）。
        """
        unique_widths = sorted(set([d[0] for d in self.demands]), reverse=True)
        patterns = set()

        # 单规格模式（把一种宽度重复到最大）
        max_counts = {w: self._max_count_for_width(w) for w in unique_widths}
        for w in unique_widths:
            max_k = max_counts[w]
            for k in range(1, max_k + 1):
                patt = tuple([w] * k)
                if self.check_pattern(list(patt)):
                    patterns.add(tuple(sorted(patt, reverse=True)))

        # 受限多规格组合（限制不同规格种类数，以及总枚举数enum_limit）
        enum_counter = 0
        for r in range(2, self.max_combo_widths + 1):
            for combo in itertools.combinations(unique_widths, r):
                ranges = [range(1, max_counts[w] + 1) for w in combo]
                for counts in itertools.product(*ranges):
                    enum_counter += 1
                    if enum_counter > self.enum_limit:
                        break
                    patt_list = []
                    for w, c in zip(combo, counts):
                        patt_list.extend([w] * c)
                    patt_tuple = tuple(sorted(patt_list, reverse=True))
                    if self.check_pattern(list(patt_tuple)):
                        patterns.add(patt_tuple)
                if enum_counter > self.enum_limit:
                    break
            if enum_counter > self.enum_limit:
                print("  ↳ 已达到组合枚举上限，提前停止多规格组合生成")
                break

        # 贪心填充：尝试将已有模式用较大规格填满剩余空间以获得更高利用率的变体
        base_patterns = list(patterns)
        for base in base_patterns:
            remaining = self.L - (sum(base) + (len(base) - 1) * self.loss_mm)
            if remaining <= 0:
                continue
            new_patt = list(base)
            placed = True
            while placed:
                placed = False
                for w in unique_widths:
                    add_cost = w + self.loss_mm
                    needed = add_cost if new_patt else w
                    if needed <= remaining:
                        new_patt.append(w)
                        remaining -= needed
                        placed = True
                        break
            new_tuple = tuple(sorted(new_patt, reverse=True))
            if self.check_pattern(list(new_tuple)):
                patterns.add(new_tuple)

        # 清洗：去掉明显无效、计算利用率并保留 top N
        def utilization(patt: Tuple[int, ...]) -> float:
            total = sum(patt)
            cuts = max(0, len(patt) - 1)
            used_with_loss = total + cuts * self.loss_mm
            return used_with_loss / self.L if self.L > 0 else 0.0

        patterns = [p for p in patterns if 0 < sum(p) <= self.L]
        # 过滤利用率过低的模式（保守阈值 0.2），避免丢失可行解但剔除极端低效模式
        patterns = [p for p in patterns if utilization(p) >= 0.20]
        # 按利用率降序并保留 top keep_top_patterns
        patterns.sort(key=lambda p: (utilization(p), -len(p)), reverse=True)
        if len(patterns) > self.keep_top_patterns:
            patterns = patterns[:self.keep_top_patterns]

        print(f"  → 生成切割模式数量：{len(patterns)}（受控生成）")
        return patterns

    def calculate_waste(self, pattern: List[int]) -> int:
        total_length = sum(pattern)
        cuts = len([x for x in pattern if x > 0]) - 1
        waste = self.L - total_length - cuts * self.loss_mm
        return waste

    def solve(self) -> Dict:
        """核心求解：保持原始函数签名与返回结构（result, debug）"""
        # 生成模式
        patterns = self.generate_all_patterns()
        if len(patterns) == 0:
            return ({
                "success": False,
                "message": "无法生成有效的切割模式"
            }, {})

        # 建模
        prob = LpProblem("Steel_Cutting_Stock", LpMinimize)
        pattern_vars = [LpVariable(f"pattern_{i}", lowBound=0, cat='Integer') 
                        for i in range(len(patterns))]
        prob += lpSum(pattern_vars), "Total_bars_used"

        demand_dict = {width: count for width, count in self.demands}
        unique_widths = sorted(demand_dict.keys())

        for width in unique_widths:
            prob += (
                lpSum(pattern_vars[i] * patterns[i].count(width) 
                      for i in range(len(patterns))) >= demand_dict[width]
            , f"Demand_{width}mm")

        # 求解
        prob.solve(PULP_CBC_CMD(msg=0))
        if prob.status != 1:
            return ({"success": False, "message": f"求解失败，状态码: {prob.status}"}, {})

        # 提取解
        bins = []
        pattern_usage = {}
        for i, var in enumerate(pattern_vars):
            count = int(var.varValue)
            if count > 0:
                pattern_usage[patterns[i]] = count
                for _ in range(count):
                    bins.append(list(patterns[i]))

        # 结果统计与计划（保持原方法）
        stats = self._calculate_stats(bins)
        summary_plan = self._generate_summary_plan(pattern_usage)
        detail_plan = self._generate_detail_plan(bins)

        result = {
            "success": True,
            "parameters": {
                "original_length": self.L_original,
                "effective_length": self.L,
                "head_cut": self.head_cut,
                "tail_cut": self.tail_cut,
                "cutting_loss": self.loss_mm,
                "demands": self.demands
            },
            "statistics": {
                "total_bars": stats['total_bars'],
                "utilization": round(stats['utilization'], 2),
                "total_waste": stats['total_waste'],
                "total_cutting_loss": stats['total_loss'],
                "total_head_tail_cut": (self.head_cut + self.tail_cut) * stats['total_bars']
            },
            "summary_plan": summary_plan,
            "detail_plan": detail_plan,
            "demand_verification": self._verify_demands(bins)
        }

        debug = {
            "patterns": patterns,
            "pattern_usage": pattern_usage,
            "bins": bins,
            "stats": stats
        }

        return result, debug

    def _calculate_stats(self, bins: List[List[int]]) -> Dict:
        total_bins = len(bins)
        total_waste = sum(self.calculate_waste(bin_items) for bin_items in bins)
        total_loss = sum((len(bin_items) - 1) * self.loss_mm for bin_items in bins)
        total_used = sum(sum(bin_items) for bin_items in bins)
        utilization = (total_used / (total_bins * self.L)) * 100 if total_bins > 0 else 0

        return {
            'total_bars': total_bins,
            'total_waste': total_waste,
            'total_loss': total_loss,
            'total_used': total_used,
            'utilization': utilization
        }

    def _generate_summary_plan(self, pattern_usage: Dict) -> List[Dict]:
        summary = []
        for pattern, count in pattern_usage.items():
            cuts = len(pattern) - 1
            total_length = sum(pattern)
            loss = cuts * self.loss_mm
            waste = self.L - total_length - loss

            cutting_list = list(pattern)
            counts = Counter(cutting_list)
            cutting_list_str = " + ".join(f"{length} * {cnt}" for length, cnt in counts.items())

            summary.append({
                "pattern_id": len(summary) + 1,
                "count": count,
                "cutting_list": list(pattern),
                "cutting_list_str": cutting_list_str,
                "total_length": total_length,
                "cutting_times": cuts,
                "cutting_loss": loss,
                "waste": waste,
                "utilization": round((total_length / self.L) * 100, 2)
            })
        return summary

    def _generate_detail_plan(self, bins: List[List[int]]) -> List[Dict]:
        details = []
        for i, bin_items in enumerate(bins, 1):
            cuts = len(bin_items) - 1
            total_length = sum(bin_items)
            loss = cuts * self.loss_mm
            waste = self.L - total_length - loss

            details.append({
                "bar_number": i,
                "original_length": self.L_original,
                "head_cut": self.head_cut,
                "tail_cut": self.tail_cut,
                "effective_length": self.L,
                "cutting_list": bin_items,
                "total_used": total_length,
                "cutting_times": cuts,
                "cutting_loss": loss,
                "waste": waste,
                "utilization": round((total_length / self.L) * 100, 2)
            })
        return details

    def _verify_demands(self, bins: List[List[int]]) -> Dict:
        produced = {}
        for bin_items in bins:
            for item in bin_items:
                produced[item] = produced.get(item, 0) + 1

        verification = []
        all_satisfied = True
        for width, demand in self.demands:
            actual = produced.get(width, 0)
            satisfied = actual >= demand
            if not satisfied:
                all_satisfied = False

            verification.append({
                "width": width,
                "demand": demand,
                "actual": actual,
                "satisfied": satisfied
            })

        return {
            "all_satisfied": all_satisfied,
            "details": verification
        }

    def print_summary(self, debug):

        patterns = debug["patterns"]
        usage = debug["pattern_usage"]
        bins = debug["bins"]
        stats = debug["stats"]

        print("\n" + "="*80)
        print("💡 切割优化算法调试信息（可读格式）")
        print("="*80)

        # 基本信息
        print(f"原材料长度：{self.L_original} mm")
        print(f"有效长度：{self.L} mm（去头 {self.head_cut}，去尾 {self.tail_cut}）")
        print(f"每刀损耗：{self.loss_mm} mm")
        print(f"总使用根数：{stats['total_bars']}")
        print(f"总体利用率：{round(stats['utilization'],2)} %")
        print(f"总浪费：{stats['total_waste']} mm")
        print(f"总切割损耗：{stats['total_loss']} mm")

        print("\n--- 切割模式使用情况 ---")
        for i, (pattern, count) in enumerate(usage.items(), 1):
            c = Counter(pattern)
            pattern_str = " + ".join(f"{k}×{v}" for k,v in c.items())
            print(f"模式 {i}: 使用 {count} 次 | {pattern_str}")

        print("\n--- 每根钢材详细切割 ---")
        for i, b in enumerate(bins, 1):
            c = Counter(b)
            b_str = " + ".join(f"{k}×{v}" for k,v in c.items())
            print(f"第 {i} 根: {b_str}")

        print("="*80)
        print("✔ 调试信息结束\n")


def optimize_cutting(L: int, demands: List[List[int]], loss_mm: int = 5, 
                     head_cut: int = 0, tail_cut: int = 0) -> str:
    """
    API接口函数 - 返回JSON字符串
    """
    optimizer = SteelCuttingOptimizer(L, demands, loss_mm, head_cut, tail_cut)
    result, debug = optimizer.solve()
    optimizer.print_summary(debug)
    return json.dumps(result, ensure_ascii=False, indent=2)


# 示例使用
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("示例----------------")
    print("=" * 80)
    result3 = optimize_cutting(
        L = 6000,  # 原材料长度
        demands = [
            [123, 1000],
            [1596, 105],
            [65, 521],
            [851, 73],
            [50, 851]
        ],
        loss_mm = 2,
        head_cut=0,
        tail_cut=0
    )

    # 写入文件
    with open('result3_cut.json', 'w', encoding='utf-8') as f:
        f.write(result3)
    print("结果已保存到: result3_cut.json")

    print("\n✓ 所有结果已保存，请查看JSON文件")

    result_test4 = optimize_cutting(
    L = 7000,
    demands = [
        [50, 500],
        [80, 300],
        [100, 200],
        [145, 120],
        [200, 90],
        [330, 70],
        [480, 60],
        [620, 55],
        [930, 40],
        [1230, 20]
    ],
    loss_mm = 2,
    head_cut = 0,
    tail_cut = 0
)
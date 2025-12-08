from ortools.sat.python import cp_model
import sys
import time
import json
from collections import defaultdict

# 增加 Python 递归深度，防止模式生成阶段崩溃
sys.setrecursionlimit(3000) 

# ========================================================================
# CuttingStockOptimizer 类定义（保持不变）
# ========================================================================
class CuttingStockOptimizer:
    # __init__
    
    def __init__(self, stock_length: int, loss_mm: int, demands: list, head_cut: int, tail_cut: int):
        self.L_raw = stock_length
        self.loss = loss_mm
        self.head_cut = head_cut
        self.tail_cut = tail_cut
        self.demands = demands
        self.L_effective = self.L_raw - self.head_cut - self.tail_cut
        
        if self.L_effective <= 0:
            raise ValueError(f"错误：有效长度 ({self.L_effective}mm) 小于等于零。请检查切头/去尾参数是否过大。")

        self.item_widths = [d[0] for d in demands]
        self.item_counts = [d[1] for d in demands]
        self.num_items = len(self.demands)
        
        self.patterns = []
        self.MAX_PATTERNS = 10000 
        self._patterns_generated_count = 0 
        
        self.result_vars = {} 

    # --- 模式生成和计算辅助函数  ---
    def _calculate_pattern_length(self, pattern: list) -> int:
        total_pieces = sum(pattern)
        if total_pieces == 0:
            return 0
        material_len = sum(pattern[i] * self.item_widths[i] for i in range(len(pattern)))
        waste_len = (total_pieces - 1) * self.loss if total_pieces > 0 else 0
        return material_len + waste_len

    def _generate_patterns_recursive(self, current_pattern: list):
        if self._patterns_generated_count >= self.MAX_PATTERNS: return
        added = False
        start_index = 0
        for i in range(len(current_pattern) - 1, -1, -1):
            if current_pattern[i] > 0:
                start_index = i
                break
                
        for i in range(start_index, self.num_items):
            temp_pattern = list(current_pattern)
            temp_pattern[i] += 1
            if self._calculate_pattern_length(temp_pattern) <= self.L_effective: 
                self._generate_patterns_recursive(temp_pattern)
                added = True
        
        if not added and sum(current_pattern) > 0:
            self._patterns_generated_count += 1
            self.patterns.append(current_pattern)
            
            if self._patterns_generated_count % 500 == 0:
                sys.stdout.write(f"\r正在生成可行切割模式... 已找到 {self._patterns_generated_count} 个")
                sys.stdout.flush()

    def generate_all_patterns(self):
        sys.stdout.write("正在生成可行切割模式...")
        sys.stdout.flush()
        initial_pattern = [0] * self.num_items
        self._generate_patterns_recursive(initial_pattern)
        sys.stdout.write(f"\r正在生成可行切割模式... 完成。共找到 {len(self.patterns)} 种可行模式。\n")
        sys.stdout.flush()

        unique_patterns = []
        seen = set()
        for p in self.patterns:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique_patterns.append(p)
        self.patterns = unique_patterns
        print(f"去重后剩余模式: {len(self.patterns)} 种")
        
        if not self.patterns:
            raise ValueError("未找到任何可行的切割模式。请检查输入参数。")

    def solve(self) -> dict:
        start_time = time.time()
        
        try:
            self.generate_all_patterns()
        except ValueError as e:
            return {"success": False, "message": str(e)}

        model = cp_model.CpModel()
        max_stock = sum(self.item_counts) 
        x = [model.NewIntVar(0, max_stock, f'pattern_{j}') for j in range(len(self.patterns))]

        for i in range(self.num_items):
            model.Add(
                sum(self.patterns[j][i] * x[j] for j in range(len(self.patterns))) 
                >= self.item_counts[i]
            )

        model.Minimize(sum(x))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0 
        status = solver.Solve(model)
        
        end_time = time.time()
        
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {
                "success": False, 
                "message": f"未找到最优解。求解状态: {solver.StatusName(status)}。请尝试增加求解时间限制或检查需求是否可行。"
            }
        
        self.result_vars = {
            "total_bars": int(solver.ObjectiveValue()),
            "solver": solver,
            "pattern_counts": [solver.Value(var) for var in x],
            "solve_time": end_time - start_time
        }
        
        return self._format_results_to_json()

    def _format_results_to_json(self) -> dict:
        solver = self.result_vars['solver']
        total_bars = self.result_vars['total_bars']
        pattern_counts = self.result_vars['pattern_counts']
        
        actual_production = defaultdict(int)
        summary_plan = []
        total_used_length_in_effective = 0
        total_internal_waste = 0
        total_cutting_loss = 0
        pattern_id_counter = 0

        for j in range(len(self.patterns)):
            count = pattern_counts[j]
            if count > 0:
                pattern_id_counter += 1
                pat = self.patterns[j]
                
                used_len_in_pattern = self._calculate_pattern_length(pat)
                internal_waste_per_bar = self.L_effective - used_len_in_pattern 
                
                total_pieces = sum(pat)
                cutting_times = total_pieces - 1 if total_pieces > 0 else 0
                cutting_loss_per_bar = cutting_times * self.loss
                
                total_used_length_in_effective += used_len_in_pattern * count
                total_internal_waste += internal_waste_per_bar * count
                total_cutting_loss += cutting_loss_per_bar * count
                
                cutting_list_raw = []
                for i, num in enumerate(pat):
                    if num > 0:
                        width = self.item_widths[i]
                        actual_production[width] += num * count
                        cutting_list_raw.extend([width] * num)
                
                cutting_list_raw.sort(reverse=True)
                utilization = round((used_len_in_pattern / self.L_effective) * 100, 2)
                
                summary_plan.append({
                    "pattern_id": pattern_id_counter,
                    "count": count,
                    "cutting_list": cutting_list_raw,
                    "total_length": used_len_in_pattern, 
                    "cutting_times": cutting_times,
                    "cutting_loss": cutting_loss_per_bar,
                    "waste": internal_waste_per_bar,
                    "utilization": utilization
                })
        
        demand_details = []
        all_satisfied = True
        for width, demand in self.demands:
            actual = actual_production[width]
            satisfied = actual >= demand
            if not satisfied:
                all_satisfied = False
            demand_details.append({
                "width": width, "demand": demand, "actual": actual, "satisfied": satisfied
            })
            
        demand_verification = {"all_satisfied": all_satisfied, "details": demand_details}
        
        detail_plan = []
        bar_number_counter = 1
        for plan in summary_plan:
            for _ in range(plan['count']):
                detail_plan.append({
                    "bar_number": bar_number_counter,
                    "original_length": self.L_raw,
                    "head_cut": self.head_cut,
                    "tail_cut": self.tail_cut,
                    "effective_length": self.L_effective,
                    "cutting_list": plan['cutting_list'],
                    "total_used": plan['total_length'],
                    "cutting_times": plan['cutting_times'],
                    "cutting_loss": plan['cutting_loss'],
                    "waste": plan['waste'],
                    "utilization": plan['utilization']
                })
                bar_number_counter += 1

        total_effective_length = total_bars * self.L_effective
        utilization = round((total_used_length_in_effective / total_effective_length) * 100, 2)
        total_head_tail_cut = total_bars * (self.head_cut + self.tail_cut)
        
        statistics = {
            "total_bars": total_bars,
            "utilization": utilization,
            "total_waste": total_internal_waste,
            "total_cutting_loss": total_cutting_loss,
            "total_head_tail_cut": total_head_tail_cut,
            "solver_time_seconds": round(self.result_vars['solve_time'], 4)
        }
        
        parameters = {
            "original_length": self.L_raw,
            "effective_length": self.L_effective,
            "head_cut": self.head_cut,
            "tail_cut": self.tail_cut,
            "cutting_loss": self.loss,
            "demands": self.demands
        }

        return {
            "success": True,
            "message": "求解成功，找到最优解。",
            "parameters": parameters,
            "statistics": statistics,
            "summary_plan": summary_plan,
            "detail_plan": detail_plan,
            "demand_verification": demand_verification
        }

# ========================================================================
# --- 用户输入/程序执行入口 ---
# ========================================================================
def print_summary_to_console(stats, summary_plan):
    """精简打印核心统计信息和汇总方案到控制台"""
    
    print("\n" + "=" * 50)
    print("🚀 钢材切割优化结果摘要 (Summary) 🚀")
    print("=" * 50)
    
    # 打印统计信息
    print("## 📊 总体统计 (Statistics)")
    print("-" * 50)
    print(f"| {'项目':<20} | {'数值':<25} |")
    print("-" * 50)
    print(f"| {'原材料总根数':<20} | {stats['total_bars']:<25} |")
    print(f"| {'总体材料利用率':<20} | {stats['utilization']:.2f}%{'':<23} |")
    print(f"| {'有效长度内余料总和':<20} | {stats['total_waste']} mm{'':<21} |")
    print(f"| {'切缝损耗总和':<20} | {stats['total_cutting_loss']} mm{'':<21} |")
    print(f"| {'去头去尾总损耗':<20} | {stats['total_head_tail_cut']} mm{'':<21} |")
    print(f"| {'求解耗时':<20} | {stats['solver_time_seconds']:.4f} 秒{'':<21} |")
    print("-" * 50)
    
    # 打印汇总切割方案
    print("\n## 🔪 汇总切割方案 (Summary Plan)")
    print("-" * 85)
    header = f"| {'ID':<4} | {'根数':<6} | {'利用率%':<8} | {'模式内余料':<10} | {'切割清单 (规格*数量)':<45} |"
    print(header)
    print("-" * 85)
    
    for plan in summary_plan:
        # 格式化切割清单
        item_counts = defaultdict(int)
        for w in plan['cutting_list']:
            item_counts[w] += 1
        
        cut_str = ", ".join([f"{w}*{c}" for w, c in item_counts.items()])
        
        row = f"| {plan['pattern_id']:<4} | {plan['count']:<6} | {plan['utilization']:.2f}{'':<6} | {plan['waste']:<10} | {cut_str:<45} |"
        print(row)

    print("-" * 85)


if __name__ == "__main__":
    # 示例配置参数 (使用您在JSON中提供的参数值)
    L_raw = 6000        
    loss_mm = 5         
    head_cut = 0       
    tail_cut = 0       
    
    demands = [         
        [1430, 80],
        [1230, 96],
        [1145, 79],
        [1092, 78],
        [143, 800],
        [123, 106],
        [114, 719],
        [86, 78],
        [12, 1206],
        [314, 79],
        [186, 178]
    ]
    
    # 定义输出文件名
    OUTPUT_FILENAME = "cutting_optimization_result.json"

    # 执行计算
    optimizer = CuttingStockOptimizer(L_raw, loss_mm, demands, head_cut, tail_cut)
    result_json = optimizer.solve()

    if result_json.get("success"):
        # 1. 写入文件
        try:
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 完整结果已成功写入文件：{OUTPUT_FILENAME}")
        except Exception as e:
            print(f"\n❌ 文件写入失败：{e}")

        # 2. 精简打印到控制台
        stats = result_json['statistics']
        summary_plan = result_json['summary_plan']
        print_summary_to_console(stats, summary_plan)
        
    else:
        print(f"\n❌ 求解失败：{result_json.get('message', '未知错误')}")
        print(json.dumps(result_json, indent=2, ensure_ascii=False))
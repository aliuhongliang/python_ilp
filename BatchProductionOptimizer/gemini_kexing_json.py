from ortools.sat.python import cp_model
import sys
import time
import json
from collections import defaultdict

# 增加 Python 递归深度
sys.setrecursionlimit(3000) 

class BatchProductionOptimizer:
    def __init__(self, raw_length_m: float, head_cut: int, tail_cut: int, loss_mm: int, demands: list, max_bars: int):
        """
        初始化优化器
        :param raw_length_m: 母材长度 (米) -> 转换为毫米
        :param head_cut: 切头长度 (毫米)
        :param tail_cut: 去尾长度 (毫米)
        :param loss_mm: 锯缝损耗 (毫米)
        :param demands: 零件规格及每套需求量 [[y1, x1], [y2, x2], ...]
        :param max_bars: 现有母材总根数 (X)
        """
        self.L_raw = int(raw_length_m * 1000) # 转换为毫米
        self.head_cut = head_cut
        self.tail_cut = tail_cut
        self.loss = loss_mm
        self.demands_per_batch = demands
        self.max_bars = max_bars
        
        # 计算每根母材的有效可用长度
        self.L_effective = self.L_raw - self.head_cut - self.tail_cut
        
        if self.L_effective <= 0:
            raise ValueError(f"错误：有效长度 ({self.L_effective}mm) 小于等于零。请检查切头/去尾参数是否过大。")

        self.item_widths = [d[0] for d in demands]   # 零件宽度 y1, y2, ...
        self.required_per_batch = [d[1] for d in demands] # 每套需求的数量 x1, x2, ...
        self.num_items = len(self.demands_per_batch)
        
        self.patterns = []
        self.MAX_PATTERNS = 10000 
        self._patterns_generated_count = 0 
        
        self.result_vars = {} 

    # --- 模式生成和计算辅助函数 (与之前一致) ---
    def _calculate_pattern_length(self, pattern: list) -> int:
        """计算一个模式消耗的总长度（只计算零件和切缝损耗）"""
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
            
            # 实时进度打印，仅用于调试
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
        
        if not self.patterns:
            raise ValueError("未找到任何可行的切割模式。请检查输入参数。")

    # --- 核心求解方法 (目标函数修改) ---
    def solve(self) -> dict:
        start_time = time.time()
        
        try:
            self.generate_all_patterns()
        except ValueError as e:
            return {"success": False, "message": str(e)}

        model = cp_model.CpModel()
        
        # 1. 变量定义
        
        # x[j]: 第 j 种模式使用的次数 (上界为最大母材数)
        x = [model.NewIntVar(0, self.max_bars, f'pattern_count_{j}') 
             for j in range(len(self.patterns))]
        
        # K: 生产的总套数 (整数变量)
        # 上界：粗略估计，最少需求零件的最大值 / 最大零件需求量
        max_total_demand = sum(self.required_per_batch) * self.max_bars # 粗略上界
        K = model.NewIntVar(0, max_total_demand, 'batch_count_K')
        
        # 2. 约束条件
        
        # 约束 A: 使用的母材总数不能超过现有数量
        model.Add(sum(x) <= self.max_bars)
        
        # 约束 B: 生产的每种零件数量必须满足 K 套的要求
        # Sum(模式j中包含零件i的数量 * 模式j的使用次数) >= K * (零件i的每套需求量)
        for i in range(self.num_items):
            # 零件 i 的实际产出总数 (Actual Production)
            actual_production_i = sum(self.patterns[j][i] * x[j] for j in range(len(self.patterns)))
            
            # 约束：实际产出 >= K * 需求量
            required_i = K * self.required_per_batch[i] # 这是一个线性表达式
            model.Add(actual_production_i >= required_i)

        # 3. 目标函数：最大化生产的总套数 K
        model.Maximize(K)

        # 4. 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0 
        status = solver.Solve(model)
        
        end_time = time.time()
        
        # 5. 结果处理
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {
                "success": False, 
                "message": f"未找到最优解。求解状态: {solver.StatusName(status)}。请检查输入参数或尝试增加求解时间限制。"
            }
        
        # 缓存求解结果变量
        self.result_vars = {
            "total_batches_K": solver.Value(K),
            "solver": solver,
            "pattern_counts": [solver.Value(var) for var in x],
            "total_bars": int(sum(solver.Value(var) for var in x)), # 实际使用了多少根母材
            "solve_time": end_time - start_time
        }
        
        return self._format_results_to_json()


    # --- 结果格式化为 JSON 对象 (适应新的统计目标) ---
    def _format_results_to_json(self) -> dict:
        solver = self.result_vars['solver']
        total_bars = self.result_vars['total_bars']
        pattern_counts = self.result_vars['pattern_counts']
        total_batches_K = self.result_vars['total_batches_K']
        
        # 临时统计变量
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
                
                # 统计数据累加
                total_used_length_in_effective += used_len_in_pattern * count
                total_internal_waste += internal_waste_per_bar * count
                total_cutting_loss += cutting_loss_per_bar * count
                
                # 汇总切割清单
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
        
        # --- 2. demand_verification (验证实际生产是否达到 K 套) ---
        demand_details = []
        all_satisfied = True
        for i in range(self.num_items):
            width = self.item_widths[i]
            demand_per_batch = self.required_per_batch[i]
            required_total = total_batches_K * demand_per_batch
            actual = actual_production[width]
            
            satisfied = actual >= required_total
            if not satisfied:
                all_satisfied = False
                
            demand_details.append({
                "width": width, 
                "demand_per_batch": demand_per_batch,
                "required_total": required_total,
                "actual": actual, 
                "satisfied": satisfied
            })
            
        demand_verification = {
            "all_satisfied": all_satisfied, # 理论上K是最优解时这里应为True
            "batches_produced": total_batches_K,
            "details": demand_details
        }
        
        # --- 3. detail_plan (使用实际使用的 total_bars 来创建) ---
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

        # --- 4. statistics ---
        total_effective_length = total_bars * self.L_effective
        utilization = round((total_used_length_in_effective / total_effective_length) * 100, 2)
        total_head_tail_cut = total_bars * (self.head_cut + self.tail_cut)
        
        statistics = {
            "max_available_bars": self.max_bars,
            "actual_bars_used": total_bars,
            "total_batches_produced": total_batches_K, # 新增
            "utilization": utilization,
            "total_waste": total_internal_waste,
            "total_cutting_loss": total_cutting_loss,
            "total_head_tail_cut": total_head_tail_cut,
            "solver_time_seconds": round(self.result_vars['solve_time'], 4)
        }
        
        # --- 5. parameters ---
        parameters = {
            "raw_length_m": self.L_raw / 1000,
            "original_length": self.L_raw,
            "effective_length": self.L_effective,
            "head_cut": self.head_cut,
            "tail_cut": self.tail_cut,
            "cutting_loss": self.loss,
            "demands_per_batch": self.demands_per_batch
        }

        # --- 6. Final JSON Output ---
        return {
            "success": True,
            "message": f"求解成功，最多可生产 {total_batches_K} 套零件，使用了 {total_bars} 根母材。",
            "parameters": parameters,
            "statistics": statistics,
            "summary_plan": summary_plan,
            "detail_plan": detail_plan,
            "demand_verification": demand_verification
        }


# ========================================================================
# --- 控制台打印和文件输出模块 ---
# ========================================================================
def print_summary_to_console(stats, summary_plan):
    """精简打印核心统计信息和汇总方案到控制台"""
    
    print("\n" + "=" * 50)
    print("🚀 批量生产优化结果摘要 (Summary) 🚀")
    print("=" * 50)
    
    # 打印统计信息
    print("## 📊 总体统计 (Statistics)")
    print("-" * 50)
    print(f"| {'项目':<25} | {'数值':<20} |")
    print("-" * 50)
    print(f"| {'最大可生产套数':<25} | {stats['total_batches_produced']:<20} |")
    print(f"| {'实际使用原材料根数':<25} | {stats['actual_bars_used']:<20} |")
    print(f"| {'总体材料利用率':<25} | {stats['utilization']:.2f}%{'':<18} |")
    print("-" * 50)
    
    # 打印汇总切割方案
    print("\n## 🔪 汇总切割方案 (Summary Plan)")
    print("-" * 85)
    header = f"| {'ID':<4} | {'根数':<6} | {'利用率%':<8} | {'模式内余料':<10} | {'切割清单 (规格*数量)':<45} |"
    print(header)
    print("-" * 85)
    
    for plan in summary_plan:
        item_counts = defaultdict(int)
        for w in plan['cutting_list']:
            item_counts[w] += 1
        
        cut_str = ", ".join([f"{w}*{c}" for w, c in item_counts.items()])
        
        row = f"| {plan['pattern_id']:<4} | {plan['count']:<6} | {plan['utilization']:.2f}{'':<6} | {plan['waste']:<10} | {cut_str:<45} |"
        print(row)

    print("-" * 85)

# --- 运行示例 ---
if __name__ == "__main__":
    # === 🚨 用户自定义输入区域 🚨 ===
    raw_length_m = 6.0         # 母材长度：y 米 (6米)
    head_cut = 50              # 切头损耗 (a 毫米)
    tail_cut = 30              # 去尾损耗 (b 毫米)
    loss_mm = 5                # 锯缝损耗 (c 毫米)
    max_bars = 100             # 现有母材数量 (x 根)
    
    # 零件规格及每套需求量 (y1 毫米, x1 根)
    demands_input = [         
        [143, 104],   # 每套需要 1430mm 零件 1 根
        [123, 202],   # 每套需要 1230mm 零件 2 根
        [1145, 17],   # 每套需要 1145mm 零件 1 根
        [1092, 21],    # 每套需要 1092mm 零件 2 根
        [92, 210],    # 每套需要 1092mm 零件 2 根
        [192, 21]    # 每套需要 1092mm 零件 2 根
    ]
    # === 🚨 结束输入区域 🚨 ===

    OUTPUT_FILENAME = "batch_optimization_result.json"

    # 执行计算
    optimizer = BatchProductionOptimizer(raw_length_m, head_cut, tail_cut, loss_mm, demands_input, max_bars)
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
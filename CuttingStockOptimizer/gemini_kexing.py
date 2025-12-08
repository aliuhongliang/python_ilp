# from ortools.sat.python import cp_model
# import sys

# class CuttingStockOptimizer:
#     def __init__(self, stock_length, loss_mm, demands):
#         """
#         :param stock_length: 原材料长度 (e.g., 6000)
#         :param loss_mm: 切割损耗 (e.g., 5)
#         :param demands: 需求列表 [[宽度, 数量], ...]
#         """
#         self.L = stock_length
#         self.loss = loss_mm
#         self.demands = demands
        
#         # 提取单独的宽度和需求量
#         self.item_widths = [d[0] for d in demands]
#         self.item_counts = [d[1] for d in demands]
#         self.num_items = len(self.demands)
        
#         # 存储生成的可行模式 (Patterns)
#         # 每个 pattern 是一个列表，表示每种规格切几个
#         self.patterns = []

#     def _calculate_pattern_length(self, pattern):
#         """
#         计算一个模式消耗的总长度（包含损耗）
#         公式：Sum(宽度 * 数量) + (总切数 - 1) * 损耗
#         注意：如果只切1段，不需要损耗（或者是切头切尾，通常工业上假设 切数=段数-1 或 段数*损耗，这里按 段数-1 计算内部损耗）
#         """
#         total_pieces = sum(pattern)
#         if total_pieces == 0:
#             return 0
        
#         material_len = sum(pattern[i] * self.item_widths[i] for i in range(len(pattern)))
#         # 损耗计算：N段需要 N-1 次切割。如果最后一段是废料也需要切断，逻辑可调。
#         # 这里假设：在一根长钢上切下 N 段，中间产生 N-1 个缝隙。
#         waste_len = (total_pieces - 1) * self.loss if total_pieces > 1 else 0
        
#         return material_len + waste_len

#     def _generate_patterns_recursive(self, current_pattern):
#         """
#         使用递归 DFS 生成所有“极大”可行模式。
#         极大模式：无法再塞入任何一种剩余最小材料的模式。
#         """
#         current_len = self._calculate_pattern_length(current_pattern)
        
#         # 尝试添加每一种类型的钢材
#         added = False
#         # 从最后一种添加的类型开始尝试（避免重复排列，如 [1,0] 和 [0,1] 算同一种）
#         # 这是一个简单的去重逻辑
#         start_index = 0
#         for i in range(len(current_pattern) - 1, -1, -1):
#             if current_pattern[i] > 0:
#                 start_index = i
#                 break
                
#         for i in range(start_index, self.num_items):
#             # 检查添加这根钢材后是否超长
#             # 增加一段，损耗可能会增加（如果从0变1没损耗，从1变2增加一个loss）
#             temp_pattern = list(current_pattern)
#             temp_pattern[i] += 1
            
#             if self._calculate_pattern_length(temp_pattern) <= self.L:
#                 self._generate_patterns_recursive(temp_pattern)
#                 added = True
        
#         # 如果无法再添加任何钢材，且该模式不为空，则认为是一个有效模式
#         if not added and sum(current_pattern) > 0:
#             self.patterns.append(current_pattern)

#     def generate_all_patterns(self):
#         """生成模式的入口"""
#         print("正在生成可行切割模式...", end="")
#         initial_pattern = [0] * self.num_items
#         self._generate_patterns_recursive(initial_pattern)
#         print(f" 完成。共找到 {len(self.patterns)} 种可行模式。")
        
#         # 过滤掉极其低效的模式（可选，这里为了求最优解保留所有）
#         # 但去重是必须的（递归逻辑已大致去重，这里做最后保险）
#         unique_patterns = []
#         seen = set()
#         for p in self.patterns:
#             t = tuple(p)
#             if t not in seen:
#                 seen.add(t)
#                 unique_patterns.append(p)
#         self.patterns = unique_patterns
#         print(f"去重后剩余模式: {len(self.patterns)}")

#     def solve(self):
#         # 1. 生成模式
#         self.generate_all_patterns()

#         # 2. 建立 ILP 模型
#         model = cp_model.CpModel()
        
#         # 变量：x[j] 表示第 j 种模式使用的原材料根数
#         # 上界：粗略估计，最坏情况每根原材料只切一个最小零件
#         max_stock = sum(self.item_counts) 
#         x = []
#         for j in range(len(self.patterns)):
#             x.append(model.NewIntVar(0, max_stock, f'pattern_{j}'))

#         # 3. 约束条件：满足每种钢材的需求量
#         # Sum(模式j中包含钢材i的数量 * 模式j的使用次数) >= 需求量i
#         for i in range(self.num_items):
#             model.Add(
#                 sum(self.patterns[j][i] * x[j] for j in range(len(self.patterns))) 
#                 >= self.item_counts[i]
#             )

#         # 4. 目标函数：最小化使用的原材料总根数
#         model.Minimize(sum(x))

#         # 5. 求解
#         solver = cp_model.CpSolver()
#         # 设置求解时间限制（秒），防止极大规模卡死
#         solver.parameters.max_time_in_seconds = 10.0 
#         status = solver.Solve(model)

#         # 6. 输出结果
#         if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
#             print("-" * 30)
#             print(f"【最优解】需要原材料总数: {solver.ObjectiveValue()} 根")
#             print("-" * 30)
            
#             total_waste = 0
#             details = []
            
#             for j in range(len(self.patterns)):
#                 count = solver.Value(x[j])
#                 if count > 0:
#                     pat = self.patterns[j]
#                     used_len = self._calculate_pattern_length(pat)
#                     waste = self.L - used_len
#                     total_waste += waste * count
                    
#                     # 格式化输出该模式详情
#                     cut_str = []
#                     for i, num in enumerate(pat):
#                         if num > 0:
#                             cut_str.append(f"{self.item_widths[i]}mm*{num}")
                    
#                     details.append({
#                         "count": count,
#                         "cut_detail": ", ".join(cut_str),
#                         "used_len": used_len,
#                         "waste": waste
#                     })
            
#             # 打印详细方案
#             print(f"{'使用根数':<10} | {'切割方案 (规格*数量)':<30} | {'单根利用率':<10} | {'余料'}")
#             for d in details:
#                 usage_rate = (d['used_len'] / self.L) * 100
#                 print(f"{d['count']:<10} | {d['cut_detail']:<30} | {usage_rate:.2f}%     | {d['waste']}mm")
            
#             print("-" * 30)
#             print(f"总体平均利用率: {(1 - total_waste / (solver.ObjectiveValue() * self.L)) * 100:.2f}%")
#         else:
#             print("未找到可行解。")

# # --- 用户输入部分 ---
# if __name__ == "__main__":
#     # 配置参数
#     L = 6000
#     loss_mm = 5
#     demands = [
#         [123, 100],
#         [789, 1000],
#         [60, 521],
#         [952, 55],
#         [70, 743]
#     ]

#     # 执行计算
#     optimizer = CuttingStockOptimizer(L, loss_mm, demands)
#     optimizer.solve()



from ortools.sat.python import cp_model
import sys
import time

# 增加 Python 递归深度，防止模式生成阶段崩溃
sys.setrecursionlimit(3000) 

class CuttingStockOptimizer:
    def __init__(self, stock_length: int, loss_mm: int, demands: list, head_cut: int, tail_cut: int):
        """
        :param stock_length: 原材料总长度 (L_raw)
        :param loss_mm: 切割损耗 (kerf)
        :param demands: 需求列表 [[宽度, 数量], ...]
        :param head_cut: 每次使用的原材料需要切除的头部长度 (H)
        :param tail_cut: 每次使用的原材料需要切除的尾部长度 (T)
        """
        self.L_raw = stock_length
        self.loss = loss_mm
        self.head_cut = head_cut
        self.tail_cut = tail_cut
        self.demands = demands
        
        # 计算每根原材料的有效可用长度
        self.L_effective = self.L_raw - self.head_cut - self.tail_cut
        
        if self.L_effective <= 0:
            raise ValueError(f"错误：有效长度 ({self.L_effective}mm) 小于等于零。请检查切头/去尾参数是否过大。")

        self.item_widths = [d[0] for d in demands]
        self.item_counts = [d[1] for d in demands]
        self.num_items = len(self.demands)
        
        self.patterns = []
        self.MAX_PATTERNS = 10000 
        self._patterns_generated_count = 0 

    def _calculate_pattern_length(self, pattern: list) -> int:
        """
        计算一个模式消耗的总长度（只计算零件和切缝损耗，不含头尾损耗）
        公式：Sum(宽度 * 数量) + (总切数 - 1) * 损耗
        """
        total_pieces = sum(pattern)
        if total_pieces == 0:
            return 0
        
        # 1. 计算所有零件的实际长度总和
        material_len = sum(pattern[i] * self.item_widths[i] for i in range(len(pattern)))
        
        # 2. 计算切缝损耗 (N 段需要 N-1 次切割)
        waste_len = (total_pieces - 1) * self.loss if total_pieces > 0 else 0
        
        return material_len + waste_len

    def _generate_patterns_recursive(self, current_pattern: list):
        # 模式数量达到上限，停止生成
        if self._patterns_generated_count >= self.MAX_PATTERNS:
            return

        added = False
        start_index = 0
        
        for i in range(len(current_pattern) - 1, -1, -1):
            if current_pattern[i] > 0:
                start_index = i
                break
                
        for i in range(start_index, self.num_items):
            temp_pattern = list(current_pattern)
            temp_pattern[i] += 1
            
            # **核心修改：检查模式长度是否 <= 有效可用长度**
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

        # 去重处理（保持不变）
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

    def solve(self):
        start_time = time.time()
        
        # 0. 打印有效长度信息
        print(f"原材料总长 (L_raw): {self.L_raw}mm, 切头 (H): {self.head_cut}mm, 去尾 (T): {self.tail_cut}mm")
        print(f"**单根有效可用长度 (L_effective): {self.L_effective}mm**")
        print("-" * 30)
        
        try:
            self.generate_all_patterns()
        except ValueError as e:
            print(f"错误：{e}")
            return

        # 2. 建立 ILP 模型 (与原代码逻辑相同)
        model = cp_model.CpModel()
        max_stock = sum(self.item_counts) 
        x = [model.NewIntVar(0, max_stock, f'pattern_{j}') for j in range(len(self.patterns))]

        # 约束条件：满足需求量
        for i in range(self.num_items):
            model.Add(
                sum(self.patterns[j][i] * x[j] for j in range(len(self.patterns))) 
                >= self.item_counts[i]
            )

        # 目标函数：最小化使用的原材料总根数
        model.Minimize(sum(x))

        # 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0 
        status = solver.Solve(model)
        
        end_time = time.time()

        # 6. 输出结果
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            total_bars = int(solver.ObjectiveValue())
            print("-" * 30)
            print(f"【最优解】需要原材料总数: {total_bars} 根")
            print(f"总计算耗时: {end_time - start_time:.2f} 秒")
            print("-" * 30)
            
            total_explicit_waste = total_bars * (self.head_cut + self.tail_cut) # 头尾损耗总和
            total_internal_waste = 0 # 模式内余料总和
            total_used_material_len = 0
            
            print(f"{'使用根数':<10} | {'模式内余料':<10} | {'总利用率':<10} | {'切割方案 (规格*数量)'}")
            
            for j in range(len(self.patterns)):
                count = solver.Value(x[j])
                if count > 0:
                    pat = self.patterns[j]
                    used_len_in_pattern = self._calculate_pattern_length(pat) # 零件 + 切缝
                    
                    # 模式内余料 (即 L_effective - L_used_in_pattern)
                    internal_waste = self.L_effective - used_len_in_pattern
                    total_internal_waste += internal_waste * count

                    total_used_material_len += used_len_in_pattern * count
                    
                    # 格式化输出该模式详情
                    cut_str = []
                    for i, num in enumerate(pat):
                        if num > 0:
                            cut_str.append(f"{self.item_widths[i]}mm*{num}")
                    
                    # 计算单根总利用率：只算有效部分 / 总原料长
                    total_waste_per_bar = internal_waste + self.head_cut + self.tail_cut
                    usage_rate_total = ((self.L_raw - total_waste_per_bar) / self.L_raw) * 100

                    print(f"{count:<10} | {internal_waste:<10} | {usage_rate_total:.2f}% | {', '.join(cut_str)}")
            
            # 整体计算
            total_raw_material = total_bars * self.L_raw
            overall_waste = total_explicit_waste + total_internal_waste
            overall_usage_rate = 100 - (overall_waste / total_raw_material) * 100
            
            print("-" * 30)
            print(f"总原材料长度投入: {total_raw_material:.0f} mm")
            print(f"总废料长度 (含头尾): {overall_waste:.0f} mm")
            print(f"**总体平均利用率: {overall_usage_rate:.2f}%**")

        else:
            print(f"未找到最优解。求解状态: {solver.StatusName(status)}")


# --- 用户输入/程序执行入口 ---
if __name__ == "__main__":
    # 示例配置参数 (增加切头和去尾参数)
    L_raw = 6000        # 原材料总长度
    loss_mm = 5         # 切割损耗
    head_cut = 10       # 切头损耗，例如 30mm (取值 0-50)
    tail_cut = 10      # 去尾损耗，例如 150mm (取值 0-200)
    
    demands = [         # [宽度, 需求量]
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

    # 执行计算
    optimizer = CuttingStockOptimizer(L_raw, loss_mm, demands, head_cut, tail_cut)
    optimizer.solve()
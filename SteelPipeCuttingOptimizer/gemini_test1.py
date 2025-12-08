from pulp import *
from collections import defaultdict
import time

def get_max_value_pattern_dp(stock_len: int, standard_lengths: list, cut_loss: int) -> list:
    """
    使用动态规划（DP）找到给定母材长度下的最大产值（总切割长度）切割模式。
    
    Args:
        stock_len (int): 母材长度。
        standard_lengths (list): 标准件规格列表。
        cut_loss (int): 每次有效切割的损耗。
        
    Returns:
        list: 包含找到的唯一最大产值模式的列表，如果找不到则返回空列表。
    """
    
    # DP 数组: dp[i] 存储长度 i 可以得到的最大有效切割长度（产值）
    # path[i] 存储达到 dp[i] 时，切下的最后一个标准件的长度
    dp = [-1] * (stock_len + 1)
    path = [None] * (stock_len + 1)
    
    # 初始条件：长度 0 时切割长度为 0
    dp[0] = 0
    
    # 动态规划填充：i 代表当前可以使用的剩余长度
    for i in range(1, stock_len + 1):
        for std_len in standard_lengths:
            # 每次切割所需的总长度 = 标准件长度 + 损耗
            # 假设每个切下的标准件都伴随一次 cut_loss
            required_len = std_len + cut_loss 
            
            # 只有当当前剩余长度 i 足够切下这个件（含损耗）时
            if i >= required_len:
                # 检查前面状态是否可达 (dp[i - required_len] != -1)
                if dp[i - required_len] != -1:
                    new_value = dp[i - required_len] + std_len
                    # 如果找到了更好的产值，则更新
                    if new_value > dp[i]:
                        dp[i] = new_value
                        path[i] = std_len
            
    # --- 回溯路径，找到最优切割模式 ---
    
    best_value = 0
    best_index = -1
    
    # 寻找在 stock_len 范围内，哪一个长度能产生最大切割长度
    # 注意：我们必须确保最后剩下的余料 (stock_len - i) 是非负的
    for i in range(stock_len, -1, -1):
        if dp[i] > best_value:
            best_value = dp[i]
            best_index = i
            
    # 如果找不到任何切割模式
    if best_index == -1 or best_value == 0:
        return []

    # 回溯
    current_len = best_index
    pattern_counts = defaultdict(int)
    
    while current_len > 0 and path[current_len] is not None:
        std_len = path[current_len]
        pattern_counts[std_len] += 1
        
        # 减去这个件所需的长度 (含损耗)
        required_len = std_len + cut_loss
        current_len -= required_len

    # 如果 current_len < 0，说明 DP 模型或回溯逻辑有误，应避免
    if current_len < 0:
         # 这是一个保护性检查，理论上 DP 算法应该避免这个情况
         return [] 

    # 计算最终模式数据
    cut_pieces_list = [pattern_counts[l] for l in standard_lengths]
    total_cut_length = best_value
    consumed_length = best_index
    remaining_waste = stock_len - consumed_length
    
    # 返回包含唯一最优模式的列表
    return [{
        'counts': cut_pieces_list,           # 对应 standard_lengths 顺序的件数
        'total_length': total_cut_length,    # 模式总切割长度 (产值)
        'waste': remaining_waste,            # 余料
        'parent_stock_len': stock_len
    }]


def cutting_stock_optimization(
    stock_materials: dict,  # {长度: 数量}
    standard_lengths: list, # [长度1, 长度2, ...]
    cut_loss: int           # 切割损耗
) -> dict:
    """
    求解一维下料问题，目标是最大化总切割长度（产值），并用完所有母材。
    使用 DP 预生成最大产值模式，再用 ILP 求解分配问题。
    """
    start_time = time.time()
    
    # --- 1. 使用 DP 生成最优切割模式 ---
    all_patterns = {}
    
    print("--- 1. 模式生成 (DP) ---")
    
    for stock_len, stock_count in stock_materials.items():
        if stock_count > 0:
            patterns_list = get_max_value_pattern_dp(stock_len, standard_lengths, cut_loss)
            
            if patterns_list:
                # 理论上 DP 只会返回一个最大产值模式
                all_patterns[stock_len] = patterns_list
            else:
                print(f"警告：长度 {stock_len} mm 的母材无法切出任何标准件。")
                all_patterns[stock_len] = []

    if not any(all_patterns.values()):
        return {"error": "无法从任何母材中切割出任何标准件。"}
    
    # --- 2. 建立并求解 ILP 模型 ---
    
    print("--- 2. ILP 模型建立与求解 ---")

    # 创建问题实例：最大化问题
    prob = LpProblem("Steel_Pipe_Cutting_Optimization", LpMaximize)
    
    # 决策变量：x[(stock_len, pattern_index)] 表示母材长度为 stock_len 的
    # 第 pattern_index 种切割模式被使用的次数
    x = {}
    
    for stock_len, patterns in all_patterns.items():
        for i, pattern in enumerate(patterns):
            var_name = f"x_{stock_len}_{i}"
            # 变量必须是整数 (Integer)
            x[(stock_len, i)] = LpVariable(var_name, lowBound=0, cat='Integer')
            
    # 目标函数：最大化总切割长度（产值）
    # Maximize SUM ( x[(stock_len, i)] * pattern['total_length'] )
    prob += lpSum([
        x[(stock_len, i)] * pattern['total_length'] 
        for stock_len, patterns in all_patterns.items() 
        for i, pattern in enumerate(patterns)
    ]), "Total_Value"
    
    # 约束条件 1：母材数量限制（每种母材长度都必须用完）
    # SUM x[(stock_len, i)] == stock_materials[stock_len]
    for stock_len, count in stock_materials.items():
        if count > 0 and stock_len in all_patterns:
            # 只有在存在切割模式的情况下才添加约束
            if all_patterns[stock_len]:
                prob += lpSum([
                    x[(stock_len, i)] 
                    for i, _ in enumerate(all_patterns[stock_len])
                ]) == count, f"Stock_Limit_{stock_len}"
            
    # 求解问题
    prob.solve()
    
    # --- 3. 结果解析 ---
    end_time = time.time()
    
    status = LpStatus[prob.status]
    if status != 'Optimal':
        return {"error": f"求解器未能找到最优解，状态: {status}"}

    # 统计结果
    total_cut_pieces = defaultdict(int)
    total_waste = 0
    used_patterns_details = []
    
    for stock_len, patterns in all_patterns.items():
        for i, pattern in enumerate(patterns):
            var = x.get((stock_len, i))
            if var and var.varValue > 0:
                times_used = int(round(var.varValue)) # 四舍五入取整
                
                # 更新标准件数量
                for idx, count in enumerate(pattern['counts']):
                    total_cut_pieces[standard_lengths[idx]] += count * times_used
                    
                # 更新余料
                total_waste += pattern['waste'] * times_used
                
                # 记录使用的切割模式
                used_patterns_details.append({
                    'parent_stock_len': stock_len,
                    'times_used': times_used,
                    'pieces_cut': {standard_lengths[idx]: count for idx, count in enumerate(pattern['counts']) if count > 0},
                    'remaining_waste_per_pipe': pattern['waste'],
                })

    # 最终结果
    result = {
        "status": status,
        "max_total_value": value(prob.objective),
        "total_cut_pieces_summary": dict(total_cut_pieces),
        "total_waste_sum": total_waste,
        "used_patterns_details": used_patterns_details,
        "standard_lengths_order": standard_lengths,
        "time_taken": f"{end_time - start_time:.2f} seconds"
    }
    
    return result


# 示例数据（您提供的卡死数据）
stock_materials = {
    6000: 114,  # 114根长6000的母材
}

standard_lengths = [1303, 451, 214, 230, 123, 78, 93] # 7种标准件规格
cut_loss = 5 # 每次切割损耗 5mm

# 运行求解
results = cutting_stock_optimization(stock_materials, standard_lengths, cut_loss)

# --- 结果输出 ---
print("\n" + "=" * 50)
print("             钢管切割优化最终结果")
print("=" * 50)
print(f"求解状态: {results.get('status')}")
print(f"最大总产值 (总切割长度): {results.get('max_total_value'):,.2f} mm")
print(f"总耗时: {results.get('time_taken')}")
print("-" * 50)

print("## 🏭 标准件产出数量总览")
for length, count in results.get('total_cut_pieces_summary', {}).items():
    print(f"规格 {length} mm: {count} 根")

print("-" * 50)

print("## ♻️ 母材使用详情与余料")
total_waste = 0
for detail in results.get('used_patterns_details', []):
    waste = detail['remaining_waste_per_pipe'] * detail['times_used']
    total_waste += waste
    
    pieces_str = ', '.join([f"{l}mm x {c}" for l, c in detail['pieces_cut'].items()])
    print(f"母材长度: {detail['parent_stock_len']} mm")
    print(f"  - **使用次数**: {detail['times_used']} 次")
    print(f"  - **切割模式**: {pieces_str}")
    print(f"  - **每根余料**: {detail['remaining_waste_per_pipe']} mm (总余料: {waste} mm)")
    
print("-" * 50)
print(f"总余料合计: {total_waste:,.2f} mm")
print("=" * 50)
# import math
# from typing import Dict

# class CuttingOptimizer:
#     """
#     一维下料问题的简化计算器。
#     根据母材尺寸、损耗和一套产品的需求，计算最多能切割多少套产品。
#     假设每套产品必须在单根母材的净长度内完整切割。
#     """

#     def __init__(self,
#                  x_mother_material: int,
#                  y_length_meters: float,
#                  a_head_cut_mm: int,
#                  b_tail_cut_mm: int,
#                  c_cutting_loss_mm: int,
#                  set_requirements: Dict[int, int]):
#         """
#         初始化切割参数和需求。

#         Args:
#             x_mother_material (int): 母材的只数 (X)。
#             y_length_meters (float): 单根母材的长度 (Y) (米)。
#             a_head_cut_mm (int): 切头损耗 (a) (毫米)。
#             b_tail_cut_mm (int): 去尾损耗 (b) (毫米)。
#             c_cutting_loss_mm (int): 单次切割损耗 (c) (毫米)。
#             set_requirements (Dict[int, int]): 一套产品所需的规格和根数。
#                 格式为 {短料长度y_i (mm): 根数x_i (根)}。
#         """
#         # 母材参数
#         self.X = x_mother_material
#         # 将母材长度从米转换为毫米
#         self.L_mm = y_length_meters * 1000  
#         self.a = a_head_cut_mm
#         self.b = b_tail_cut_mm
#         self.c = c_cutting_loss_mm
        
#         # 需求参数
#         self.requirements = set_requirements
        
#         # 预计算变量
#         self.S_pure_length = 0  # 一套的纯料总长
#         self.K_total_pieces = 0  # 一套的短料总根数
#         self._pre_calculate_set_metrics()

#     def _pre_calculate_set_metrics(self):
#         """计算一套产品的纯料总长和短料总根数。"""
#         for length_mm, count in self.requirements.items():
#             self.S_pure_length += length_mm * count
#             self.K_total_pieces += count

#     def calculate_total_sets(self) -> int:
#         """
#         执行计算，返回总共可以切割出的完整套数。
        
#         Returns:
#             int: 总共可以切割出的完整套数。
#         """

#         # --- 1. 计算一套产品的总耗长 (L_set_total) ---
        
#         # 假设每切一根短料，就产生一次切割损耗 c
#         C_set_total_loss = self.K_total_pieces * self.c
        
#         # 一套的总耗长 (包含料长和所有切割损耗)
#         L_set_total = self.S_pure_length + C_set_total_loss

#         # --- 2. 计算单根母材的净可用长度 (L_net) ---
        
#         # L_net = L - a - b
#         L_net_available = self.L_mm - self.a - self.b
        
#         # --- 3. 计算单根母材可切套数 (N_single) ---

#         if L_net_available < L_set_total:
#             # print(f"警告：单根母材净可用长度 ({L_net_available:.2f} mm) 不足以切割一套所需总长 ({L_set_total} mm)。")
#             N_single_material_sets = 0
#         else:
#             # N_single = floor(L_net / L_set_total)
#             N_single_material_sets = math.floor(L_net_available / L_set_total)

#         # --- 4. 计算总共可切套数 (N_total) ---
        
#         N_total_sets = N_single_material_sets * self.X
        
#         return N_total_sets

#     def get_details(self) -> Dict:
#         """返回详细的计算步骤结果。"""
#         L_net_available = self.L_mm - self.a - self.b
        
#         C_set_total_loss = self.K_total_pieces * self.c
#         L_set_total = self.S_pure_length + C_set_total_loss
        
#         N_single_material_sets = 0
#         if L_net_available >= L_set_total:
#             N_single_material_sets = math.floor(L_net_available / L_set_total)

#         return {
#             "母材总只数 (X)": self.X,
#             "单根母材总长 (L)": f"{self.L_mm:.2f} mm",
#             "切头/去尾损耗 (a+b)": f"{self.a + self.b} mm",
#             "单根母材净可用长度 (L_net)": f"{L_net_available:.2f} mm",
#             "--- 一套产品需求 ---": "---",
#             "纯料总长 (S)": f"{self.S_pure_length} mm",
#             "短料总根数 (K)": self.K_total_pieces,
#             "总切割损耗 (C_set)": f"{C_set_total_loss} mm",
#             "一套总耗长 (L_set_total)": f"{L_set_total} mm",
#             "--- 结果 ---": "---",
#             "单根母材可切套数": N_single_material_sets,
#             "总共可切完整套数": self.calculate_total_sets()
#         }

# # --- 完整脚本示例运行 ---
# if __name__ == "__main__":
    
#     # 场景设定：使用之前讨论的参数
    
#     # 1. 定义母材和损耗参数
#     X_mother_material = 500         # 10 只母材
#     Y_length_meters = 6.0          # 6.0 米
#     A_head_cut_mm = 20             # 切头 20 mm
#     B_tail_cut_mm = 10             # 去尾 10 mm
#     C_cutting_loss_mm = 3          # 切损 3 mm
    
#     # 2. 定义一套产品的需求：{短料长度(mm): 根数}
#     REQUIREMENTS = {
#         1500: 101,
#         800: 300,
#         80: 976, 
#         180: 304,
#         254: 39
#     }

#     print("--- 🚀 开始计算 ---")
    
#     # 3. 创建 CuttingOptimizer 实例
#     optimizer = CuttingOptimizer(
#         x_mother_material=X_mother_material,
#         y_length_meters=Y_length_meters,
#         a_head_cut_mm=A_head_cut_mm,
#         b_tail_cut_mm=B_tail_cut_mm,
#         c_cutting_loss_mm=C_cutting_loss_mm,
#         set_requirements=REQUIREMENTS
#     )

#     # 4. 获取计算结果和详细步骤
#     total_sets = optimizer.calculate_total_sets()
#     details = optimizer.get_details()
    
#     print("\n--- ✅ 计算详情 ---")
#     for key, value in details.items():
#         print(f"{key:<30}: {value}")

#     print("\n------------------------------")
#     print(f"💰 最终结果：总共可切完整套数: {total_sets} 套")
#     print("------------------------------")

import math
from typing import Dict, Any

class CuttingPoolCalculator:
    """
    一维下料问题的总材料池计算器。
    将所有母材的总长度视为一个可用资源池，计算理论上最多能切割的完整套数。
    
    计算公式: 总套数 = 地板( [总母材长度 - 总切头去尾损耗] / [一套所需的总耗长] )
    """

    def __init__(self,
                 x_mother_material: int,
                 y_length_meters: float,
                 a_head_cut_mm: int,
                 b_tail_cut_mm: int,
                 c_cutting_loss_mm: int,
                 set_requirements: Dict[int, int]):
        """
        初始化切割参数和需求。

        Args:
            x_mother_material (int): 母材的只数 (X)。
            y_length_meters (float): 单根母材的长度 (Y) (米)。
            a_head_cut_mm (int): 切头损耗 (a) (毫米)。
            b_tail_cut_mm (int): 去尾损耗 (b) (毫米)。
            c_cutting_loss_mm (int): 单次切割损耗 (c) (毫米)。
            set_requirements (Dict[int, int]): 一套产品所需的规格和根数。
                格式为 {短料长度y_i (mm): 根数x_i (根)}。
        """
        # 母材参数
        self.X = x_mother_material
        # 单根母材长度 (毫米)
        self.L_mm = y_length_meters * 1000  
        self.a = a_head_cut_mm
        self.b = b_tail_cut_mm
        self.c = c_cutting_loss_mm
        
        # 需求参数
        self.requirements = set_requirements
        
        # 预计算变量
        self.S_pure_length = 0     # 一套的纯料总长
        self.K_total_pieces = 0    # 一套的短料总根数
        self._pre_calculate_set_metrics()

    def _pre_calculate_set_metrics(self):
        """计算一套产品的纯料总长和短料总根数。"""
        for length_mm, count in self.requirements.items():
            self.S_pure_length += length_mm * count
            self.K_total_pieces += count

    def calculate_total_sets(self) -> int:
        """
        执行计算，返回总共可以切割出的理论最大完整套数。
        """

        # --- 1. 计算一套产品的总耗长 (L_set_total) ---
        
        # 总切割损耗 (C_set) 假设每切一根短料，就产生一次切割损耗 c
        C_set_total_loss = self.K_total_pieces * self.c
        
        # 一套的总耗长 (包含料长和所有切割损耗)
        L_set_total = self.S_pure_length + C_set_total_loss

        # --- 2. 计算总材料池的净可用长度 (L_avail) ---
        
        # 总毛长度
        L_gross = self.X * self.L_mm
        
        # 总固定损耗 (切头 a + 去尾 b，应用于每根母材)
        L_fixed_loss = self.X * (self.a + self.b)
        
        # 总净可用长度 (用于切割短料和切割损耗)
        L_avail = L_gross - L_fixed_loss
        
        # --- 3. 计算理论最大套数 (N_max) ---

        if L_avail < L_set_total:
            return 0
            
        # N_max = floor(L_avail / L_set_total)
        N_total_sets = math.floor(L_avail / L_set_total)
        
        return N_total_sets

    def get_details(self) -> Dict[str, Any]:
        """返回详细的计算步骤结果。"""
        
        L_gross = self.X * self.L_mm
        L_fixed_loss = self.X * (self.a + self.b)
        L_avail = L_gross - L_fixed_loss

        C_set_total_loss = self.K_total_pieces * self.c
        L_set_total = self.S_pure_length + C_set_total_loss
        
        N_total_sets = self.calculate_total_sets()
        
        L_remaining = L_avail - N_total_sets * L_set_total

        return {
            "母材总只数 (X)": self.X,
            "单根母材长度 (Y)": f"{self.L_mm:.2f} mm",
            "总毛长度 (X * Y)": f"{L_gross:.2f} mm",
            "总固定损耗 (X * (a+b))": f"{L_fixed_loss:.2f} mm",
            "总净可用长度 (L_avail)": f"{L_avail:.2f} mm",
            "--- 一套产品需求 ---": "---",
            "纯料总长 (S)": f"{self.S_pure_length} mm",
            "短料总根数 (K)": self.K_total_pieces,
            "总切割损耗 (C_set)": f"{C_set_total_loss} mm",
            "一套总耗长 (L_set_total)": f"{L_set_total} mm",
            "--- 结果 ---": "---",
            "理论最大可切套数": N_total_sets,
            "切割后剩余长度 (L_avail - N_total * L_set_total)": f"{L_remaining:.2f} mm"
        }

# --- 完整脚本示例运行 ---
if __name__ == "__main__":
    
    # 1. 定义母材和损耗参数
    X_mother_material = 500         # 10 只母材
    Y_length_meters = 6.0          # 6.0 米
    A_head_cut_mm = 20             # 切头 20 mm
    B_tail_cut_mm = 10             # 去尾 10 mm
    C_cutting_loss_mm = 3          # 切损 3 mm
    
    # 2. 定义一套产品的需求：{短料长度(mm): 根数}
    REQUIREMENTS = {
        1500: 101,
        800: 300,
        80: 976, 
        180: 304,
        254: 39
    }

    print("--- 🚀 开始计算：基于总材料池的方法 ---")
    
    # 3. 创建 CuttingPoolCalculator 实例
    calculator = CuttingPoolCalculator(
        x_mother_material=X_mother_material,
        y_length_meters=Y_length_meters,
        a_head_cut_mm=A_head_cut_mm,
        b_tail_cut_mm=B_tail_cut_mm,
        c_cutting_loss_mm=C_cutting_loss_mm,
        set_requirements=REQUIREMENTS
    )

    # 4. 获取计算结果和详细步骤
    total_sets = calculator.calculate_total_sets()
    details = calculator.get_details()
    
    print("\n--- ✅ 计算详情 ---")
    for key, value in details.items():
        print(f"{key:<40}: {value}")

    print("\n------------------------------")
    print(f"💰 最终结果：总共可切理论最大套数: {total_sets} 套")
    print("------------------------------")
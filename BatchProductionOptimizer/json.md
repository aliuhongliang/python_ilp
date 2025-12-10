{
  "metadata": {
    "optimization_time": 0.125,          // 优化计算耗时（秒）
    "cutting_loss": 0.1,                 // 切割损耗长度（米/次）
    "timestamp": "2024-01-15T10:30:00",  // 优化时间戳
    "algorithm_version": "1.0",          // 算法版本
    "total_pipes_processed": 6           // 处理的母材总数
  }
}

{
  "input_data": {
    "raw_materials": [  // 母材列表
      {
        "material_id": "MAT_10.0m",  // 母材标识符
        "length": 10.0,             // 母材长度（米）
        "count": 3                  // 该长度母材的数量
      },
      {
        "material_id": "MAT_8.0m",
        "length": 8.0,
        "count": 2
      }
    ],
    "standard_parts": [  // 标准件列表
      {
        "name": "A",      // 标准件名称
        "length": 3.0,    // 标准件长度（米）
        "price": 100.0    // 标准件价格（元）
      },
      {
        "name": "B",
        "length": 2.0,
        "price": 60.0
      }
    ],
    "problem_description": "钢管切割优化问题"  // 问题描述
  }
}

{
  "summary": {
    "total_pipes": 6,                  // 处理的母材总根数
    "total_material_length": 58.0,     // 母材总长度（米）
    "total_value": 1250.0,             // 总产值（元）
    "total_waste": 5.8,                // 总余料长度（米）
    "overall_utilization": 90.0,       // 总体材料利用率（%）
    "total_pieces_produced": 18,       // 生产的标准件总数
    "total_cuts": 12,                  // 总切割次数
    "part_summary": {                  // 零件生产汇总
      "A": {
        "length": 3.0,                 // 零件长度
        "price": 100.0,                // 零件单价
        "total_count": 8,              // 该零件总生产数量
        "total_value": 800.0,          // 该零件总产值
        "value_per_meter": 33.33       // 价值密度（元/米）
      },
      "B": {
        "length": 2.0,
        "price": 60.0,
        "total_count": 10,
        "total_value": 450.0,
        "value_per_meter": 30.0
      }
    },
    "performance_metrics": {            // 性能指标
      "value_per_meter": 21.55,        // 每米母材产值（元/米）
      "value_per_cut": 104.17,         // 每次切割产值（元/次）
      "pieces_per_pipe": 3.0,          // 每根母材平均生产件数
      "waste_per_pipe": 0.97           // 每根母材平均余料（米）
    }
  }
}


{
  "cutting_plans": [  // 每根母材的切割方案列表
    {
      "pipe_id": "MAT_10.0m_1",    // 母材唯一标识符
      "pipe_length": 10.0,         // 母材原始长度
      "total_value": 300.0,        // 该母材切割产值
      "waste_length": 0.8,         // 该母材余料长度
      "utilization_rate": 92.0,    // 该母材利用率（%）
      "total_pieces": 3,           // 该母材生产件数
      "cutting_count": 2,          // 该母材切割次数
      "part_counts": {             // 该母材切割零件数量
        "A": 3,                    // 零件A数量
        "B": 0                     // 零件B数量
      },
      "cutting_details": [  // 详细切割清单
        {
          "part_name": "A",        // 零件名称
          "part_length": 3.0,      // 零件长度
          "count": 3,              // 该零件切割数量
          "total_length": 9.0,     // 该零件总长度
          "total_value": 300.0,    // 该零件总产值
          "sequence": [1, 2, 3]    // 切割顺序（可选）
        }
      ],
      "material_usage": {          // 材料使用情况
        "used_length": 9.2,        // 已使用长度（含损耗）
        "waste_percentage": 8.0,   // 余料百分比
        "loss_length": 0.2         // 切割损耗总长度
      }
    },
    // ... 更多母材的切割方案
  ]
}

{
  "analysis": {
    "length_group_analysis": {  // 按长度分组分析
      "10.0": {                  // 母材长度分组
        "count": 3,             // 该长度母材数量
        "total_value": 900.0,    // 该组总产值
        "total_waste": 2.4,     // 该组总余料
        "total_pieces": 9,      // 该组总生产件数
        "avg_utilization": 92.0, // 该组平均利用率
        "avg_value": 300.0,     // 该组单根平均产值
        "best_scheme": "3×A"    // 该组最佳切割方案
      },
      "8.0": {
        "count": 2,
        "total_value": 350.0,
        "total_waste": 1.4,
        "total_pieces": 6,
        "avg_utilization": 91.25,
        "avg_value": 175.0,
        "best_scheme": "2×A+1×B"
      }
    },
    "part_efficiency": {  // 零件效率分析
      "A": {
        "value_density": 33.33,      // 价值密度（元/米）
        "production_efficiency": 1.0, // 生产效率（0-1）
        "production_ratio": 0.444,   // 生产占比（该零件数/总零件数）
        "total_material_used": 24.0  // 该零件使用的总材料长度
      },
      "B": {
        "value_density": 30.0,
        "production_efficiency": 0.9,
        "production_ratio": 0.556,
        "total_material_used": 20.0
      }
    },
    "waste_analysis": {  // 余料分析
      "total_waste_percentage": 10.0,  // 总余料率
      "avg_waste_per_pipe": 0.97,      // 平均每根余料
      "waste_distribution": {          // 余料分布
        "0-0.5m": 2,                   // 余料0-0.5米的母材数量
        "0.5-1m": 3,
        "1-1.5m": 1,
        ">1.5m": 0
      },
      "waste_reduction_potential": 0.3  // 余料减少潜力（%）
    },
    "efficiency_metrics": {  // 效率指标
      "value_per_meter": 21.55,     // 每米产值
      "value_per_cut": 104.17,      // 每次切割产值
      "pieces_per_meter": 0.31,     // 每米生产件数
      "value_per_pipe": 208.33,     // 每根母材产值
      "efficiency_score": 0.85      // 效率评分（0-1）
    },
    "cutting_patterns": {  // 切割模式分析
      "most_common_pattern": "3×A",  // 最常见切割模式
      "pattern_count": 3,            // 该模式使用次数
      "unique_patterns": 4,          // 不同切割模式数量
      "pattern_distribution": {      // 模式分布
        "3×A": 3,
        "2×A+1×B": 2,
        "1×A+2×B": 1
      }
    },
    "improvement_suggestions": [  // 改进建议
      "增加长度8米的母材可提高利用率2%",
      "考虑调整标准件B的长度为1.9米可减少余料"
    ]
  }
}
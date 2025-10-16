#!/usr/bin/env python3
"""
生成分子生成RL训练数据集
- 13个模板 × 10个描述变体 = 130条数据
- 输出为parquet格式，符合VERL的RLHFDataset要求
"""

import pandas as pd
import json
from pathlib import Path

# ============================================================================
# 13个模板（来自SFT训练数据）
# ============================================================================
TEMPLATES = [
    "Based on the given information, generate a molecule that meets the desired specifications: {description}",
    "Give me a molecule that satisfies the conditions outlined in the description: {description}",
    "Generate a molecule based on this description: {description}",
    "Can you create a molecule that matches the given characteristics? {description}",
    "I need a molecule that meets the following conditions: {description} Please represent the molecule in SMILES.",
    "Suppose there is a molecule that meets the following description: {description} Please write the SMILES representation of it.",
    "{description} Use the above information to create a molecule.",
    "Build a molecule that meets the requirement: {description}",
    "Generate a molecule that fulfills the requirement: {description}",
    "Conceptualize a molecule that meets the specified attribute(s): {description}",
    "Come up with a molecule based on the description: {description}",
    "Could you please return a molecule that adheres to this description? {description}",
    "I give you a description of a molecule, and you need to return one molecule in SMILES that meets the description. The description: {description}"
]

# ============================================================================
# 10个EGFR抑制剂描述变体
# 涵盖不同的表达方式和侧重点
# ============================================================================
DESCRIPTIONS = [
    # 变体1: 强调高效力
    "A highly potent EGFR inhibitor for lung cancer treatment",
    
    # 变体2: 强调IC50指标
    "An EGFR inhibitor with IC50 less than 10 nM for lung cancer",
    
    # 变体3: 强调选择性
    "A selective EGFR tyrosine kinase inhibitor with minimal off-target effects",
    
    # 变体4: 强调临床应用
    "An effective EGFR-TKI for non-small cell lung cancer (NSCLC) treatment",
    
    # 变体5: 强调突变型EGFR
    "A potent inhibitor targeting EGFR T790M mutation in resistant lung cancer",
    
    # 变体6: 强调药物性质
    "A drug-like EGFR inhibitor with good oral bioavailability for lung cancer",
    
    # 变体7: 强调活性
    "A highly active EGFR kinase inhibitor for oncology applications",
    
    # 变体8: 强调靶向治疗
    "A targeted EGFR inhibitor for advanced lung adenocarcinoma",
    
    # 变体9: 强调新一代药物
    "A next-generation EGFR inhibitor overcoming resistance mechanisms",
    
    # 变体10: 强调安全性和效力平衡
    "A potent and safe EGFR inhibitor with favorable pharmacokinetic properties"
]

# ============================================================================
# 生成数据集
# ============================================================================
def generate_dataset():
    """生成130条RL训练数据"""
    
    data_list = []
    index = 0
    
    print("=" * 80)
    print("开始生成分子生成RL数据集")
    print("=" * 80)
    print(f"模板数量: {len(TEMPLATES)}")
    print(f"描述变体数量: {len(DESCRIPTIONS)}")
    print(f"总数据量: {len(TEMPLATES)} × {len(DESCRIPTIONS)} = {len(TEMPLATES) * len(DESCRIPTIONS)}")
    print()
    
    # 遍历每个描述变体
    for desc_id, description in enumerate(DESCRIPTIONS, start=1):
        print(f"处理描述变体 {desc_id}/{len(DESCRIPTIONS)}: {description[:60]}...")
        
        # 对每个描述，应用所有13个模板
        for template_id, template in enumerate(TEMPLATES, start=1):
            # 用模板格式化描述
            prompt_content = template.format(description=description)
            
            # 构建VERL格式的数据
            data_item = {
                "data_source": "molecule_generation",
                
                # prompt使用chat格式（会被tokenizer.apply_chat_template处理）
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                
                # 任务能力类型
                "ability": "molecule_generation",
                
                # reward_model字段（ground_truth留空，不会被读取）
                "reward_model": {
                    "ground_truth": "",  # 分子生成任务不需要ground_truth
                    "style": "molecular_property"
                },
                
                # 额外信息
                "extra_info": {
                    "index": index,  # 必需且唯一
                    "task": "egfr_molecule_generation",
                    "template_id": template_id,  # 使用的模板编号（1-13）
                    "description_id": desc_id,   # 使用的描述变体编号（1-10）
                    "raw_description": description,  # 原始描述
                    "split": "train"
                }
            }
            
            data_list.append(data_item)
            index += 1
        
        print(f"  ✅ 生成了 {len(TEMPLATES)} 条数据（模板1-13）")
    
    print()
    print(f"✅ 总共生成 {len(data_list)} 条数据")
    
    return data_list

# ============================================================================
# 保存为parquet格式
# ============================================================================
def save_to_parquet(data_list, output_path):
    """保存数据为parquet格式"""
    
    print()
    print("=" * 80)
    print("保存数据集")
    print("=" * 80)
    
    # 转换为DataFrame
    # 注意：prompt、reward_model、extra_info需要保持为dict/list类型
    df = pd.DataFrame(data_list)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 保存为parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"✅ 数据集已保存到: {output_path}")
    
    # 验证保存的数据
    df_loaded = pd.read_parquet(output_path)
    print(f"✅ 验证: 成功加载 {len(df_loaded)} 条数据")
    
    return df

# ============================================================================
# 显示示例数据
# ============================================================================
def show_examples(data_list, num_examples=3):
    """显示几条示例数据"""
    
    print()
    print("=" * 80)
    print(f"示例数据（前{num_examples}条）")
    print("=" * 80)
    
    for i in range(min(num_examples, len(data_list))):
        print(f"\n--- 示例 {i+1} ---")
        print(json.dumps(data_list[i], indent=2, ensure_ascii=False))

# ============================================================================
# 统计信息
# ============================================================================
def print_statistics(data_list):
    """打印数据集统计信息"""
    
    print()
    print("=" * 80)
    print("数据集统计")
    print("=" * 80)
    
    # 统计每个template_id的数量
    template_counts = {}
    description_counts = {}
    
    for item in data_list:
        template_id = item["extra_info"]["template_id"]
        description_id = item["extra_info"]["description_id"]
        
        template_counts[template_id] = template_counts.get(template_id, 0) + 1
        description_counts[description_id] = description_counts.get(description_id, 0) + 1
    
    print(f"\n总数据量: {len(data_list)}")
    print(f"模板数量: {len(template_counts)} (每个模板应有 {len(DESCRIPTIONS)} 条)")
    print(f"描述变体数量: {len(description_counts)} (每个描述应有 {len(TEMPLATES)} 条)")
    
    print(f"\n每个模板的数据量:")
    for tid in sorted(template_counts.keys()):
        print(f"  Template {tid:2d}: {template_counts[tid]:3d} 条")
    
    print(f"\n每个描述变体的数据量:")
    for did in sorted(description_counts.keys()):
        desc = DESCRIPTIONS[did-1][:50]
        print(f"  Description {did:2d}: {description_counts[did]:3d} 条 - {desc}...")
    
    # 检查index唯一性
    indices = [item["extra_info"]["index"] for item in data_list]
    if len(indices) == len(set(indices)):
        print(f"\n✅ 所有index唯一 ({len(indices)} 个)")
    else:
        print(f"\n❌ 警告: index有重复！")

# ============================================================================
# 生成验证集
# ============================================================================
def generate_val_dataset():
    """生成13条验证数据（13个模板 × 1个描述）"""
    
    data_list = []
    
    print()
    print("=" * 80)
    print("开始生成验证集")
    print("=" * 80)
    
    # 选择一个描述变体用于验证（使用第1个）
    val_description = DESCRIPTIONS[0]
    print(f"验证集使用描述: {val_description}")
    print(f"模板数量: {len(TEMPLATES)}")
    print(f"总数据量: {len(TEMPLATES)} × 1 = {len(TEMPLATES)}")
    print()
    
    # 对所有13个模板应用这个描述
    for template_id, template in enumerate(TEMPLATES, start=1):
        # 用模板格式化描述
        prompt_content = template.format(description=val_description)
        
        # 构建VERL格式的数据
        data_item = {
            "data_source": "molecule_generation",
            
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            
            "ability": "molecule_generation",
            
            "reward_model": {
                "ground_truth": "",
                "style": "molecular_property"
            },
            
            "extra_info": {
                "index": template_id - 1,  # 0-12
                "task": "egfr_molecule_generation",
                "template_id": template_id,
                "description_id": 1,
                "raw_description": val_description,
                "split": "val"
            }
        }
        
        data_list.append(data_item)
    
    print(f"✅ 生成了 {len(data_list)} 条验证数据")
    
    return data_list

# ============================================================================
# 主函数
# ============================================================================
def main():
    # 输出路径
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    train_output_path = output_dir / "molecule_generation_train.parquet"
    val_output_path = output_dir / "molecule_generation_val.parquet"
    
    print()
    print("=" * 80)
    print("分子生成RL数据集生成器")
    print("=" * 80)
    print(f"训练集输出: {train_output_path}")
    print(f"验证集输出: {val_output_path}")
    print()
    
    # 生成训练数据
    train_data_list = generate_dataset()
    
    # 显示训练集统计信息
    print_statistics(train_data_list)
    
    # 显示训练集示例
    show_examples(train_data_list, num_examples=3)
    
    # 保存训练集
    train_df = save_to_parquet(train_data_list, train_output_path)
    
    # 生成验证集
    val_data_list = generate_val_dataset()
    
    # 显示验证集示例
    print()
    print("=" * 80)
    print("验证集示例（前2条）")
    print("=" * 80)
    show_examples(val_data_list, num_examples=2)
    
    # 保存验证集
    val_df = save_to_parquet(val_data_list, val_output_path)
    
    # 最终总结
    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"✅ 训练集已生成: {train_output_path}")
    print(f"   数据量: {len(train_data_list)} 条 (13模板 × 10描述)")
    print(f"\n✅ 验证集已生成: {val_output_path}")
    print(f"   数据量: {len(val_data_list)} 条 (13模板 × 1描述)")
    print(f"\n使用方式:")
    print(f"  在VERL训练脚本中设置:")
    print(f'  export TRAIN_FILES="[\\"{train_output_path}\\"]"')
    print(f'  export VAL_FILES="[\\"{val_output_path}\\"]"')
    print()

if __name__ == "__main__":
    main()


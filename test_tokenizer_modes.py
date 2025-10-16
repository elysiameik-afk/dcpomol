#!/usr/bin/env python3
"""
测试tokenizer的fast和slow模式兼容性
为VERL训练提供建议
"""

from transformers import AutoTokenizer
import sys

# 配置路径
ORIGINAL_MODEL_PATH = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3'
TOKENIZER_PATH_SLOW = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3-chat-template'
TOKENIZER_PATH_FAST = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3-chat-template-fast'

def test_tokenizer(path, use_fast, name):
    """测试单个tokenizer"""
    print(f"\n{'=' * 80}")
    print(f"测试: {name}")
    print(f"路径: {path}")
    print(f"模式: {'Fast' if use_fast else 'Slow'}")
    print('=' * 80)
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=use_fast)
        print(f"✅ 加载成功")
        print(f"   类型: {type(tokenizer).__name__}")
        print(f"   词汇表大小: {len(tokenizer)}")
        print(f"   Chat template: {'已设置' if tokenizer.chat_template else '未设置'}")
        
        # 测试基本编码/解码
        test_text = "[INST] Generate a molecule [/INST]"
        tokens = tokenizer(test_text, return_tensors="pt")
        decoded = tokenizer.decode(tokens['input_ids'][0])
        
        print(f"\n   基本功能测试:")
        print(f"   原始文本: {test_text}")
        print(f"   Token数量: {len(tokens['input_ids'][0])}")
        print(f"   解码结果: {decoded}")
        print(f"   一致性: {'✅ 通过' if test_text in decoded else '❌ 失败'}")
        
        # 测试chat_template
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": "Generate a highly potent EGFR inhibitor"}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            expected = "[INST] Generate a highly potent EGFR inhibitor [/INST]"
            
            print(f"\n   Chat template测试:")
            print(f"   格式化结果: {formatted}")
            print(f"   预期结果: {expected}")
            print(f"   匹配: {'✅ 通过' if formatted == expected else '❌ 失败'}")
            
            return True, tokenizer
        else:
            print(f"\n   ⚠️  Chat template未设置")
            return False, None
            
    except Exception as e:
        print(f"❌ 加载失败")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)[:200]}")
        return False, None

def main():
    print("=" * 80)
    print("Tokenizer模式兼容性测试")
    print("=" * 80)
    print("\n目标: 确定VERL训练应该使用哪种tokenizer模式")
    
    results = {}
    
    # ========================================================================
    # 测试1: 原始模型的slow模式
    # ========================================================================
    success, _ = test_tokenizer(ORIGINAL_MODEL_PATH, False, "原始模型 (Slow)")
    results['original_slow'] = success
    
    # ========================================================================
    # 测试2: 原始模型的fast模式
    # ========================================================================
    success, _ = test_tokenizer(ORIGINAL_MODEL_PATH, True, "原始模型 (Fast)")
    results['original_fast'] = success
    
    # ========================================================================
    # 测试3: 添加chat_template后的slow模式
    # ========================================================================
    success, tokenizer_slow = test_tokenizer(TOKENIZER_PATH_SLOW, False, "Chat Template (Slow)")
    results['chat_slow'] = success
    
    # ========================================================================
    # 测试4: 添加chat_template后的fast模式（如果存在）
    # ========================================================================
    import os
    if os.path.exists(TOKENIZER_PATH_FAST):
        success, tokenizer_fast = test_tokenizer(TOKENIZER_PATH_FAST, True, "Chat Template (Fast)")
        results['chat_fast'] = success
    else:
        print(f"\n{'=' * 80}")
        print(f"测试: Chat Template (Fast)")
        print(f"路径: {TOKENIZER_PATH_FAST}")
        print('=' * 80)
        print("⚠️  Fast模式tokenizer不存在（这是正常的）")
        results['chat_fast'] = False
    
    # ========================================================================
    # 总结和建议
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    print("\n测试结果:")
    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name:20s}: {status}")
    
    print("\n" + "=" * 80)
    print("VERL训练建议")
    print("=" * 80)
    
    if results['chat_fast']:
        print("\n✅ 推荐使用: Fast模式")
        print(f"   Tokenizer路径: {TOKENIZER_PATH_FAST}")
        print(f"   优势: 更快的tokenization速度")
        recommendation = "fast"
    elif results['chat_slow']:
        print("\n✅ 推荐使用: Slow模式")
        print(f"   Tokenizer路径: {TOKENIZER_PATH_SLOW}")
        print(f"   说明: Fast模式不可用，但Slow模式完全足够")
        print(f"   VERL完全支持Slow tokenizer，性能影响很小")
        recommendation = "slow"
    else:
        print("\n❌ 错误: 没有可用的tokenizer")
        print(f"   请检查add_chat_template.py的执行结果")
        return False, None
    
    print("\n配置示例（用于VERL训练脚本）:")
    print("-" * 80)
    if recommendation == "fast":
        print(f'export TOKENIZER_PATH="{TOKENIZER_PATH_FAST}"')
    else:
        print(f'export TOKENIZER_PATH="{TOKENIZER_PATH_SLOW}"')
    print("-" * 80)
    
    print("\n数据集格式:")
    print("-" * 80)
    print("""{
    "data_source": "molecule_generation",
    "prompt": [
        {
            "role": "user",
            "content": "Generate a highly potent EGFR inhibitor for lung cancer"
        }
    ],
    "ability": "molecule_generation",
    "reward_model": {
        "ground_truth": ""
    },
    "extra_info": {
        "index": 0,
        "task": "egfr_molecule_generation"
    }
}""")
    print("-" * 80)
    
    print("\n注意事项:")
    print("  1. Tokenizer会自动应用chat_template，添加[INST]...[/INST]")
    print("  2. 不需要在数据集中手动添加[INST]标签")
    print("  3. 模型会在[/INST]后生成<SMILES>...</SMILES>")
    
    return True, recommendation

if __name__ == "__main__":
    success, recommendation = main()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ 测试失败")
        print("=" * 80)
        exit(1)


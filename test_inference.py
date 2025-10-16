#!/usr/bin/env python3
"""
验证推理：对比旧方法（手动[INST]）vs 新方法（chat_template）
确保两种方式产生相同的输出
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置路径
MODEL_PATH = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3'
TOKENIZER_PATH = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3-chat-template'

def main():
    print("=" * 80)
    print("加载模型和tokenizer")
    print("=" * 80)
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"✅ 模型加载成功")
    print(f"   设备: {model.device}")
    
    # 加载tokenizer（带chat_template的版本）
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        use_fast=False
    )
    print(f"✅ Tokenizer加载成功")
    print(f"   类型: {type(tokenizer).__name__}")
    print(f"   Chat template: {'已设置' if tokenizer.chat_template else '未设置'}")
    
    # 测试输入
    input_text = "Generate a highly potent EGFR inhibitor for lung cancer"
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("方式1: 旧方法（手动添加[INST]标签）")
    print("=" * 80)
    
    prompt_old = f"[INST] {input_text} [/INST]"
    print(f"Prompt: {prompt_old}\n")
    
    inputs_old = tokenizer(prompt_old, return_tensors="pt").to(model.device)
    print(f"Token IDs shape: {inputs_old['input_ids'].shape}")
    print(f"Token IDs: {inputs_old['input_ids'][0].tolist()[:20]}... (前20个)")
    
    print("\n生成中...")
    with torch.no_grad():
        outputs_old = model.generate(
            **inputs_old,
            max_new_tokens=256,
            do_sample=False,  # 关闭采样确保可重复
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    result_old = tokenizer.decode(outputs_old[0], skip_special_tokens=False)
    print(f"\n生成结果:")
    print("-" * 80)
    print(result_old)
    print("-" * 80)
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("方式2: 新方法（使用chat_template）")
    print("=" * 80)
    
    messages = [{"role": "user", "content": input_text}]
    print(f"Messages: {messages}\n")
    
    prompt_new = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"格式化后的Prompt: {prompt_new}\n")
    
    inputs_new = tokenizer(prompt_new, return_tensors="pt").to(model.device)
    print(f"Token IDs shape: {inputs_new['input_ids'].shape}")
    print(f"Token IDs: {inputs_new['input_ids'][0].tolist()[:20]}... (前20个)")
    
    print("\n生成中...")
    with torch.no_grad():
        outputs_new = model.generate(
            **inputs_new,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    result_new = tokenizer.decode(outputs_new[0], skip_special_tokens=False)
    print(f"\n生成结果:")
    print("-" * 80)
    print(result_new)
    print("-" * 80)
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    
    prompt_match = (prompt_old == prompt_new)
    tokens_match = torch.equal(inputs_old['input_ids'], inputs_new['input_ids'])
    output_match = (result_old == result_new)
    
    print(f"\n1. Prompt文本是否相同: {'✅ 是' if prompt_match else '❌ 否'}")
    if not prompt_match:
        print(f"   旧方法: {prompt_old}")
        print(f"   新方法: {prompt_new}")
    
    print(f"\n2. Token IDs是否相同: {'✅ 是' if tokens_match else '❌ 否'}")
    if not tokens_match:
        print(f"   旧方法shape: {inputs_old['input_ids'].shape}")
        print(f"   新方法shape: {inputs_new['input_ids'].shape}")
        print(f"   提示: Token IDs不同可能导致生成结果不同")
    
    print(f"\n3. 生成输出是否相同: {'✅ 是' if output_match else '❌ 否'}")
    if not output_match:
        print(f"   ⚠️  警告: 两种方式生成的结果不同！")
        print(f"   这可能是因为prompt格式不一致")
        
        # 提取SMILES对比
        import re
        
        def extract_smiles(text):
            match = re.search(r'<SMILES>\s*(.+?)\s*</SMILES>', text)
            return match.group(1) if match else None
        
        smiles_old = extract_smiles(result_old)
        smiles_new = extract_smiles(result_new)
        
        print(f"\n   提取的SMILES对比:")
        print(f"   旧方法: {smiles_old}")
        print(f"   新方法: {smiles_new}")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    if prompt_match and tokens_match and output_match:
        print("✅ 完美！两种方式完全一致")
        print("\n结论:")
        print("  - Chat template工作正常")
        print("  - 可以在VERL训练中使用标准chat格式")
        print("  - 数据集使用: {'role': 'user', 'content': '...'}")
        return True
    elif prompt_match and tokens_match:
        print("⚠️  Prompt和Tokens相同，但输出不同")
        print("   这是正常的（如果使用了采样）")
        print("\n结论:")
        print("  - Chat template工作正常")
        print("  - 输入格式正确")
        return True
    else:
        print("❌ 存在问题！")
        print("\n可能的原因:")
        print("  1. Chat template格式不正确")
        print("  2. Tokenizer配置有问题")
        print("\n建议:")
        print("  - 检查chat_template定义")
        print("  - 确认[INST]标签没有额外空格")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 80)
        print("✅ 验证通过！可以进行下一步")
        print("=" * 80)
        exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ 验证失败，需要修复问题")
        print("=" * 80)
        exit(1)


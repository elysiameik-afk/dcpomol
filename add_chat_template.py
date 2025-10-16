#!/usr/bin/env python3
"""
添加chat_template到tokenizer并保存
这个脚本会：
1. 加载原始tokenizer
2. 添加Mistral格式的chat_template
3. 验证chat_template工作正常
4. 保存修改后的tokenizer（slow和fast模式）
"""

from transformers import AutoTokenizer

# 配置路径
ORIGINAL_MODEL_PATH = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3'
SAVE_PATH_SLOW = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3-chat-template'
SAVE_PATH_FAST = '/root/autodl-tmp/llm4mol/rlmodels/LlaSMol-EGFR-Final-exp3-chat-template-fast'

def main():
    print("=" * 80)
    print("步骤1: 加载原始tokenizer (slow模式)")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(
        ORIGINAL_MODEL_PATH,
        use_fast=False
    )
    
    print(f"✅ Tokenizer加载成功")
    print(f"   类型: {type(tokenizer).__name__}")
    print(f"   词汇表大小: {len(tokenizer)}")
    print(f"   原始chat_template: {tokenizer.chat_template}")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤2: 添加Mistral chat_template")
    print("=" * 80)
    
    # Mistral格式: [INST] user_message [/INST] assistant_response
    # 当add_generation_prompt=True时，只添加[INST]...[/INST]，等待模型生成
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    )
    
    print(f"✅ Chat template已添加")
    print(f"   模板内容: {tokenizer.chat_template}")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤3: 验证chat_template")
    print("=" * 80)
    
    test_messages = [
        {"role": "user", "content": "Generate a highly potent EGFR inhibitor for lung cancer"}
    ]
    
    try:
        result = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        expected = "[INST] Generate a highly potent EGFR inhibitor for lung cancer [/INST]"
        
        print(f"输入messages:")
        print(f"  {test_messages}")
        print(f"\n格式化后的文本:")
        print(f"  {result}")
        print(f"\n预期输出:")
        print(f"  {expected}")
        print(f"\n匹配检查: {'✅ 完全匹配' if result == expected else '❌ 不匹配'}")
        
        if result != expected:
            print(f"\n⚠️  警告: 输出格式与预期不符，可能需要调整chat_template")
            return False
            
    except Exception as e:
        print(f"❌ Chat template测试失败: {e}")
        return False
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤4: 保存tokenizer (slow模式)")
    print("=" * 80)
    
    try:
        tokenizer.save_pretrained(SAVE_PATH_SLOW)
        print(f"✅ Slow tokenizer已保存到:")
        print(f"   {SAVE_PATH_SLOW}")
        
        # 验证保存的tokenizer
        tokenizer_reloaded = AutoTokenizer.from_pretrained(SAVE_PATH_SLOW, use_fast=False)
        test_result = tokenizer_reloaded.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"✅ 重新加载验证成功")
        print(f"   格式化结果: {test_result}")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤5: 尝试保存fast模式 (可选)")
    print("=" * 80)
    
    try:
        tokenizer_fast = AutoTokenizer.from_pretrained(
            ORIGINAL_MODEL_PATH,
            use_fast=True
        )
        tokenizer_fast.chat_template = tokenizer.chat_template
        tokenizer_fast.save_pretrained(SAVE_PATH_FAST)
        
        print(f"✅ Fast tokenizer已保存到:")
        print(f"   {SAVE_PATH_FAST}")
        
        # 验证
        tokenizer_fast_reloaded = AutoTokenizer.from_pretrained(SAVE_PATH_FAST, use_fast=True)
        test_result_fast = tokenizer_fast_reloaded.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"✅ Fast模式重新加载验证成功")
        print(f"   格式化结果: {test_result_fast}")
        
    except Exception as e:
        print(f"⚠️  Fast模式保存失败: {e}")
        print(f"   这不影响使用，可以继续使用slow模式")
        print(f"   VERL完全支持slow tokenizer")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("✅ Chat template添加成功！")
    print(f"\n推荐使用的tokenizer路径:")
    print(f"  {SAVE_PATH_SLOW}")
    print(f"\n下一步:")
    print(f"  1. 运行 test_inference.py 验证推理结果")
    print(f"  2. 运行 test_tokenizer_modes.py 测试fast/slow兼容性")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 脚本执行失败，请检查错误信息")
        exit(1)
    else:
        print("\n✅ 脚本执行成功")
        exit(0)


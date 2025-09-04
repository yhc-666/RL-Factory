"""
从零开始理解 RL-Factory 中的 Prompt 构造过程
===========================================

作者：Claude Code
目的：通过实例详细讲解 prompt 的构造、chat template 的作用，以及多轮对话的实现
"""

# ========================================
# 第一部分：什么是 Chat Template？
# ========================================

"""
Chat Template 是一个格式化规则，告诉系统如何将对话转换成模型能理解的文本。

想象一下，当你和 AI 对话时，实际上有三种角色：
1. system: 系统提示（定义 AI 的行为）
2. user: 用户输入
3. assistant: AI 的回复

不同的模型需要不同的格式来区分这些角色。
"""

# 示例：原始对话数据
raw_conversation = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "今天天气怎么样？"},
    {"role": "assistant", "content": "我需要查询天气信息..."}
]

# Qwen 模型的 Chat Template 会将其转换为：
qwen_formatted_text = """<|im_start|>system
你是一个有帮助的助手<|im_end|>
<|im_start|>user
今天天气怎么样？<|im_end|>
<|im_start|>assistant
我需要查询天气信息...<|im_end|>"""

# GPT 模型可能使用不同的格式：
gpt_formatted_text = """System: 你是一个有帮助的助手
User: 今天天气怎么样？
Assistant: 我需要查询天气信息..."""

print("=" * 60)
print("基础概念：Chat Template 的作用")
print("=" * 60)
print("\n原始对话数据：")
for msg in raw_conversation:
    print(f"  {msg}")
print("\n转换后的 Qwen 格式：")
print(qwen_formatted_text)

# ========================================
# 第二部分：工具调用系统
# ========================================

"""
在 RL-Factory 中，AI 不仅能对话，还能调用工具（如搜索、计算等）。
这需要在 prompt 中告诉模型有哪些工具可用。
"""

# 工具定义示例
tools_definition = [
    {
        "name": "search",
        "description": "搜索互联网信息",
        "parameters": {
            "query": "搜索关键词"
        }
    },
    {
        "name": "calculator",
        "description": "进行数学计算",
        "parameters": {
            "expression": "数学表达式"
        }
    }
]

# 带工具的系统提示
system_with_tools = """你是一个有帮助的助手。你可以使用以下工具：

1. search: 搜索互联网信息
   参数: query (搜索关键词)
   
2. calculator: 进行数学计算
   参数: expression (数学表达式)

当需要使用工具时，请使用以下格式：
<tool_call>
{"name": "工具名", "arguments": {"参数名": "参数值"}}
</tool_call>"""

print("\n" + "=" * 60)
print("工具系统：如何在 Prompt 中定义工具")
print("=" * 60)
print("\n带工具的系统提示：")
print(system_with_tools)

# ========================================
# 第三部分：理解三种工作模式
# ========================================

"""
get_prompt 方法的三种模式对应对话的不同阶段：
"""

class PromptModeExamples:
    
    def mode_initial_example(self):
        """
        模式1：initial（初始模式）
        用途：开始一个新的对话或问题
        场景：用户刚提出问题时
        """
        
        # 输入数据：完整的对话历史
        input_data = [
            {"role": "system", "content": "你是一个有帮助的助手，可以使用搜索工具"},
            {"role": "user", "content": "北京今天的天气如何？"}
        ]
        
        # 生成的 prompt（带工具定义）
        generated_prompt = """<|im_start|>system
你是一个有帮助的助手，可以使用搜索工具

可用工具：
- search: 搜索互联网信息<|im_end|>
<|im_start|>user
北京今天的天气如何？<|im_end|>
<|im_start|>assistant
"""
        
        print("\n【Initial 模式示例】")
        print("场景：用户提出新问题")
        print("输入：", input_data)
        print("生成的 Prompt：")
        print(generated_prompt)
        print("注意：末尾的 <|im_start|>assistant 表示等待 AI 回复")
        
        return generated_prompt
    
    def mode_tool_call_example(self):
        """
        模式2：tool_call（工具调用模式）
        用途：处理工具执行的结果
        场景：AI 调用了工具，工具返回结果后
        """
        
        # AI 先输出了工具调用请求
        ai_tool_request = """让我搜索一下北京今天的天气。
<tool_call>
{"name": "search", "arguments": {"query": "北京今天天气"}}
</tool_call>"""
        
        # 工具执行后返回的结果
        tool_result = """搜索结果：
北京今天晴天，温度 15-25°C，空气质量良好。"""
        
        # 使用 tool_call 模式生成的 prompt
        generated_prompt = """<|im_start|>tool
搜索结果：
北京今天晴天，温度 15-25°C，空气质量良好。<|im_end|>
<|im_start|>assistant
"""
        
        print("\n【Tool Call 模式示例】")
        print("场景：工具返回结果后")
        print("AI 的工具调用请求：")
        print(ai_tool_request)
        print("\n工具返回的结果：")
        print(tool_result)
        print("\n生成的 Prompt（注意 role=tool）：")
        print(generated_prompt)
        print("注意：现在 AI 需要基于工具结果生成最终回答")
        
        return generated_prompt
    
    def mode_assistant_response_example(self):
        """
        模式3：assistant_response（助手响应模式）
        用途：处理 AI 的中间响应
        场景：在多轮对话中，AI 需要继续之前的回复
        """
        
        # AI 的中间回复
        assistant_content = "让我帮你查询更详细的信息..."
        
        # 使用 assistant_response 模式
        generated_prompt = """<|im_start|>assistant
让我帮你查询更详细的信息...<|im_end|>
<|im_start|>assistant
"""
        
        print("\n【Assistant Response 模式示例】")
        print("场景：AI 需要继续回复")
        print("AI 的中间内容：", assistant_content)
        print("生成的 Prompt：")
        print(generated_prompt)
        
        return generated_prompt

# 执行示例
print("\n" + "=" * 60)
print("三种工作模式详解")
print("=" * 60)

examples = PromptModeExamples()
examples.mode_initial_example()
examples.mode_tool_call_example()
examples.mode_assistant_response_example()

# ========================================
# 第四部分：完整的多轮对话流程
# ========================================

print("\n" + "=" * 60)
print("完整多轮对话流程示例")
print("=" * 60)

class MultiTurnConversation:
    """演示一个完整的多轮对话，包含工具调用"""
    
    def simulate_conversation(self):
        print("\n【完整对话流程】\n")
        
        # 第1轮：用户提问
        print("=" * 40)
        print("第1步：用户提问")
        print("=" * 40)
        conversation = [
            {"role": "system", "content": "你是助手，可以使用搜索和计算工具"},
            {"role": "user", "content": "北京到上海的距离是多少？开车需要多久？"}
        ]
        print("对话历史：", conversation)
        print("\n使用 get_prompt(mode='initial') 生成：")
        prompt1 = """<|im_start|>system
你是助手，可以使用搜索和计算工具
[工具定义...]<|im_end|>
<|im_start|>user
北京到上海的距离是多少？开车需要多久？<|im_end|>
<|im_start|>assistant"""
        print(prompt1)
        
        # 第2轮：AI 调用工具
        print("\n" + "=" * 40)
        print("第2步：AI 决定调用搜索工具")
        print("=" * 40)
        ai_response1 = """我来帮您查询北京到上海的距离信息。
<tool_call>
{"name": "search", "arguments": {"query": "北京到上海距离 开车时间"}}
</tool_call>"""
        print("AI 输出：")
        print(ai_response1)
        
        # 第3轮：工具返回结果
        print("\n" + "=" * 40)
        print("第3步：工具返回结果")
        print("=" * 40)
        tool_result = "搜索结果：北京到上海直线距离约1084公里，开车距离约1200公里，需要12-14小时"
        print("工具结果：", tool_result)
        print("\n使用 get_prompt(mode='tool_call') 生成：")
        prompt2 = """<|im_start|>tool
搜索结果：北京到上海直线距离约1084公里，开车距离约1200公里，需要12-14小时<|im_end|>
<|im_start|>assistant"""
        print(prompt2)
        
        # 第4轮：AI 给出最终答案
        print("\n" + "=" * 40)
        print("第4步：AI 基于工具结果回答")
        print("=" * 40)
        ai_final_response = """根据搜索结果，北京到上海的距离信息如下：
- 直线距离：约1084公里
- 开车距离：约1200公里
- 开车时间：约12-14小时

如果您选择开车，建议中途休息，注意行车安全。"""
        print("AI 最终回答：")
        print(ai_final_response)
        
        # 第5轮：用户追问
        print("\n" + "=" * 40)
        print("第5步：用户追问")
        print("=" * 40)
        conversation.append({"role": "assistant", "content": ai_final_response})
        conversation.append({"role": "user", "content": "如果时速100公里，需要多久？"})
        print("新的用户问题：如果时速100公里，需要多久？")
        print("\n使用 get_prompt(mode='initial') 生成新一轮对话...")

# 执行多轮对话示例
multi_turn = MultiTurnConversation()
multi_turn.simulate_conversation()

# ========================================
# 第五部分：配置自定义
# ========================================

print("\n" + "=" * 60)
print("如何自定义配置 Prompt")
print("=" * 60)

config_explanation = """
在 RL-Factory 中，你可以通过以下方式自定义 prompt：

1. 【修改配置文件】 
   路径：verl/trainer/config/rl_factory_ppo_trainer.yaml
   
   env:
     enable_thinking: True      # 启用思考链
     max_prompt_length: 2048    # 最大长度
     config_path: 自定义工具配置文件路径

2. 【自定义工具定义】
   创建文件：envs/configs/my_tools.pydata
   
   [
       {'mcpServers': {
           'my_tool': {
               'command': 'python3',
               'args': ['path/to/my_tool.py']
           }
       }}
   ]

3. 【修改系统提示】
   在代码中自定义 system_content：
   
   system_prompt = "你是一个专门解决数学问题的助手..."
   conversation = [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": user_input}
   ]

4. 【控制生成行为】
   - add_generation_prompt=True：添加 <|im_start|>assistant 等待生成
   - add_generation_prompt=False：不添加，用于构建历史

5. 【启用特殊功能】
   enable_thinking=True：为支持思考链的模型（如 QwQ）添加思考标记
"""

print(config_explanation)

# ========================================
# 总结
# ========================================

print("\n" + "=" * 60)
print("核心要点总结")
print("=" * 60)

summary = """
1. Chat Template 是对话格式化规则，不同模型需要不同格式
2. get_prompt 的三种模式对应对话的不同阶段：
   - initial: 新对话/新问题
   - tool_call: 处理工具结果
   - assistant_response: AI 继续回复
3. 多轮对话通过不断拼接历史和新内容实现
4. 工具调用通过特殊标记 <tool_call> 实现
5. 可通过配置文件和代码两种方式自定义 prompt
"""

print(summary)

print("\n" + "=" * 60)
print("建议：运行这个文件查看完整输出，理解每个阶段的 prompt 构造")
print("python prompt_explanation.py")
print("=" * 60)
#!/bin/bash
# GRPO (Group Relative Policy Optimization) 训练脚本
# 用于复现 Search-R1 的多轮工具使用强化学习后训练

# 设置错误时退出(-e)，打印执行的命令(-x)
set -e -x

# ==================== 环境变量配置 ====================
# 基础模型路径：需要微调的语言模型（如 Qwen3-4B）
export MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B
# 奖励模型路径：用于评估生成质量的模型（如 QwQ-32B）
export REWARD_MODEL_PATH=/your/path/to/huggingface.co/Qwen/QwQ-32B

# 结果保存路径：训练过程中的checkpoints和日志
export RESULT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/RL-Factory/results

# ==================== 启动 GRPO 训练 ====================
python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    # === 算法配置 ===
    algorithm.adv_estimator=grpo\  # 使用GRPO作为优势估计器（而非标准PPO）
    
    # === 数据配置 ===
    data.train_files=data/nq_search/train.parquet\  # 训练数据：NaturalQuestions搜索任务
    data.val_files=data/nq_search/test.parquet\    # 验证数据
    data.train_batch_size=128\                      # 总批次大小（会被分配到多个GPU）
    data.max_prompt_length=4096\                    # 最大prompt长度（tokens）
    data.max_response_length=512\                   # 最大响应长度（tokens）
    
    # === Actor模型配置 ===
    actor_rollout_ref.model.path=$MODEL_PATH\       # 使用环境变量指定的模型路径
    actor_rollout_ref.model.use_remove_padding=True\  # 移除padding以提高效率
    actor_rollout_ref.model.enable_gradient_checkpointing=True\  # 梯度检查点以节省显存
    
    # === Actor优化器配置 ===
    actor_rollout_ref.actor.optim.lr=1e-6\          # 学习率（较小以保证稳定）
    actor_rollout_ref.actor.ppo_mini_batch_size=32\ # PPO小批次大小
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16\  # 每GPU微批次大小
    
    # === KL散度正则化 ===
    actor_rollout_ref.actor.use_kl_loss=True\       # 启用KL损失（防止偏离原始模型太远）
    actor_rollout_ref.actor.kl_loss_coef=0.001\     # KL损失系数
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\  # 低方差KL损失类型
    
    # === FSDP（全分片数据并行）配置 ===
    actor_rollout_ref.actor.fsdp_config.param_offload=True\      # 参数卸载到CPU
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\  # 优化器状态卸载到CPU
    actor_rollout_ref.actor.state_masking=True\     # 启用状态掩码（仅在有效位置计算损失）
    
    # === Rollout（生成）配置 ===
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16\  # 计算log概率的批次大小
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\  # 张量并行大小（1表示不使用）
    actor_rollout_ref.rollout.name=vllm\            # 使用vLLM作为推理引擎
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75\  # GPU显存利用率
    actor_rollout_ref.rollout.n=4\                  # 每个prompt生成4个响应（GRPO特性）
    actor_rollout_ref.rollout.max_turns=2\          # 最大对话轮数
    
    # === Reference模型配置 ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16\  # 参考模型的批次大小
    actor_rollout_ref.ref.fsdp_config.param_offload=False\  # 参考模型不卸载（保持在GPU）
    
    # === vLLM引擎配置 ===
    actor_rollout_ref.rollout.enforce_eager=False\  # 不强制eager模式（使用CUDA图加速）
    actor_rollout_ref.rollout.free_cache_engine=True\  # 释放缓存引擎以节省显存
    
    # === 环境（工具使用）配置 ===
    actor_rollout_ref.env.name=search\              # 使用search环境（对应envs/search.py）
    actor_rollout_ref.env.mcp_mode=stdio\           # MCP通信模式（标准输入输出）
    actor_rollout_ref.env.tool_manager=qwen3\       # 工具管理器类型
    actor_rollout_ref.env.enable_thinking=False\    # 是否启用思考模式（类似o1）
    actor_rollout_ref.env.config_path=envs/configs/mcp_tools.pydata\  # MCP工具配置文件
    actor_rollout_ref.env.use_process_reward=False\ # 不使用过程奖励（仅结果奖励）
    
    # === 奖励模型配置 ===
    reward_rollout.if_use_reward_rollout=False\     # 不使用奖励rollout（使用规则奖励）
    reward_rollout.rollout.tensor_model_parallel_size=4\  # 奖励模型张量并行
    reward_rollout.rollout.gpu_memory_utilization=0.65\  # 奖励模型显存利用率
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\  # 奖励模型路径
    reward_rollout.rollout.free_cache_engine=True\  # 释放缓存
    reward_rollout.rollout.response_length=2048\    # 奖励模型最大响应长度
    reward_model.reward_manager=parallel\           # 并行奖励管理器
    
    # === 算法控制参数 ===
    algorithm.kl_ctrl.kl_coef=0.001\                # KL控制系数（动态调整）
    
    # === 训练器配置 ===
    trainer.critic_warmup=0\                        # Critic预热步数（0表示不预热）
    trainer.logger=['tensorboard']\                 # 使用TensorBoard记录
    trainer.project_name='GRPO_search'\             # 项目名称（用于日志）
    trainer.experiment_name='search_with_thinking'\ # 实验名称
    trainer.n_gpus_per_node=8\                      # 每节点GPU数量
    trainer.nnodes=1\                                # 节点数量（单机训练）
    trainer.val_before_train=False\                 # 训练前不验证
    trainer.default_local_dir=$RESULT_DIR\          # 本地保存目录
    trainer.default_hdfs_dir=null\                  # HDFS目录（不使用）
    trainer.save_freq=20\                            # 每20个epoch保存一次checkpoint
    trainer.test_freq=10\                            # 每10个epoch测试一次
    trainer.total_epochs=5 \                        # 总训练轮数
    $@ 2>&1 | tee grpo.log  # 接受额外参数，并将输出同时保存到grpo.log

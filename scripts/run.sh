#!/bin/bash

DATA_ROOT=../datasets
PROJECT_ROOT=./
outdir=${PROJECT_ROOT}/output
SEED=1



flag="--root_dir ${DATA_ROOT} \           # 数据集根目录
      --img_root ../datasets/RGB_Observations_nochange \  # RGB 图像观测数据目录
      --split MapGPT_72_scenes_processed \  # 使用的数据集分割名称
      --start 0 \                         # 数据集样本开始索引（包含）
      --end 1 \                           # 数据集样本结束索引（不包含）
      --output_dir ${outdir} \             # 实验结果输出目录
      --max_action_len 25 \                # 单个指令允许的最大动作步数
      --save_pred \                        # 保存预测的导航路径
      --stop_after 3 \                     # 允许LLM选择'stop'动作的最小（非停止）步数
      --llm gpt-4o \                       # 用于环境理解和规划的主要LLM
      --response_format json \             # 期望LLM返回JSON格式
      --max_tokens 1000 \                  # LLM生成响应的最大token数
      --dataset r2r \                      # 数据集类型 (Room-to-Room)
      --model_name gpt-4o \                # 用于日志/结果命名的模型标识
      --feedback_method gpt \              # 反馈方法 (使用GPT生成)
      --eval_type val_unseen \             # 评估类型 (验证集未见场景)
      --action_level 1 \                   # 动作抽象级别
      --action_gpt_model gpt-4o \          # 用于生成具体动作指令的LLM
      --max_prompt_token 60000 \           # 发送给LLM的提示最大token数
      --temperature 0.7 \                  # 控制LLM生成随机性 (0.7表示中等随机性)
      --error_mode error \                 # 遇到错误时的处理模式
      --num_beams 1 \                      # Beam search 数量 (1表示不使用)
      --seed $SEED \                       # 随机种子，用于复现性
      --enable_map_pruning \               # 启用动态地图修剪
      --map_pruning_step_threshold 10 \    # 地图节点被视为'旧'的步数阈值
      --pruning_keep_recent_steps 3 \      # 强制保留最近访问节点的步数窗口
      --pruning_start_step 15 \            # 开始执行修剪的步数
      --pruning_max_nodes_per_step 1 \     # 每步最多修剪节点数
      --w_time 1.0 \                       # 时间权重
      --w_degree 2.0 \                     # 度权重
      --w_frontier 5.0 \                   # 前沿权重
      --w_dist 0.5 \                       # 距离权重
      --log_pruning_scores \               # 启用详细得分日志记录
      $@"


python vln/main_gpt.py $flag

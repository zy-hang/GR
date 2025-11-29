# 2025 腾讯广告算法大赛 {Doga} 方案

复赛第26名
Score: 0.116742
NDCG@10: 0.0935916
HitRate@10: 0.168271


 # README.md

## 项目简介

本项目实现了一个基于 **HSTU (Heterogeneous Sequential Transformer with User modeling)** 架构的推荐系统模型，专为处理**用户-物品异构序列**而设计。模型融合了多模态特征（稀疏、连续、数组、Embedding）、时间特征（离散+连续）、动作类型建模，并采用了多项训练优化技巧。

---

## 核心特性

### 1. 模型架构亮点

#### 1.1 HSTU 多头注意力机制
- **分离的 Q/K/V/U 投影**：使用 `linear_dim` 和 `attention_dim` 分别控制线性维度和注意力维度
- **RoPE (Rotary Position Embedding)**：
  - 可选启用 `--use_rope`（默认开启）
  - 支持部分维度应用 `--rope_partial_dim`（默认 32）
  - Base theta 可配置 `--rope_theta`（默认 10000.0）
- **时间感知偏置**：
  - `RelativeBucketedTime_hstu`：基于时间间隔的 log 分桶偏置
  - `RelativePositionalBias_hstu`：可学习的相对位置偏置
- **PinRec 风格的 FiLM 调制**：
  - 根据动作类型（曝光/点击/转化）对序列表征进行条件归一化
  - 可学习的缩放因子 `r_scale` 和 `b_scale`

#### 1.2 特征工程
- **用户侧特征**：稀疏、数组、连续特征，通过独立 DNN + SE (Squeeze-and-Excitation) 加权
- **物品侧特征**：
  - **历史分支**（含时间特征）：稀疏、数组、连续、时间离散/连续、动作 Embedding、多模态 Embedding
  - **目标分支**（Plain，无时间）：稀疏、数组、连续、动态特征、多模态 Embedding
  - 两套独立的 SE 模块和 DNN 映射
- **时间特征离散化**（12 维）：
  - 星期几 (`201`)、小时 (`202`)、一天中的时段 (`203`)
  - 到下一事件的时间差分桶 (`204`)、到最近事件的时间差分桶 (`206`)
  - 会话间隔桶 (`207`, 4 档)、周数 (`208`)、月份 (`209`)、是否周末 (`210`)、周内小时 (`211`)、是否节假日/电商大促 (`212`)
- **物品动态特征**（10 维）：
  - 曝光/点击计数、CTR、分位数档位、异常检测、用户数、热度标记等

#### 1.3 损失函数与负采样
- **InfoNCE 风格的流式 log-sum-exp**：
  - 单遍计算所有负样本的 logsumexp，避免显存爆炸
  - 支持跨 step 的负样本缓存（Memory Bank）
- **负样本分桶采样**：
  - 根据物品曝光数分 6 个桶（≤3, ≤9, ≤54, ≤100, ≤1000, >1000）
  - 桶权重可配置（默认 `[0.251, 0.265, 0.274, 0.068, 0.113, 0.029]`）
  - 降低热门物品撞样率
- **Margin 机制**：
  - 点击 Margin (`--margin_click`, 默认 0.02)
  - 转化 Margin (`--margin_conv`, 默认 0.02)
- **正样本位置加权**：
  - 序列后部的正样本权重更高（线性插值，参数 `pos_weight_a=0.8`）

---

### 2. 训练优化技巧

#### 2.1 混合精度训练
- **BF16 自动混合精度**：
  - 前向/反向使用 `torch.autocast(dtype=torch.bfloat16)`
  - Loss 计算在 FP32 下完成
- **AdamW8bit 优化器**：
  - 使用 `bitsandbytes.optim.AdamW8bit` 降低显存占用
  - 分离 weight decay：LayerNorm/Bias 不加 decay，其余参数加 `--weight_decay`（默认 0.01）

#### 2.2 梯度管理
- **梯度累积**：
  - `--grad_accum_steps`（默认 2）：每 N 个 micro-batch 更新一次参数
  - 有效 batch size = `batch_size × grad_accum_steps`
- **梯度裁剪**：
  - 前 3000 步：`max_norm=20.0`
  - 3000 步后：`max_norm=2.5`（`--clip_norm`）
- **梯度检查点 (Gradient Checkpointing)**：
  - `--grad_checkpoint`（默认开启）
  - `--ckpt_every`（默认 1）：每 N 个 Attention 层做一次 checkpoint
  - 显存换时间，适合深层模型（默认 48 层）

#### 2.3 学习率调度
- **Cosine with Warmup**：
  - Warmup 比例：`--warmup_ratio`（默认 0.01）
  - 总步数 = `(len(train_loader) / grad_accum_steps) × num_epochs`
  - 半周期衰减（`num_cycles=0.5`）

#### 2.4 负样本缓存 (Memory Bank)
- **跨 step 缓存**：
  - `--neg_cache_steps`（默认 5）：保留前 K 个 step 的负样本
  - 每次训练时动态采样历史负样本
- **可选 CPU offload**：
  - `--neg_cache_on_cpu`：将缓存放到 CPU，计算前搬回 GPU
  - 节省显存，适合大 batch 场景

#### 2.5 正则化
- **L2 Embedding 正则**：
  - `--l2_emb`（默认 0.0001）
  - 对 `item_emb` 和 `item_emb_` 的权重施加 L2 惩罚
- **Dropout**：
  - `--dropout_rate`（默认 0.2）
  - 应用于 Embedding、U/V 输出、Attention 分数

#### 2.6 参数初始化
- **Embedding**：截断正态分布（std=0.02）
- **Linear**：截断正态分布（std=0.02），Bias 初始化为 0
- **LayerNorm**：Gamma=1, Beta=0
- **PinRec FiLM 层**：权重和 Bias 初始化为 0（保持初始无调制）

---

### 3. 数据处理

#### 3.1 序列构造
- **Left Padding**：
  - 序列长度固定为 `maxlen + 1`（默认 101 + 1 = 102）
  - 不足部分在左侧填充 0
- **User Token**：
  - 序列开头插入用户信息（`token_type=2`）
  - 若无用户信息，插入默认特征（`token_type=3`）
- **Item Token**：
  - 序列主体为物品 ID（`token_type=1`）
  - 每个 item 对应一个动作类型（曝光=0, 点击=1, 转化=2）

#### 3.2 特征填充
- **缺失值处理**：
  - 稀疏特征：填充为 Top-3 高频值的首位（而非 0）
  - 连续特征：填充为 0.0
  - 数组特征：填充为 `[0]`
  - 多模态 Embedding：填充为全 0 向量
- **时间特征注入**：
  - 仅对历史分支注入时间离散/连续特征
  - 目标分支（pos/neg）不含时间特征

#### 3.3 多模态特征
- **支持 6 种模态**：
  - `81` (32 维)、`82` (1024 维)、`83` (3584 维)、`84` (4096 维)、`85` (3584 维)、`86` (3584 维)
  - 通过 `--mm_emb_id` 指定（默认 `['81']`）
- **线性变换**：
  - 所有多模态 Embedding 统一映射到 32 维
  - 使用独立的 `nn.Linear` 层

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 1024 | 单卡 batch size |
| `--grad_accum_steps` | 2 | 梯度累积步数（有效 BS = batch_size × 此值） |
| `--lr` | 0.0015 | 学习率 |
| `--weight_decay` | 0.01 | 权重衰减（不含 LayerNorm/Bias） |
| `--warmup_ratio` | 0.01 | Warmup 步数占比 |
| `--clip_norm` | 2.5 | 梯度裁剪阈值（3000 步后） |
| `--maxlen` | 101 | 序列最大长度（实际为 maxlen+1） |
| `--hidden_units` | 512 | 隐藏层维度 |
| `--num_blocks` | 48 | Transformer 层数 |
| `--num_heads` | 16 | 多头注意力头数 |
| `--linear_dim` | 64 | U/V 线性维度 |
| `--attention_dim` | 64 | Q/K 注意力维度 |
| `--dropout_rate` | 0.2 | Dropout 比例 |
| `--l2_emb` | 0.0001 | Embedding L2 正则系数 |
| `--num_epochs` | 8 | 训练轮数 |
| `--grad_checkpoint` | True | 启用梯度检查点 |
| `--ckpt_every` | 1 | 每 N 层做一次检查点 |
| `--neg_cache_steps` | 5 | 负样本缓存步数 |
| `--neg_cache_on_cpu` | False | 将负样本缓存放到 CPU |
| `--margin_click` | 0.02 | 点击 Margin |
| `--margin_conv` | 0.02 | 转化 Margin |
| `--use_rope` | True | 启用 RoPE |
| `--rope_theta` | 10000.0 | RoPE base theta |
| `--rope_partial_dim` | 32 | RoPE 应用维度（0=全部） |
| `--load_time_intervals` | True | 加载时间间隔数据 |
| `--mm_emb_id` | ['81'] | 多模态特征 ID（可多选：81-86） |
| `--state_dict_path` | None | 断点续训模型路径 |

---

## 目录结构

```
JoyRec/
├── dataset.py          # 数据集类（MyDataset, MyTestDataset）
├── model.py            # HSTU 模型定义
├── main.py             # 训练主流程
├── tools.py            # 工具函数（损失计算、环境检查、断点保存等）
├── TencentGR_1k_new/   # 数据目录
│   ├── seq.jsonl
│   ├── seq_offsets.pkl
│   ├── item_feat_dict.json
│   ├── indexer.pkl
│   └── creative_emb/
├── logs/               # 训练日志
├── tf_events/          # TensorBoard 事件
└── checkpoints/        # 模型检查点
```

---

## 关键技术细节

### 1. 流式 InfoNCE 计算
```python
# 单遍流式 log-sum-exp，避免显存爆炸
lse = s_pos.clone()  # 初始化为正样本分数
for neg_bank in [历史缓存, 当前负样本]:
    for chunk in neg_bank:  # 分块处理
        s_neg = q @ chunk.T  # [M, chunk_size]
        lse = torch.logaddexp(lse, s_neg.logsumexp(dim=1))
loss = lse - s_pos  # 每个正样本的对比损失
```

### 2. 位置加权策略
```python
# 序列后部正样本权重更高
rank = mask.cumsum(dim=1) - mask  # [0, 1, 2, ..., N-1]
weights = (1 - a) + 2*a * (rank / N)  # a=0.8
loss = (loss_vec * weights).sum() / weights.sum()
```



---



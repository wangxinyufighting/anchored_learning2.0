# KDFlow Anchored Learning Patch

这是一个基于 KDFlow 的 Anchored Learning / Explicit Distributional Control 实现补丁。

论文方法对应关系：

- 先用标准 SFT 从 `p_base` 训练固定参考模型 `p_sft`。
- Anchored Learning 阶段重新从 `p_base` 初始化学生模型 `p_theta^(0)`。
- 每隔 `K = anchor_inner_epochs` 个 epoch 刷新外层快照 `p_theta^(t)`。
- 用 `p_theta^(t)` 与固定 `p_sft` 插值得到 anchor：
  - `logit`: `z_q = (1 - alpha) * z_snapshot + alpha * z_sft`
  - `probability`: `q = (1 - alpha) * softmax(z_snapshot) + alpha * softmax(z_sft)`
- 最小化 `KL(q_anchor || p_student)`。

## 文件说明

```text
install_patch.py
kdflow/algorithms/anchored_kd.py
scripts/prepare_alpaca_json.py
examples/anchored_learning/run_qwen2_5_anchor.sh
```

## 安装补丁

```bash
git clone https://github.com/songmzhang/KDFlow.git
cd KDFlow
pip install -e .
pip install flash_attn --no-build-isolation

# 假设这个补丁包解压在 /path/to/kdflow_anchored_learning_patch
python /path/to/kdflow_anchored_learning_patch/install_patch.py --repo .
```

补丁会修改以下 KDFlow 文件，并为原文件生成 `.bak.anchor` 备份：

```text
kdflow/arguments/distillation_args.py
kdflow/ray/train/student_group.py
kdflow/ray/train/student_actor.py
kdflow/trainer/off_policy_kd_trainer.py
```

同时会新增：

```text
kdflow/algorithms/anchored_kd.py
```

## 你的 JSON 数据格式

原始数据：

```json
[
  {"instruction": "xxx", "input": "xxx", "output": "xx"},
  {"instruction": "xxx", "input": "xxx", "output": "xx"}
]
```

先转换成 KDFlow 默认 SFTDataset 更容易读取的 prompt/response 格式：

```bash
python /path/to/kdflow_anchored_learning_patch/scripts/prepare_alpaca_json.py \
  --input_file data/train.json \
  --output_file data/train.kdflow.json \
  --template simple
```

输出格式：

```json
[
  {"input": "instruction + optional input", "output": "xx"}
]
```

训练时使用：

```bash
--train_dataset_path data/train.kdflow.json \
--input_key input \
--output_key output
```

KDFlow 的 `apply_chat_template` 默认是 `True`，因此 `input` 会被当作 user message，`output` 会被当作 assistant message。对 Qwen/Llama-Instruct 这类带 chat template 的模型通常不用关闭它。

## 一键示例

在 KDFlow 仓库根目录执行：

```bash
cp -r /path/to/kdflow_anchored_learning_patch/scripts .
cp -r /path/to/kdflow_anchored_learning_patch/examples .

BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
RAW_JSON=data/train.json \
WORK_DIR=outputs/anchored_qwen \
NUM_GPUS=1 \
TEACHER_TP_SIZE=1 \
GLOBAL_BSZ=16 \
MICRO_BSZ=1 \
bash examples/anchored_learning/run_qwen2_5_anchor.sh
```

## 两阶段手动命令

### Stage 1：训练固定 SFT reference，即 `p_sft`

```bash
python -m kdflow.cli.train_sft \
  --student_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --train_dataset_path data/train.kdflow.json \
  --input_key input \
  --output_key output \
  --num_nodes 1 \
  --num_gpus_per_node 1 \
  --num_epochs 3 \
  --train_batch_size 16 \
  --micro_train_batch_size 1 \
  --learning_rate 1e-5 \
  --bf16 \
  --gradient_checkpointing \
  --chunked_loss_size 2048 \
  --save_path outputs/sft_ref
```

### Stage 2：Anchored Learning，从 `p_base` 重新初始化学生模型

论文默认配置是 `alpha=0.5`、`logit` 插值、`K=5` inner-loop epochs、`T=5` outer iterations。因此这里总 epoch 数设置为 `25 = K * T`。

```bash
python -m kdflow.cli.train_kd_off_policy \
  --student_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --teacher_name_or_path outputs/sft_ref \
  --train_dataset_path data/train.kdflow.json \
  --input_key input \
  --output_key output \
  --num_nodes 1 \
  --num_gpus_per_node 1 \
  --teacher_tp_size 1 \
  --teacher_pp_size 1 \
  --teacher_dp_size 1 \
  --kd_algorithm anchored_kd \
  --kd_loss_fn kl \
  --kd_ratio 1.0 \
  --anchor_alpha 0.5 \
  --anchor_interpolation logit \
  --anchor_inner_epochs 5 \
  --anchor_snapshot_mode model \
  --num_epochs 25 \
  --train_batch_size 16 \
  --micro_train_batch_size 1 \
  --learning_rate 1e-5 \
  --bf16 \
  --gradient_checkpointing \
  --chunked_loss_size 2048 \
  --save_path outputs/anchored
```

## 关键参数

```text
--kd_algorithm anchored_kd
--kd_ratio 1.0
--anchor_alpha 0.5
--anchor_interpolation logit        # logit 或 probability
--anchor_inner_epochs 5             # paper K
--num_epochs 25                     # paper T * K
--anchor_snapshot_mode model        # exact frozen p_theta^(t)
```

`--anchor_snapshot_mode model` 会在每个 student actor 内保留一个冻结模型副本，最贴近论文，但会增加显存/内存开销。如果你的 KDFlow/FSDP 环境不允许 deepcopy 或显存不足，可以改成：

```bash
--anchor_snapshot_mode detached_current
```

这个模式会用“当前 step 的 student logits detach”近似 `p_theta^(t)`，资源更省，但不再是严格的外层冻结快照版本。

## 说明

- 这个实现复用 KDFlow 的 off-policy KD 训练链路、Ray actor、teacher hidden cache、chunked loss 思路。
- `p_sft` 必须是和 `p_base` 同 tokenizer / vocab 的模型，因为 anchor 插值在 token logits/probabilities 上进行。
- 推荐先不用 LoRA 跑通全量小模型；如果使用 LoRA，请确认 SFT reference 能被 KDFlow teacher/SGLang 正常加载，必要时先 merge adapter。

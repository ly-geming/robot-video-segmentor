# 🚀 快速开始（我们当前配置）

## 1) 进入项目目录

```bash
cd /data/LFT-W02_data/shuaizhou/human_video_data/video2tasks
```

## 2) 激活环境

```bash
conda activate annotate_sz
```

## 3) 安装项目（首次或代码更新后）

```bash
pip install -e .
```

## 4) 填写你的 API Key（只需一次）

编辑文件：`config_ours.yaml`

把下面字段改成你的 key：

```yaml
worker:
  openai_chat:
    api_key: "在这里填写你的AIHubMix Key"
```

当前我们使用的后端与模型：

```yaml
worker:
  backend: "openai_chat"
  openai_chat:
    base_url: "https://aihubmix.com/v1"
    model: "qwen3.5-397b-a17b"
```

## 5) 启动（两个终端）

终端 1（启动服务端）：

```bash
cd /data/LFT-W02_data/shuaizhou/human_video_data/video2tasks
conda activate annotate_sz
v2t-server --config config_ours.yaml
```

终端 2（启动 Worker）：

```bash
cd /data/LFT-W02_data/shuaizhou/human_video_data/video2tasks
conda activate annotate_sz
v2t-worker --config config_ours.yaml
```

## 6) 可选：先验证配置

```bash
python -m video2tasks.cli.validate_config --config config_ours.yaml
```

---

💡 说明

- 你可以继续用自己的配置文件名（`config_ours.yaml`），不需要改默认 `config.yaml`。
- 可以启动多个 Worker 并行处理（开多个终端执行同一条 `v2t-worker --config config_ours.yaml` 即可）。

## 工作原理（简要）

1. 数据扫描  
`v2t-server` 会按 `datasets` 扫描 `root/subset`，把每个子目录当作一个 sample，并在 sample 目录里找 `Frame_*.mp4`。
2. 分窗与抽帧  
对每个视频按 `window_sec/step_sec` 切重叠窗口；每个窗口均匀抽 `frames_per_window` 帧，统一缩放为 `target_width/target_height` 后转成 PNG base64。
3. Worker 推理  
`v2t-worker` 从 server 拉窗口任务，调用后端模型（当前是 `openai_chat`）得到 `vlm_json`，核心字段是：

- `transitions`: 窗口内切换点索引
- `instructions`: 各子任务文字描述
- `thought`: 可选推理说明

1. 结果落盘  
每个窗口结果会追加写入 `windows.jsonl`（一行一个窗口 JSON）。  
所有窗口完成后，server 聚合切分点并生成最终 `segments.json`，同时写入 `.DONE` 标记。
2. 断点续跑  
再次启动会跳过已有 `.DONE` 的 sample，只处理未完成部分。

## `windows.jsonl` 是什么

- 这是窗口级中间结果文件，不是最终结果。
- 每行类似：
  - `task_id`: 如 `demo::test_one_w0`
  - `window_id`: 窗口编号
  - `vlm_json`: 模型返回的 `thought/transitions/instructions`
- 最终请看同目录下的 `segments.json`。


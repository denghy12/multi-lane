# 服务器环境恢复记录（2026-09-10）

## 结论

本次服务器重置只清除了容器层状态。挂载在 `/mnt/haoyuan/workspace/` 下的代码工作树、
EMOTIC 数据集、CLIP/Face 权重、日志和实验结果均保留；原 `/opt/conda/envs/ddp`、
`/opt/conda/envs/cocoer-preprocess` 与 `/root/.ssh/` 中的授权信息丢失。

不要使用重置后 base 环境的 Python 3.11 / PyTorch 2.6 继续复现实验。历史正式实验记录的是
Python 3.9、PyTorch 2.0.1+cu118 与 CUDA 11.8 wheel。

## 已恢复环境

### 训练环境 `/opt/conda/envs/ddp`

- Python 3.9.23（当前 conda-forge 不再提供审计记录中的 3.9.25，属于补丁级差异）
- PyTorch 2.0.1+cu118
- torchvision 0.15.2+cu118
- torchaudio 2.0.2+cu118
- NumPy 1.26.4
- SciPy 1.13.1
- Pillow 9.2.0
- timm 0.6.7
- open-clip-torch 2.17.1
- matplotlib 3.9.0
- seaborn 0.12.2
- torchmetrics 0.11.2
- tqdm 4.65.0
- wandb 0.15.5

恢复命令：

```bash
/opt/conda/bin/conda create -n ddp python=3.9 pip -y
/opt/conda/envs/ddp/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118
cd /mnt/haoyuan/workspace/multi-lane-main
/opt/conda/envs/ddp/bin/python -m pip install -r requirements.txt scipy==1.13.1
```

### Face 预处理环境 `/opt/conda/envs/cocoer-preprocess`

- Python 3.9.23
- PyTorch/torchvision/torchaudio 与训练环境相同（项目包初始化和数据变换需要）
- NumPy 1.26.4
- SciPy 1.13.1
- Pillow 11.3.0
- OpenCV headless 4.10.0.84
- ONNXRuntime 1.18.0
- InsightFace 0.7.3
- scikit-learn 1.6.1
- scikit-image 0.24.0
- albumentations 2.0.8

恢复命令：

```bash
/opt/conda/bin/conda create -n cocoer-preprocess python=3.9 pip -y
/opt/conda/envs/cocoer-preprocess/bin/python -m pip install \
  numpy==1.26.4 scipy==1.13.1 pillow==11.3.0 \
  opencv-python-headless==4.10.0.84 onnxruntime==1.18.0 insightface==0.7.3
/opt/conda/envs/cocoer-preprocess/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118
```

## 保留资产核验

- EMOTIC：`/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC`
- CLIP ViT-B/16：`/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt`
- SCRFD：`/mnt/haoyuan/workspace/baseline_sources/cocoer_insightface/models/buffalo_l/det_10g.onnx`
- SCRFD SHA-256：`5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`
- 正式实验根目录：`/mnt/haoyuan/workspace/emotic_benchmark_runs`

## 验收结果

- 两套环境 `pip check`：通过，无依赖冲突。
- CUDA：8 张 RTX 4090 均可见；PyTorch CUDA build 11.8；GPU 运算通过。
- 项目完整单测：169/169 通过。
- Image-token Adapter GPU smoke：layer1/b32/scale0.1、BCE + Adapter ASL、AMP/TF32，
  forward/backward 和梯度路由通过。
- Face CPU smoke：真实 EMOTIC 图片经 SCRFD 成功检测到人脸。
- Face crop → frozen CLIP GPU smoke：4 个真实样本通过。
- 端到端训练 smoke：Full、seed0、OOF fold0、task0、1 epoch，56 steps、skipped=0，
  validation mAP 39.543071；未读取 test、未保存 checkpoint。

服务器验证日志位于实验 worktree：

- `logs/server_env_restore_unittest_20260910.log`
- `logs/server_env_restore_adapter_gpu_smoke_20260910.log`
- `logs/server_env_restore_face_gpu_smoke_20260910.log`

端到端 smoke 产物位于：

`/mnt/haoyuan/workspace/emotic_benchmark_runs/server_env_restore_smoke_20260910/full_fold0_task0`

## SSH/Git 恢复

- 已重新把本机 `~/.ssh/id_ed25519.pub` 写入服务器 `authorized_keys`，无密码登录通过。
- 本机 `~/.ssh/config` 已为该服务器固定端口、IdentityFile 和 agent forwarding。
- 普通 `ssh 172.31.214.226` 会转发本机已加载的 GitHub key；服务器内
  `ssh -T git@github.com` 已通过，因此保持远端 `git@github.com:denghy12/multi-lane.git` 不变。
- 私钥没有复制到服务器。若容器再次重置，只需重新执行 `ssh-copy-id`，随后按本文重建环境。

`/mnt/haoyuan/workspace/multi-lane-main-test-only` 仍停在 `0b59138`。该工作树原有的两个已修改
Python 文件、两个未跟踪脚本和 `datasets` 项均保持原样；本次恢复未清理、覆盖或提交它们。

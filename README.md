# Ai-workbench

个人ai-coding学习项目

## 安装环境

Ubuntu 系统 Python 不能直接 `pip install`（PEP 668）。在仓库根一次性建虚拟环境并装依赖：

```bash
python3 -m venv .venv && .venv/bin/pip install -r hot_project/requirements.txt
```

之后用 `.venv` 里的解释器，不要再用系统的 `pip`：

```bash
.venv/bin/python3 -m hot_project
# 或更新代码并重启 Web：
bash hot_project/pull_restart.sh
```

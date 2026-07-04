# Workspace Hygiene

## 目的

任务中的缓存、日志和一次性调试脚本若长期残留，会掩盖正式改动并增加误提交风险。仓库使用安全白名单清理：默认只检查并打印候选项，只有显式传入 `--apply` 才删除。

## 自动清理范围

- 缓存目录：`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`.ipynb_checkpoints/`；
- 临时目录：`tmp/`、`temp/`；
- 临时文件：`*.pyc`、`*.log`；
- 项目根目录的一次性脚本：`debug_*.py`、`try_*.py`、`tmp_*.py`、`temp_*.py`、`check_*_tmp.py`、`test_*_tmp.py`；
- `scratch/` 仅在显式传入 `--include-scratch` 时清理。

清理脚本不会跟随符号链接，也不会扫描本地数据集目录或虚拟环境目录。即使 `src/`、`tests/`、`configs/`、`reports/` 或 `demo/` 下存在名为 `tmp`、`temp`、`scratch` 的目录，脚本也不会整目录删除其中的正式文件。

## 自动保护范围

清理脚本不会按正式代码或文档的扩展名做宽泛删除。以下内容绝不会自动清理：

- `README.md`、`AGENTS.md`、`project_mvp_plan.md`；
- `src/`、`data/`、`tests/`、`configs/`、`reports/`、`demo/` 中的正式 `.py`、`.md` 和 `.yaml` 文件；
- `scripts/check_env.py`、requirements、environment 配置；
- 数据集、模型权重和不属于清理白名单的用户文件。

不确定文件默认保留，并在任务总结中标记为“需要用户确认”。

## 使用方法

Dry-run：

```bash
python scripts/clean_workspace.py
```

执行清理：

```bash
python scripts/clean_workspace.py --apply
```

同时清理 `scratch/`：

```bash
python scripts/clean_workspace.py --apply --include-scratch
```

显示更详细的候选原因和路径：

```bash
python scripts/clean_workspace.py --verbose
```

## 提交前检查

建议每次提交前依次执行：

```bash
git status --short
python scripts/clean_workspace.py
python scripts/clean_workspace.py --apply
git status --short
```

执行 `--apply` 前必须先查看 dry-run 清单；若候选项用途不明确，停止清理并请求用户确认。

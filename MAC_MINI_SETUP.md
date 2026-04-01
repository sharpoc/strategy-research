# Mac mini 初始化说明

这份文档用于把研究与执行链路迁到新的 `Mac mini`。

## 目标

让 `Mac mini` 成为策略大脑，负责：

- 晚间选股
- 缺天补数
- 回测
- 参数优化
- 盘中跟踪刷新

## 需要拉取的仓库

```bash
git clone git@github.com:sharpoc/strategy-research.git
```

如需本地联调展示服务，再拉：

```bash
git clone git@github.com:sharpoc/strategy-lab.git
```

## 需要准备的环境

- Python
- PostgreSQL
- Git / SSH

## 必需配置

在新电脑上重新准备：

- `TUSHARE_TOKEN`
- `TUSHARE_HTTP_URL`
- 本地 PostgreSQL 连接
- 任何绝对路径型环境变量

## 建议复制的本地目录

如果希望沿用旧电脑的断点和研究成果，建议复制：

- `/Users/lvxue/work/量化/output/`

这个目录不在 Git 中，但会包含：

- Tushare 导出
- screen 导出
- replay 结果
- research_backtests
- 缓存与 checkpoint

## 迁移顺序

1. 配 GitHub SSH
2. 拉取 `strategy-research`
3. 安装依赖
4. 初始化 PostgreSQL
5. 配环境变量
6. 复制旧电脑 `output/`（如果需要）
7. 先跑单日 runner 验证
8. 再考虑正式接管定时任务

## 当前建议优先接手的研究任务

如果 `Mac mini` 先接“当前最值的研究”，建议优先是：

1. `核心高管连增臻选`
2. 继续优化它的 `final` 轻确认层
3. 不动 `stage1` 入口，不急着接网页和线上

对应交接文档：

- [/Users/lvxue/work/量化/docs/CORE_MANAGEMENT_LIGHT_FINAL_HANDOFF.md](/Users/lvxue/work/量化/docs/CORE_MANAGEMENT_LIGHT_FINAL_HANDOFF.md)

建议先在 `Mac mini` 上验证两条命令：

```bash
python3 /Users/lvxue/work/量化/scripts/run_tushare_core_management_accumulation_strategy.py --end-date 20260320
python3 /Users/lvxue/work/量化/scripts/run_core_management_final_review.py --stats-json /tmp/core_management_6m_stats.json
```

## 注意事项

- 仓库只提交代码、配置、文档
- 不要把 `output/`、行情数据、导出结果提交进 Git
- 上下文迁移优先看：
  - `README.md`
  - `CURRENT_STATE.md`
  - 本文件

# Mac mini 定时执行说明

这套链路默认按下面的职责拆开：

- `strategy-research`
  - 负责重任务与定时执行
  - 默认入口：`ops/run_mac_mini_holder_pipeline.sh`
- `strategy-lab`
  - 负责接收结果、入库、页面展示
  - 通过 `LAB_SYNC_SCRIPT` 作为后置同步钩子接入

## 先准备

1. 复制 `.env.mac_mini.example` 为 `.env.mac_mini`
2. 填好 `TUSHARE_TOKEN`
3. 如果已经有服务仓库同步脚本，填上 `LAB_SYNC_SCRIPT`

## 手工试跑

```bash
cd /Users/eagod/ai-dev/策略实验室/strategy-research
chmod +x ops/run_mac_mini_holder_pipeline.sh ops/install_launchd_holder_pipeline.sh
cp .env.mac_mini.example .env.mac_mini
./ops/run_mac_mini_holder_pipeline.sh
```

默认行为：

- 跑当天 `holder` 单日 pure-python runner
- 自动开启 `--resume-existing --require-complete`
- 成功后，如果存在 `LAB_SYNC_SCRIPT`，继续触发同步

## 安装定时任务

```bash
cd /Users/eagod/ai-dev/策略实验室/strategy-research
./ops/install_launchd_holder_pipeline.sh
```

默认安装为：

- 标签：`com.sharpoc.strategyresearch.holder.pipeline`
- 时间：工作日 `21:30`

也可以临时指定：

```bash
HOUR=22 MINUTE=5 ./ops/install_launchd_holder_pipeline.sh
```

## 建议的服务仓库同步方式

建议在 `strategy-lab` 里单独放一个脚本，例如：

```bash
/Users/lvxue/work/策略实验室/scripts/sync_holder_result_from_mac_mini.sh
```

它负责做三件事：

1. 接收 `TRADE_DATE`
2. 把研究仓库当天结果安全入库或推接口
3. 必要时刷新页面缓存

这样研究仓库不用硬编码服务端细节，后续你要改入库方式、改 API、改 ECS 地址，都只改服务仓库。

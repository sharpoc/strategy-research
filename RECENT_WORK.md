# Recent Work

## 2026-06-04 · 停用全部 5 个 Mac mini 策略定时器（策略已前台隐藏）

线上 4 条策略（财报催化/股东户数、筹码增强、事件信念、尾盘反抽）已在展示站前台隐藏：首页不再露出、`/strategies` 返回 404、`/api/internal/strategies/.../tracking-snapshots` 无前台消费。这些盘后/盘中 pipeline 产出的数据已没人看，故停用对应定时器以省 tushare 频率与机器资源。

### 操作（本机 Mac mini，可逆，未改任何脚本/算法）
- 停用 5 个 launchd（`~/Library/LaunchAgents/`，工作日）：
  - `tracking.pipeline` 09:40–15:10（9 次，盘中跟踪这 4 策略持仓）
  - `rebound.pipeline` 14:30（尾盘反抽）
  - `holder.pipeline` 21:30（财报催化 + 股东户数）
  - `holder-chip.pipeline` 22:15（筹码增强）
  - `event-conviction.pipeline` 22:45（事件信念）
- 方式：`launchctl bootout gui/$(id -u)/<label>` + `com.sharpoc.strategyresearch.*.plist` → `.plist.disabled`。plist 与脚本内容一字未动。
- 验证：`launchctl list | grep sharpoc` 仅剩 strategy-lab 的 `market-reports`、`auction-top3-notify`。

### 恢复方法
把对应 `.plist.disabled` 改回 `.plist`，`launchctl bootstrap gui/$(id -u) <plist>`（或重跑 `ops/install_launchd_*_pipeline.sh`）。

### 注意
- 不影响线上 `/limit-up`、`/top3-review`（数据来自 strategy-lab 的两个定时器，仍在跑）。
- 仅停调度，未删数据、未改前台隐藏逻辑（隐藏在 strategy-lab 侧）。

## 2026-05-11

- 收口本地 `main`：合入远端 `origin/main` 的 `docs: add Tushare 15000 research plan`。
- 新增收口提交：
  - `0123ac7 Add Mac mini strategy pipelines`
  - `115f918 Merge origin main`
- 当前本地 `main` 相对 `origin/main`：`ahead 3`，尚未推送远端。
- 已纳入 Mac mini 多策略执行链路：
  - 财报催化默认占用 `holder_increase_screening` 槽位。
  - 事件信念策略晚间 pipeline。
  - 尾盘反抽 `tail_rebound_screening` pipeline。
  - 盘中 tracking 默认覆盖 4 条线上策略。
- 已纳入 `vendor/tushare-stock-rater` 代码，用于尾盘反抽候选与相关测试。
- 已避免提交运行缓存：`vendor/tushare-stock-rater/data/cache/` 已加入 `.gitignore`。
- 本地未提交且应继续保留为本地文件：
  - `.env.mac_mini`
  - `ops/state/`
- 最近验证：
  - `python3 -m unittest discover -s tests -v`
  - 结果：`3 tests OK`
  - `PYTHONPATH=vendor/tushare-stock-rater/src python3 -m unittest discover -s vendor/tushare-stock-rater/tests -v`
  - 结果：`8 tests OK`

## 2026-05-12

- 同步 `vendor/tushare-stock-rater` 运维增强：
  - `TushareClient.call()` 增加接口级日志，输出缓存命中、接口开始、成功行数、失败原因、耗时。
  - 增加默认 socket timeout，避免 Tushare 接口无限等待。
- 最近验证：
  - `PYTHONPATH=vendor/tushare-stock-rater/src python3 -m unittest discover -s vendor/tushare-stock-rater/tests -v`
  - 结果：`9 tests OK`

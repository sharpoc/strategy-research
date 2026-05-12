# Recent Work

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

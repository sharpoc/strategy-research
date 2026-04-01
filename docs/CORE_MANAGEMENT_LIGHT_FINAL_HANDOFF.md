# 核心高管连增臻选：轻确认 final 交接说明

最后更新：`2026-04-01`

## 当前定位

这条策略当前只在本地研究层使用：

- 策略名：`核心高管连增臻选`
- 内部 id：`core_management_accumulation`
- 类型：低频、严格的事件驱动策略
- 当前状态：
  - 不接网页
  - 不接测试环境
  - 不接线上定时
  - 只供 `Mac mini` 后续继续研究

## 代码入口

- 策略模块：
  - [/Users/lvxue/work/量化/scripts/core_management_accumulation_strategy.py](/Users/lvxue/work/量化/scripts/core_management_accumulation_strategy.py)
- 单日 runner：
  - [/Users/lvxue/work/量化/scripts/run_tushare_core_management_accumulation_strategy.py](/Users/lvxue/work/量化/scripts/run_tushare_core_management_accumulation_strategy.py)
- `final` 轻确认对拍脚本：
  - [/Users/lvxue/work/量化/scripts/run_core_management_final_review.py](/Users/lvxue/work/量化/scripts/run_core_management_final_review.py)

## 策略当前共识

这条策略已经过了“能不能识别事件”的阶段，当前结论是：

- `stage1` 候选池是有价值的
- 旧版 `final` 过度强调“贴近增持成本”，而且会在同一轮事件波上重复给信号
- 当前优化方向已经切到：
  - `stage1` 负责发现候选池
  - `final` 只做轻确认 + 去重

一句话：

`stage1` 里有肉，当前最值得优化的是 `final` 最后一层怎么更稳定地挑出那一只。

## 已实现的增强项

当前策略模块里已经做进去的增强项：

- 最近一轮增持使用加权均价，不用简单均价
- `holder_type + holder_name` 身份权重映射
- 支持“同一交易日内多位核心管理层密集买入”作为连续性信号
- 当前价相对增持成本是核心评分项
- 增持后 `5~10` 日量价结构单独打分
- `融资净买入 / 主力净流 / 换手健康度` 只做辅助加分
- `final` 层已经加入：
  - `wave_signature`
  - 轻确认排序
  - 同波去重
  - 重复信号限制

## 最近 6 个月的基线结论

基线统计报告：

- [/Users/lvxue/work/量化/output/research_backtests/core_management_accumulation_final_vs_stage1_6m_report_20260326.md](/Users/lvxue/work/量化/output/research_backtests/core_management_accumulation_final_vs_stage1_6m_report_20260326.md)

区间：

- `2025-09-25 ~ 2026-03-24`

基线结果：

- `final` 真信号：`10` 次
- `stage1` 样本：`29` 只

收益口径：`T+1 开盘 -> 第 3 / 5 / 10 个交易日收盘`

- 基线 `final`
  - `3日 +5.5768%`
  - `5日 +5.1976%`
  - `10日 +3.5992%`
- 基线 `stage1`
  - `3日 +2.9590%`
  - `5日 +6.3288%`
  - `10日 +8.1691%`

这说明：

- `final` 的短期爆发力还可以
- 但中期 `5/10` 日表现落后于 `stage1`
- 问题主要在 `final` 层排序/过滤，而不是事件识别入口

## 轻确认 final 的最新结论

最新对拍目录：

- [/Users/lvxue/work/量化/output/research_backtests/core_management_final_review_20260326_175213](/Users/lvxue/work/量化/output/research_backtests/core_management_final_review_20260326_175213)

关键文件：

- 汇总：
  - [/Users/lvxue/work/量化/output/research_backtests/core_management_final_review_20260326_175213/review_summary.json](/Users/lvxue/work/量化/output/research_backtests/core_management_final_review_20260326_175213/review_summary.json)
- 报告：
  - [/Users/lvxue/work/量化/output/research_backtests/core_management_final_review_20260326_175213/review_report.md](/Users/lvxue/work/量化/output/research_backtests/core_management_final_review_20260326_175213/review_report.md)

最新轻确认 `final` 结果：

- 真信号：`4` 次
- 重复信号痕迹：`0`
- 新版 `final`
  - `3日 +4.7831%`
  - `5日 +9.6100%`
  - `10日 +5.3594%`
- 基线 `stage1`
  - `3日 +2.9590%`
  - `5日 +6.3288%`
  - `10日 +8.1691%`

当前解释：

- 新版 `final` 已经把重复信号明显压下去了
- `5日` 表现已经优于 `stage1`
- 但 `10日` 仍低于 `stage1`

所以这条策略当前不是“没做好”，而是：

`轻确认 final` 已经走通，但还没最终定版，下一轮应继续优化中期持有表现。

## 当前最值得继续做的事

Mac mini 接手后，优先按这个顺序做：

1. 固定当前 `stage1` 入口，不再动入口识别
2. 继续只优化 `final` 轻确认层
3. 专门检查为什么这几类样本还能进新版 `final`
   - `600733.SH 北汽蓝谷`
4. 把关注点放在：
   - `10日` 表现能不能继续抬高
   - 是否还能进一步减少弱结构重复信号

当前不建议做的事：

- 不要先放宽事件入口
- 不要为了多出票去降低 `stage1` 门槛
- 不要先接网页或线上

## Mac mini 上的建议复现命令

### 1. 跑某一天的单日筛选

```bash
python3 /Users/lvxue/work/量化/scripts/run_tushare_core_management_accumulation_strategy.py \
  --end-date 20260320
```

### 2. 重跑 6 个月轻确认 `final` 对拍

```bash
python3 /Users/lvxue/work/量化/scripts/run_core_management_final_review.py \
  --stats-json /tmp/core_management_6m_stats.json
```

如果本地已有缓存较热，可以把 `--api-sleep-sec` 设低一些；如果接口不稳，就保守一点。

## 输出文件怎么看

单日 runner 输出：

- `screen_summary.json`
- `stage1_candidates.csv`
- `final_candidates.csv`
- `best_pick_candidate.csv`
- `event_wave_details.csv`

轻确认对拍输出：

- `optimized_final_signals.csv`
- `review_summary.json`
- `review_report.md`

## 交接结论

这条策略当前已经进入：

`可继续研究，但不要急着上线`

最准确的交接状态是：

- 入口识别已基本成立
- `stage1` 候选池有价值
- `final` 正在从“重过滤”改成“轻确认 + 去重”
- 当前版本已经明显优于旧版 `final`
- 下一步继续优化 `10日` 表现，而不是回头放宽入口

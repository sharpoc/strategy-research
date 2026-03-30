# 回测系统设计方案

这份设计文档面向当前 `/Users/lvxue/work/量化` 目录，目标不是推翻现有研究脚本，而是在现有基础上整理出一套可长期复用的回测系统。

核心目标：

- 以后新增策略时，只需要补“策略模块 + 配置 + 卖出规则”，不再重写整套回测链路
- 同时支持：
  - 价格形态策略
  - 事件驱动策略
  - 混合评分策略
- 回测、参数优化、walk-forward、研究报告都走同一套标准接口
- 与测试环境、线上环境彻底隔离

## 1. 当前现状

现在已经有一套可用但偏“研究脚本集合”的框架，核心文件包括：

- [run_price_strategy_regime_backtest.py](/Users/lvxue/work/量化/scripts/run_price_strategy_regime_backtest.py)
- [run_price_strategy_research_suite.py](/Users/lvxue/work/量化/scripts/run_price_strategy_research_suite.py)
- [run_price_strategy_walkforward.py](/Users/lvxue/work/量化/scripts/run_price_strategy_walkforward.py)
- [strategy_exit_rules.py](/Users/lvxue/work/量化/scripts/strategy_exit_rules.py)
- [research_backtest_utils.py](/Users/lvxue/work/量化/scripts/research_backtest_utils.py)
- [research_universe_filters.py](/Users/lvxue/work/量化/scripts/research_universe_filters.py)

这套框架已经能做：

- 逐日回放
- `T+1` 开盘买入
- 统一卖出规则
- 市场状态分层
- 研究态股票池过滤
- 研究态命名 preset
- 参数随机搜索
- walk-forward

但它还不够“完整系统”，主要问题是：

- 策略接入点还不统一
- 价格型和事件型策略的接法不一样
- 调试信息、交易明细、参数试验结果还没有统一标准 schema
- 研究结果是“能跑”，但还没沉淀成一套稳定的系统约定

当前已经补上的一层是：

- 用 [research_config_presets.py](/Users/lvxue/work/量化/scripts/research_config_presets.py) 统一维护研究态 preset
- `run_price_strategy_regime_backtest.py`
- `run_price_strategy_research_suite.py`
- `optimize_price_strategy_params.py`
- `run_price_strategy_walkforward.py`

现在都支持：

- `--strategy-config-preset`
- `--strategy-config-file`

并且口径是：

- 先加载 preset
- 再叠加显式文件覆盖

这样能避免“研究总入口、优化器、walk-forward 各自拿不同 JSON 文件”的隐性跑偏。

## 2. 系统设计原则

这套回测系统建议坚持 8 条原则：

1. 单一事实来源
- 同一策略的回测、优化、walk-forward、研究报告必须共用同一套信号计算函数。

2. 严格时间截断
- 任何数据都必须按 `as_of_date <= trade_date` 截断，避免前视偏差。

3. 研究与生产隔离
- 回测系统只写研究目录和研究缓存，不改测试环境、线上环境、网页服务。

4. 策略插件化
- 新策略必须通过统一协议接入，不能每条都写一套专用 runner。

5. 执行和信号分层
- “今天选什么”与“买了以后怎么卖”必须拆开。

6. 参数与逻辑分离
- 规则逻辑放代码，阈值放配置，方便后续优化。

7. 可复盘
- 每次回测都要能落出完整的 `summary / trades / equity / debug candidates / params`。

8. 先稳后快
- 先保证结果可信，再考虑并行、缓存、增量更新。

## 3. 建议的系统分层

建议把完整回测系统整理成下面 7 层。

### 3.1 数据层

职责：

- 拉取和缓存原始行情、事件、财务、筹码、指数数据
- 保证所有数据都能按日期历史回放

建议统一接口：

```python
class DataProvider:
    def get_trade_dates(self, start_date: str, end_date: str) -> list[str]: ...
    def get_stock_basic(self, as_of_date: str) -> pd.DataFrame: ...
    def get_daily_history(self, trade_dates: list[str]) -> pd.DataFrame: ...
    def get_event_snapshot(self, event_name: str, as_of_date: str) -> pd.DataFrame: ...
    def get_factor_snapshot(self, factor_name: str, as_of_date: str) -> pd.DataFrame: ...
```

建议支持两种模式：

- `api_mode`
  - 直接走 Tushare / 自定义接口
- `replay_mode`
  - 只读本地缓存和历史快照

这是后面处理 `星曜增持臻选` 最关键的一层。

当前已经开始落地的数据准备清单工具：

- [backtest_data_catalog.py](/Users/lvxue/work/量化/scripts/backtest_data_catalog.py)
- [audit_backtest_data_inventory.py](/Users/lvxue/work/量化/scripts/audit_backtest_data_inventory.py)

它们的作用是：

- 统一描述“每条策略依赖哪些缓存 / 快照 / 导出目录”
- 在本地直接盘点“当前是否已经具备 2~3 年研究条件”
- 在接新策略前先回答：缺的是价格数据、事件数据，还是历史快照

同时建议配套一个“数据补齐脚本”：

- [prepare_backtest_market_cache.py](/Users/lvxue/work/量化/scripts/prepare_backtest_market_cache.py)
- [prepare_backtest_data_for_strategy.py](/Users/lvxue/work/量化/scripts/prepare_backtest_data_for_strategy.py)

它负责把价格型策略最核心的三类数据先补齐：

- `trade_cal`
- `stock_basic_all`
- `daily_all_<trade_date>`

这样以后开始长区间回测前，先补缓存，再跑研究，流程会更稳。

进一步的理想入口是：

- 直接按策略补数，而不是手动输入几年
- 也就是 `prepare_backtest_data_for_strategy.py` 这类脚本读取策略元信息后，自己决定：
  - 这条策略至少需要几年历史
  - 如果用户给了回测起点，还要额外预留 `history_bars` 的 warmup 窗口
  - 当前本地缺多少
  - 应该先补价格数据，还是提醒补事件快照

### 3.2 股票池层

职责：

- 处理基础 universe 过滤
- 让所有策略都能复用同一套基础股票池逻辑

建议拆成两层：

- `universe_filter`
  - `ST / *ST / 北交所 / 科创板 / 退市整理 / 上市天数 / 价格 / 成交额`
- `candidate_filter`
  - 策略研究态过滤，比如只保留 `retest_hold`

现有 [research_universe_filters.py](/Users/lvxue/work/量化/scripts/research_universe_filters.py) 可以直接演进成这一层。

### 3.3 信号层

职责：

- 对每只股票计算当日策略特征
- 给出是否命中、买点类型、结构关键位、风险字段、分数

建议定义统一输出字段：

```python
{
  "ts_code": "...",
  "trade_date": "...",
  "signal": True,
  "signal_stage": "...",
  "raw_score": 0.0,
  "rank_score": 0.0,
  "buy_type": "...",
  "support_price": 0.0,
  "neckline_price": 0.0,
  "stop_reference": 0.0,
  "reason": "...",
  "risk_flags": "...",
}
```

再允许每条策略附加自己的扩展字段。

建议策略协议：

```python
class StrategyPlugin:
    strategy_id: str
    strategy_name: str
    history_bars: int

    def build_signal_snapshot(
        self,
        window_history: pd.DataFrame,
        stock_basic_df: pd.DataFrame,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> pd.DataFrame:
        ...
```

其中 `context` 至少包括：

- `trade_date`
- `market_regime`
- `as_of_date`
- `data_provider_meta`

### 3.4 排序与选股层

职责：

- 当日同一策略多个候选时如何排序
- 是否只选 `1` 只还是选前 `N` 只
- 是否启用研究态 gate

建议统一拆成：

- `score_builder`
- `candidate_ranker`
- `entry_gate`

这样做的好处是：

- 逻辑能拆清楚
- 你可以单独研究“分数有问题”还是“出手太多”

这点已经在 `龙门双阶` 和 `真实资金突破` 的研究里证明是必要的。

### 3.5 执行层

职责：

- 定义如何买入
- 定义滑点、手续费、印花税
- 定义资金分配

建议统一假设：

- 选股时点：交易日北京时间 `20:00`
- 入场口径：`T+1` 开盘价
- 默认仓位：
  - 单策略单票：`100%` 或 `1/N`
- 成本模型建议默认支持：
  - 买入滑点
  - 卖出滑点
  - 佣金
  - 印花税

建议先做成可配置：

```json
{
  "entry_mode": "next_open",
  "buy_slippage_bps": 10,
  "sell_slippage_bps": 10,
  "commission_bps": 3,
  "stamp_tax_bps": 10
}
```

### 3.6 卖出层

职责：

- 结构止损
- 趋势走弱止损
- 保本移动止损
- 目标位退出
- 时间退出

现有 [strategy_exit_rules.py](/Users/lvxue/work/量化/scripts/strategy_exit_rules.py) 已经是一个雏形。

建议未来统一成：

```python
class ExitPolicy:
    def build_exit_plan(self, signal_row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...
    def simulate_exit(self, price_path: pd.DataFrame, exit_plan: dict[str, Any]) -> dict[str, Any]: ...
```

这样后面线上加“卖出提示”时，可以直接复用同一层。

### 3.7 报告层

职责：

- 输出单次回测完整结果
- 输出参数对比
- 输出 walk-forward 样本外结果
- 输出 debug 候选与坏交易诊断

建议每次回测至少输出：

- `summary.json`
- `daily_results.csv`
- `trades.csv`
- `equity_curve.csv`
- `strategy_summary.csv`
- `regime_summary.csv`
- `monthly_summary.csv`
- `exit_reason_summary.csv`
- `top_trades.csv`
- `worst_trades.csv`
- `config_snapshot.json`

## 4. 建议的目录结构

建议后续逐步演进成：

```text
/Users/lvxue/work/量化
  /backtest
    /core
      data_provider.py
      universe.py
      runner.py
      execution.py
      exit_engine.py
      metrics.py
      reporter.py
    /plugins
      limitup_l1l2.py
      platform_breakout.py
      double_bottom.py
      real_breakout.py
      holder_increase.py
    /configs
      default_execution.json
      default_metrics.json
    /reports
      ...
  /scripts
    run_backtest.py
    run_backtest_suite.py
    run_param_search.py
    run_walkforward.py
```

注意：

- 不是要求马上重构到这个结构
- 更稳的做法是“先沿着现有脚本平滑迁移”

## 5. 面向新策略的接入标准

以后每条新策略必须至少提供这 6 个东西。

1. `strategy plugin`
- 负责输出当日候选和扩展字段

2. `config`
- 阈值、窗口、权重全部放配置文件

3. `rank score`
- 必须给出排序分，不允许只返回布尔命中

4. `debug fields`
- 至少输出 5 到 10 个关键解释字段

5. `exit reference`
- 给出关键支撑/失效位，方便统一卖出层使用

6. `research baseline`
- 新策略一上来必须先有一套 baseline 回测，不允许直接进测试环境

## 6. 两类策略的统一方案

你现在的策略其实已经分成两大类。

### 6.1 价格型策略

包括：

- `龙门双阶强势臻选`
- `天衡回踩转强臻选`
- `玄枢双底反转臻选`
- `真实资金突破臻选`

这类策略适合统一走：

- `market_daily_history`
- `stock_basic`
- `market_regime`
- `strategy plugin`
- `exit policy`

### 6.2 事件/混合型策略

包括：

- `星曜增持臻选`

这类策略难点不在图形，而在：

- `事件发布时间`
- `财务公告可见时点`
- `深度指标的历史截断`

所以建议单独支持：

- `snapshot replay`
- `as_of_date` 数据恢复
- `event manifest`

结论是：

- 同一个回测系统可以同时支持两类策略
- 但数据层必须允许“价格型直接日线回放”和“事件型历史快照回放”两种模式

## 7. 回测系统的运行模式

建议统一成 5 种模式。

### 7.1 baseline

用途：

- 看当前策略默认参数有没有 edge

### 7.2 ab_test

用途：

- 比较两组参数或两种 gate

### 7.3 search

用途：

- 小规模参数搜索

### 7.4 walkforward

用途：

- 看样本外稳定性

### 7.5 replay_debug

用途：

- 调单日错误
- 看某天为什么选这只，不是那只

## 8. 新策略回测标准流程

建议以后每条新策略都按下面流程走。

1. 先写策略模块
2. 跑 `3~5` 天 smoke test
3. 跑最近 `20` 个交易日 baseline
4. 跑最近 `60~100` 个交易日 baseline
5. 只在 baseline 有 edge 时，才进入参数优化
6. 参数优化后，必须做 walk-forward
7. walk-forward 通过，再考虑推进测试环境

不要反过来：

- 不要先做大规模参数搜索，再去看 baseline
- 不要 baseline 都是负的，就直接上测试环境

## 9. 统一评价指标

建议以后所有策略都统一输出这些核心指标：

- `signal_days`
- `filled_trades`
- `avg_exit_return_pct`
- `exit_win_rate_pct`
- `profit_factor`
- `max_drawdown_pct`
- `avg_exit_hold_days`
- `avg_exit_mfe_pct`
- `avg_exit_mae_pct`
- `positive_month_ratio_pct`
- `selection_score`

再按策略需要补充：

- `regime_summary`
- `buy_type_summary`
- `entry_gate_summary`

## 10. 推荐的实现顺序

这套系统不建议一次性大改，建议按 4 个阶段推进。

当前进度：

- 第 1 阶段已经开始落地
- 当前统一注册层文件是 [backtest_strategy_registry.py](/Users/lvxue/work/量化/scripts/backtest_strategy_registry.py)
- 现有价格型策略已经可以先通过这层做统一注册，再被主研究 runner / 优化器 / walk-forward 复用

### 阶段 1

目标：

- 把现有价格型研究框架整理成统一插件接口

优先做：

- 抽 `StrategyPlugin`
- 抽统一 `SignalRow` schema
- 抽统一 `Reporter`

### 阶段 2

目标：

- 把 `星曜增持臻选` 接进统一回测系统

优先做：

- `snapshot replay data provider`
- `as_of_date` 截断标准化

### 阶段 3

目标：

- 加统一成本模型、资金曲线和组合层回测

优先做：

- 单策略资金曲线
- 多策略组合回测
- 组合层市场状态切换

### 阶段 4

目标：

- 把研究层最稳定的策略推进测试环境

前提：

- baseline 有 edge
- 60~100 日验证不差
- walk-forward 不崩

## 11. 我建议你当前项目下一步怎么做

如果按投入产出比排序，我建议是：

1. 先把价格型策略整理成统一插件协议
- 这是最省心、收益最高的一步

2. 再把 `星曜增持臻选` 的快照回放接进统一框架
- 这步完成后，你就真的拥有一套“新策略可复用”的回测系统了

3. 最后再做组合层回测
- 也就是多策略一起跑、统一资金分配

## 12. 这套系统最终应该达到的状态

理想状态下，以后新加一条策略，只需要做这几件事：

1. 新建一个 `strategy plugin`
2. 提供一个 `config json`
3. 提供一个 `exit policy`
4. 注册到 `strategy registry`
5. 跑 baseline / search / walkforward

而不是：

- 重写一个新 runner
- 重写一个新导出格式
- 重写一套新优化脚本

如果做到这里，这套系统就已经是“可长期复用”的了。

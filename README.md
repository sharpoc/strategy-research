# 量化策略研究目录

这个目录主要存放 A 股策略研究脚本、Tushare 本地筛选 runner、配置和研究文档。运行时会产生 notebook 产物、缓存和导出数据，但这些都不纳入版本控制。

## 一眼看懂

- 这是 **研究与执行仓库**，不是线上展示仓库。
- 它主要服务未来的 `Mac mini`：
  - 跑晚间选股
  - 跑补数
  - 跑回测和参数优化
  - 跑盘中跟踪刷新
- 线上展示服务仓库在：
  - [/Users/lvxue/work/策略实验室](/Users/lvxue/work/策略实验室)
- 这个仓库只放 **脚本、配置、研究文档**，不放任何行情数据、导出结果、回测结果和缓存。

## 仓库不包含什么

以下内容一律不进仓库：

- 股票行情数据
- Tushare 原始接口导出
- 选股结果导出
- 回测结果
- 历史快照
- notebook 运行产物
- 页面缓存、数据库导出或恢复中间文件
- 本地调试日志和临时 scratch 文件

当前目录里，至少这些路径都属于运行数据：

- `output/jupyter-notebook/tushare_exports/`
- `output/jupyter-notebook/tushare_screen_exports/`
- `output/research_backtests/`
- `output/cache/`
- `.ipynb_checkpoints`
- `tk.csv`

## 当前定位

这个仓库当前承担两类工作：

- 策略研究
  - 新策略定义
  - 参数优化
  - walk-forward
  - 研究报告
- 执行准备
  - 纯 Python 单天 runner
  - 断点续跑
  - 失败只补缺口
  - 未来给 `Mac mini` 的定时执行链路

## 与线上服务的分工

- `Mac mini`
  - 运行这个仓库里的重任务
  - 产出每天最终结果与跟踪快照
  - 未来通过接口把结果推到线上
- `ECS`
  - 只负责页面展示、轻量查询和接收结果

也就是说：

- 线上不再直接承担重型研究和补数任务
- 这个仓库会逐步成为“策略大脑”

如果后续切换到别的 AI，先读这份 README，再看具体脚本。

如果要看“未来统一回测系统怎么搭”，再读：

- [BACKTEST_SYSTEM_DESIGN.md](/Users/lvxue/work/量化/BACKTEST_SYSTEM_DESIGN.md)

## 维护约定

- 以后每次改这个目录里的策略逻辑、runner、导出结构或本地运行方式，都要同步更新这份 README。
- 不在 README 里写 token、密码、私钥。
- 这个目录偏研究与本地验证，不等于已经接入网页或线上服务。

## 目录定位

这个目录负责：

- 策略形态识别与评分逻辑
- Tushare 本地日线筛选 runner
- 独立研究回测框架
- notebook 与导出结果
- 本地缓存

网页展示与线上部署主项目在：

- `/Users/lvxue/work/策略实验室`

## 当前主要策略脚本

### 1. 星曜增持臻选

- runner: [run_tushare_holder_strategy.py](/Users/lvxue/work/量化/scripts/run_tushare_holder_strategy.py)
- 纯模块: [holder_strategy_core.py](/Users/lvxue/work/量化/scripts/holder_strategy_core.py)
- 纯模块单日 runner: [run_tushare_holder_strategy_core.py](/Users/lvxue/work/量化/scripts/run_tushare_holder_strategy_core.py)
- 按交易日补线上结果: [run_holder_strategy_daily_range.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_daily_range.py)
- 历史快照生成器: [run_holder_strategy_snapshot_range.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_snapshot_range.py)
- snapshot replay 回测: [run_holder_strategy_replay_backtest.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_replay_backtest.py)
- 离线优化器: [optimize_holder_strategy_params.py](/Users/lvxue/work/量化/scripts/optimize_holder_strategy_params.py)
- 卖出优化器: [optimize_holder_exit_rules.py](/Users/lvxue/work/量化/scripts/optimize_holder_exit_rules.py)
- snapshot walk-forward: [run_holder_strategy_walkforward.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_walkforward.py)
- 当前研究推荐参数: [holder_strategy_research_best.json](/Users/lvxue/work/量化/configs/holder_strategy_research_best.json)
- 类型：综合评分策略
- 说明：当前版本已经不是纯“增持事件硬门槛”策略，而是偏稳健、防守优先的全市场综合筛选，增持事件作为辅助加分
- 当前状态：
  - 已经把 notebook 里的主流程抽成纯 Python 模块
  - `stage1 / stage2 / best_pick` 可以用旧导出结果做离线对拍
  - 自定义 `TUSHARE_HTTP_URL` 现在至少能拉到 `trade_cal / stock_basic / stk_holdertrade`
  - 但整段逐日历史重放的稳定性还不够，接口偶发超时，研究态先继续以导出回放为主
  - 已经补上 `snapshot replay / 卖出优化 / walk-forward` 本地研究链路
  - 当前短样本结论是：入口参数有提升，卖出规则暂时不是主矛盾
  - 还没有切到 live/test runner，避免影响本地测试环境
  - 研究层当前优先走 `export replay`，等历史快照生成更稳定后再扩到更长窗口 walk-forward
  - `run_holder_strategy_daily_range.py` 现在默认走“单天串行、断点续跑、失败只补缺口”的执行层
  - `holder_strategy_core.py` 现在会把单天导出目录当作 checkpoint：`deep_metrics_stage1.csv / stage2_cyq_metrics.csv / screen_progress.json`
  - 如果某一天因为 `daily / fina_indicator / adj_factor / stk_factor_pro` 抖动导致只跑完一部分，下一次会只补没完成的股票，不会整天重来
  - 旧的完整导出仍兼容，缺少 `stage1_complete / stage2_complete` 字段时会默认视作已完成，不会把老结果误判成残缺

### 2. 龙门双阶强势臻选

- runner: [run_tushare_limitup_l1l2_strategy.py](/Users/lvxue/work/量化/scripts/run_tushare_limitup_l1l2_strategy.py)
- 形态模块: [limitup_l1l2_strategy.py](/Users/lvxue/work/量化/scripts/limitup_l1l2_strategy.py)
- 类型：涨停后 L1/L2 结构策略
- 说明：每天只保留当日最强 1 只

### 3. 天衡回踩转强臻选

- runner: [run_tushare_platform_breakout_strategy.py](/Users/lvxue/work/量化/scripts/run_tushare_platform_breakout_strategy.py)
- 形态模块: [platform_breakout_retest_strategy.py](/Users/lvxue/work/量化/scripts/platform_breakout_retest_strategy.py)
- 类型：平台突破后回踩转强策略
- 说明：每天只保留当日最强 1 只

### 4. 玄枢双底反转臻选

- runner: [run_tushare_double_bottom_strategy.py](/Users/lvxue/work/量化/scripts/run_tushare_double_bottom_strategy.py)
- 形态模块: [double_bottom_strategy.py](/Users/lvxue/work/量化/scripts/double_bottom_strategy.py)
- 类型：底部双底右侧确认策略
- 当前状态：仅本地测试，尚未接入网页与线上调度

核心思路：

- 前面必须先有一段明显下跌
- 再识别 `L1 -> H -> L2`
- `L2` 不能明显弱于 `L1`
- `H -> L2` 回踩阶段要求缩量
- 当前必须属于 `A/B/C` 三类右侧买点之一
- 最终只保留当日评分最高的一只

买点定义：

- `A`：当日放量突破颈线
- `B`：已突破后缩量回踩确认
- `C`：L2 后右侧启动但尚未正式突破颈线

当前默认优先级：

- `B > A > C`

### 5. 真实资金突破臻选

- 形态模块: [real_fund_breakout_strategy.py](/Users/lvxue/work/量化/scripts/real_fund_breakout_strategy.py)
- 研究基线配置: [real_breakout_research_baseline.json](/Users/lvxue/work/量化/configs/real_breakout_research_baseline.json)
- 当前推荐研究配置: [real_breakout_research_downtrend.json](/Users/lvxue/work/量化/configs/real_breakout_research_downtrend.json)
- 类型：`缩量平台 + 放量突破 + 健康换手` 的成功率优先策略
- 当前状态：仅本地研究，尚未接入网页、测试环境和线上调度
- 当前研究结论：
  - 全市场直接做“突破当天 + 震荡乱打”效果一般
  - 当前最有效的方向，是只做 `retest_hold / follow_through`
  - 且研究态只在 `下跌趋势` 里出手，明显优于全市场硬做

### 6. 核心高管连增臻选

- runner: [run_tushare_core_management_accumulation_strategy.py](/Users/lvxue/work/量化/scripts/run_tushare_core_management_accumulation_strategy.py)
- 策略模块: [core_management_accumulation_strategy.py](/Users/lvxue/work/量化/scripts/core_management_accumulation_strategy.py)
- 6个月 `final` 轻确认对拍: [run_core_management_final_review.py](/Users/lvxue/work/量化/scripts/run_core_management_final_review.py)
- 类型：事件驱动策略
- 当前状态：仅本地研究，尚未接入网页、测试环境和线上调度
- 核心逻辑：
  - 核心管理层最近一轮连续增持
  - 使用最近一轮增持的加权均价作为成本区
  - 当前价格仍处于合理增持成本区附近
  - 增持后 `5~10` 日量价结构没有走坏，并开始修复
- 当前默认增强项：
  - `holder_type + holder_name` 身份权重映射
  - 支持“同一交易日内多位核心管理层密集买入”作为连续性信号，不再只认多交易日
  - 当前价相对增持成本做核心评分
  - 增持后 `5~10` 日结构单独打分
  - `融资净买入 / 主力净流 / 换手健康度` 只做辅助加分，不做硬门槛
  - 当前优化方向已经从“继续压严 final”切到“`stage1` 作为候选池，`final` 做轻确认 + 去重”
  - 当前已确认：最近 6 个月 `stage1` 的 5/10 日表现强于旧版 `final`，所以后续优先优化最终排序，而不是放宽事件入口
  - `run_core_management_final_review.py` 会直接复用现有 6 个月统计里的候选交易日，按日期重建新版 `final`，并输出 `optimized_final_signals.csv / review_summary.json / review_report.md`

## 独立研究框架

这部分是为了做“市场状态分层回测”和后续“参数优化”新加的，和现有网页/测试环境隔离。

当前文件：

- [backtest_data_catalog.py](/Users/lvxue/work/量化/scripts/backtest_data_catalog.py)
- [backtest_strategy_registry.py](/Users/lvxue/work/量化/scripts/backtest_strategy_registry.py)
- [market_regime.py](/Users/lvxue/work/量化/scripts/market_regime.py)
- [research_backtest_utils.py](/Users/lvxue/work/量化/scripts/research_backtest_utils.py)
- [research_config_presets.py](/Users/lvxue/work/量化/scripts/research_config_presets.py)
- [research_universe_filters.py](/Users/lvxue/work/量化/scripts/research_universe_filters.py)
- [holder_strategy_core.py](/Users/lvxue/work/量化/scripts/holder_strategy_core.py)
- [run_tushare_holder_strategy_core.py](/Users/lvxue/work/量化/scripts/run_tushare_holder_strategy_core.py)
- [run_holder_strategy_snapshot_range.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_snapshot_range.py)
- [holder_replay_utils.py](/Users/lvxue/work/量化/scripts/holder_replay_utils.py)
- [run_holder_strategy_replay_backtest.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_replay_backtest.py)
- [optimize_holder_strategy_params.py](/Users/lvxue/work/量化/scripts/optimize_holder_strategy_params.py)
- [optimize_holder_exit_rules.py](/Users/lvxue/work/量化/scripts/optimize_holder_exit_rules.py)
- [run_holder_strategy_walkforward.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_walkforward.py)
- [audit_backtest_data_inventory.py](/Users/lvxue/work/量化/scripts/audit_backtest_data_inventory.py)
- [prepare_backtest_market_cache.py](/Users/lvxue/work/量化/scripts/prepare_backtest_market_cache.py)
- [prepare_backtest_data_for_strategy.py](/Users/lvxue/work/量化/scripts/prepare_backtest_data_for_strategy.py)
- [run_price_strategy_regime_backtest.py](/Users/lvxue/work/量化/scripts/run_price_strategy_regime_backtest.py)
- [run_price_strategy_research_suite.py](/Users/lvxue/work/量化/scripts/run_price_strategy_research_suite.py)
- [strategy_exit_rules.py](/Users/lvxue/work/量化/scripts/strategy_exit_rules.py)
- [optimize_price_strategy_params.py](/Users/lvxue/work/量化/scripts/optimize_price_strategy_params.py)
- [optimize_exit_rules.py](/Users/lvxue/work/量化/scripts/optimize_exit_rules.py)
- [run_price_strategy_walkforward.py](/Users/lvxue/work/量化/scripts/run_price_strategy_walkforward.py)

作用分别是：

- `backtest_data_catalog.py`
  - 回测系统的数据目录清单和策略数据需求定义
  - 统一描述“某条策略至少依赖哪些缓存 / 快照 / 导出目录”
  - 当前已经把 4 条价格型策略和 `星曜增持臻选` 的研究数据需求收敛进同一层
- `backtest_strategy_registry.py`
  - 当前价格型策略的统一注册层
  - 负责维护 `strategy_id -> strategy_name -> history_bars -> build_candidates`
  - 现在也带有 `recommended_history_years / required_dataset_ids / optional_dataset_ids`
  - 也承接策略专属的 research ranking / entry gate
  - 以后新价格型策略优先在这里注册，不要再直接把注册表散落到各个 runner 里
- `market_regime.py`
  - 给每个交易日打上 `上涨趋势 / 下跌趋势 / 震荡趋势` 标签
- `research_backtest_utils.py`
  - 统一处理研究态缓存、交易日、历史行情、前瞻收益表
  - 现在也支持直接从本地 `daily_all` 缓存回放行情，不一定非要实时接口
- `research_config_presets.py`
  - 统一维护研究态命名 preset
  - 当前 `run_price_strategy_regime_backtest.py / run_price_strategy_research_suite.py / optimize_price_strategy_params.py / optimize_exit_rules.py / run_price_strategy_walkforward.py`
    都支持 `--strategy-config-preset`
  - 价格型研究入口也都支持 `--exit-config-preset`
  - preset 会先加载，再叠加对应的 `--strategy-config-file / --exit-config-file`
  - 这样以后跑研究总入口、参数优化、卖出优化、walk-forward 时，不用每次手工拼 JSON 文件
- `holder_strategy_core.py`
  - 把 `星曜增持臻选` 从 notebook 逻辑抽成纯 Python 模块
  - 当前已经拆出 `candidate_base -> stage1 -> stage2 -> best_pick` 三层纯函数
  - `fina_indicator / forecast / cyq_perf` 这些深度指标现在支持按 `as_of_date` 做历史截断，尽量减少前视偏差
- `run_tushare_holder_strategy_core.py`
  - 用纯模块直接跑单日增持策略，不需要再执行 notebook
  - 现在支持通过 `config-file / config-json` 覆盖参数，方便拿研究参数做本地验证
  - 现在也支持 `--resume-existing` 和 `--require-complete`
  - 适合以后线上/测试环境切到更轻的纯 Python 执行入口
- `run_holder_strategy_daily_range.py`
  - 按交易日区间补“线上口径”的单天结果，不是研究态整段回放
  - 默认就是 `resume-existing + require-complete`
  - 适合补 `20260309~20260320` 这类线上缺口日期
  - 当前已经验证：`20260309~20260320` 的开市日导出本地齐全，可作为线上缺口恢复的优先数据源
- `../策略实验室/scripts/backfill_holder_missing_days.py`
  - 给网页 PostgreSQL 用的安全补数脚本
  - 会先查数据库里哪些交易日已经完整存在，再只补缺的日期
  - 优先吃本地已有的 `holder_increase_screen_<date>` 导出
  - 只在本地导出缺失时，才建议回退到单天 runner
  - 本地已验证：对 `20260309~20260320` 做 dry-run 能正确识别缺口；真实写入验证时只补了缺的 `20260312`
  - 当前默认是“轻补数”：只写缺失选股结果，不默认跑市场历史回填，也不默认重建首页缓存
- `run_holder_strategy_snapshot_range.py`
  - 给 `星曜增持臻选` 批量生成历史 `holder_increase_screen_<date>` 快照
  - 支持按日期区间、断点恢复、跳过已完成导出
  - 默认写入隔离目录 `output/research_backtests/holder_snapshots`
  - 这是后续把增持策略接入统一 walk-forward 的前置脚本
- `optimize_holder_strategy_params.py`
  - 当前是 `export replay` 模式
  - 直接读取 `holder_increase_screen_<date>` 导出和本地行情缓存做离线参数对比
  - 现在支持 `snapshot-root`，可以直接读取研究态快照，不污染测试环境导出
  - 适合在接口异常时先验证“哪些阈值方向值得保留”
- `holder_replay_utils.py`
  - `星曜增持臻选` 的 snapshot replay 共用工具层
  - 负责扫描 `holder_increase_screen_<date>` 快照目录、重建 `best_pick_candidate`、拼接前瞻收益和卖出路径
  - 当前也顺手收敛了 replay 里重复字段，例如 `chip_score_x / chip_score_y`
- `run_holder_strategy_replay_backtest.py`
  - 用历史导出快照做 `T+1 开盘进场` 的本地回放
  - 输出 `daily_results / strategy_summary / regime_summary / monthly_summary / exit_reason_summary / summary.json`
  - 这是当前最稳的 `星曜增持臻选` 回测入口
- `optimize_holder_exit_rules.py`
  - 对 `星曜增持臻选` 做 snapshot replay 卖出优化
  - 逻辑和价格型 `optimize_exit_rules.py` 一致，但入口换成了 holder 快照回放
  - 当前主要用来判断：这条策略的瓶颈到底在入口还是在卖出
- `run_holder_strategy_walkforward.py`
  - 对 `星曜增持臻选` 做 snapshot replay 版 walk-forward
  - 每个 fold 先在训练快照里随机搜索入口参数，再去验证快照里看样本外表现
  - 当前更适合做“小样本 sanity check”，不是最终长样本结论
- `audit_backtest_data_inventory.py`
  - 本地数据就绪度盘点工具
  - 会统一检查交易日历、全市场日线、增持快照、事件缓存、财务缓存等是否已经具备
  - 适合在准备新策略回测前先判断：当前本地数据是否已经够做 `2~3` 年研究
- `prepare_backtest_market_cache.py`
  - 本地全市场日线补齐脚本
  - 用来把 `trade_cal / stock_basic_all / daily_all` 按指定区间补到本地缓存
  - 适合在开始大区间回测前，先把价格型策略依赖的核心日线补齐到 `2.5~3` 年
- `prepare_backtest_data_for_strategy.py`
  - 按策略自动准备研究数据的入口
  - 会读取策略推荐历史窗口，自动决定需要往前补多少日线
  - 适合以后加新策略后，先做“按策略补数”，再跑回测
- `research_universe_filters.py`
  - 给研究态候选票增加统一股票池过滤
  - 当前支持 `ST/退市/北交所/科创板`、上市交易日、价格、近20日成交额过滤
  - 只在研究入口里生效，不会偷偷改 live / 测试环境逻辑
- `run_price_strategy_regime_backtest.py`
  - 对三条价格型策略做逐日回测
  - 收益口径统一为 `T+1 开盘买入`
  - 输出按市场状态拆分的统计结果
  - 如果 walk-forward 为了卖出评估把结束日期往后 padding，会自动把研究数据集截到“今天可用的真实日期”为止，避免请求未来交易日
- `run_price_strategy_research_suite.py`
  - 研究总入口
  - 一次性导出多策略基线比较、分市场比较、最佳市场状态建议和代表性交易
- `strategy_exit_rules.py`
  - 统一处理研究态卖出逻辑
  - 当前支持三条价格型策略的结构止损、保本止损、趋势走弱退出、时间退出
- `optimize_price_strategy_params.py`
  - 针对单条价格型策略做参数随机搜索
  - 复用研究数据集，不重复拉历史数据
  - 默认按 `卖出后收益` 排序，不是只看固定持有天数
- `optimize_exit_rules.py`
  - 针对单条价格型策略做卖出规则随机搜索
  - 会先生成一次底板信号，再反复重放不同 exit 配置
  - 适合用来判断“当前瓶颈在入口还是卖出”
- `run_price_strategy_walkforward.py`
  - 对单条价格型策略做 walk-forward 验证
  - 每个 fold 只用训练窗口挑参数，再去验证窗口做样本外评估
  - 适合拿来判断“这条策略是不是只是调到了某一小段行情”

重要约束：

- 这套研究框架不接网页
- 不改现有服务配置
- 不改现有定时任务
- 不改线上或本地测试环境的默认 runner 行为
- 所有研究输出只写到新的 `output/research_backtests` 目录

### 当前已接入研究回测的策略

- `龙门双阶强势臻选`
- `天衡回踩转强臻选`
- `玄枢双底反转臻选`
- `真实资金突破臻选`

`星曜增持臻选` 当前处于“中间态”：

- 已经有纯模块和离线 replay 优化器
- 但还没正式接到和三条价格型策略同一套 `逐日 API 历史回放`
- 原因不是模块没抽出来，而是截至 `2026-03-15`：
  - 官方 token 校验失败
  - 自定义 `http_url` 关键端点返回空结果
  - 所以完整历史重放暂时无法可靠落地

当前最稳的做法是：

- 先用旧导出 + 本地缓存做离线参数方向判断
- 先把 snapshot replay 这条链路做扎实，补回测、卖出和 walk-forward
- 等接口恢复后，再把 `星曜增持臻选` 接进更长窗口的统一历史回测框架

## 常用脚本路径

- [scripts](/Users/lvxue/work/量化/scripts)
- [output](/Users/lvxue/work/量化/output)

辅助查看当前已注册价格型策略：

- [list_backtest_strategies.py](/Users/lvxue/work/量化/scripts/list_backtest_strategies.py)
- [audit_backtest_data_inventory.py](/Users/lvxue/work/量化/scripts/audit_backtest_data_inventory.py)
- [prepare_backtest_market_cache.py](/Users/lvxue/work/量化/scripts/prepare_backtest_market_cache.py)
- [prepare_backtest_data_for_strategy.py](/Users/lvxue/work/量化/scripts/prepare_backtest_data_for_strategy.py)

导出目录示例：

- `output/jupyter-notebook/tushare_screen_exports`
- `output/jupyter-notebook/tushare_limitup_l1l2_exports`
- `output/jupyter-notebook/tushare_platform_breakout_exports`
- `output/jupyter-notebook/tushare_double_bottom_exports`
- `output/research_backtests`

缓存目录示例：

- `output/cache/platform_breakout_api`
- `output/cache/limitup_l1l2_api`
- `output/cache/double_bottom_api`

## 本地运行

至少需要：

```bash
export TUSHARE_TOKEN="你的token"
```

如果走自定义 Tushare 代理接口，还需要：

```bash
export TUSHARE_HTTP_URL="http://lianghua.nanyangqiankun.top"
```

双底策略本地运行示例：

```bash
python3 /Users/lvxue/work/量化/scripts/run_tushare_double_bottom_strategy.py --end-date 20260312 --show-top 10
```

研究框架本地运行示例：

```bash
python3 /Users/lvxue/work/量化/scripts/run_price_strategy_regime_backtest.py --start-date 20260310 --end-date 20260312 --max-trade-days 3
```

本地盘点回测数据是否已经准备好的示例：

```bash
python3 /Users/lvxue/work/量化/scripts/audit_backtest_data_inventory.py
```

这个脚本会：

- 输出当前本地有哪些关键数据集
- 估算全市场日线缓存覆盖了多少年
- 检查全市场日线在这个跨度内是否连续，缺了多少交易日
- 判断每条策略是否已经具备 `replay_ready`
- 对 `星曜增持臻选` 额外区分 `export replay` 和 `api replay` 的准备情况

把全市场日线缓存补到最近 `2.5` 年的示例：

```bash
python3 /Users/lvxue/work/量化/scripts/prepare_backtest_market_cache.py --years-back 2.5 --only-missing --passes 3
```

这个脚本会：

- 先保证 `stock_basic_all` 和 `trade_cal` 已经缓存
- 再按区间补齐缺失的 `daily_all_<trade_date>` 文件
- 对接口偶发漏掉的日期，可以自动再跑多轮 missing pass
- 最后输出更新后的全市场日线覆盖范围

按策略自动准备研究数据的示例：

```bash
python3 /Users/lvxue/work/量化/scripts/prepare_backtest_data_for_strategy.py --strategy-ids limitup_l1l2,double_bottom --backtest-start-date 20240102 --only-missing --passes 3
```

这个脚本会：

- 自动读取这些策略要求的推荐历史窗口
- 如果给了 `backtest-start-date`，会额外把 `history_bars` 对应的 warmup 历史也一起准备
- 计算出本次需要补到多早
- 调用本地市场缓存补齐逻辑
- 最后把这几条策略当前的 `replay_ready / history_window_ready` 一起打印出来

研究股票池过滤配置示例：

- [price_strategy_research_filters.json](/Users/lvxue/work/量化/configs/price_strategy_research_filters.json)
- [price_strategy_research_filters_limitup_nogate.json](/Users/lvxue/work/量化/configs/price_strategy_research_filters_limitup_nogate.json)
- [price_strategy_research_filters_limitup_selective.json](/Users/lvxue/work/量化/configs/price_strategy_research_filters_limitup_selective.json)
- [price_strategy_research_all.json](/Users/lvxue/work/量化/configs/price_strategy_research_all.json)

研究 preset：

- `research`
  - 三条价格型策略统一研究过滤 + `龙门双阶` research gate
- `research_limitup_nogate`
  - 三条价格型策略统一研究过滤，但 `龙门双阶` 不启用 research gate
- `research_limitup_selective`
  - 三条价格型策略统一研究过滤，`龙门双阶` 使用更严格的 selective gate
- `research_all_price`
  - 四条价格型策略一体化研究配置
  - 包含 `真实资金突破臻选` 的保守版研究参数
- `real_breakout_baseline`
  - 真实资金突破基线研究参数
- `real_breakout_tuned`
  - 真实资金突破收紧版研究参数
- `real_breakout_downtrend`
  - 真实资金突破只在 `下跌趋势` 出手的研究参数
- `real_breakout_downtrend_selective`
  - 真实资金突破只在 `下跌趋势` 出手，并额外收紧 `pre_runup / platform_days / current_buffer`

研究 exit preset：

- `limitup_l1l2_research_best`
  - 龙门双阶 `2026-03-16` 本地 exit 优化最佳研究配置
- `real_breakout_target_first`
  - 真实资金突破盘中冲突时优先按目标价成交
- `real_breakout_research_best`
  - 真实资金突破 `2026-03-16` 本地 exit 优化最佳研究配置
- `price_selective_best`
  - 龙门双阶 + 真实资金突破的 selective 最佳 exit 组合

这份配置当前会：

- 排除 `ST / *ST / 退市整理`
- 排除北交所
- 平台突破与双底默认排除科创板
- 要求上市交易日不少于 `120`
- 要求最新价不低于 `3`
- 要求近 `20` 个交易日平均成交额不低于 `1 亿`

补充说明：

- `min_listed_trade_days` 在研究态里当前优先用 `list_date -> as_of_date` 的日历天数做代理，按大约 `1.45x` 换算为交易日门槛
- 这样可以避免“历史窗口不够长，导致老股票被误判成新股”的问题
- 如果后面要做更精确的上市满 N 个交易日判断，可以再接一层更细的 trade_cal 计数

研究总入口示例：

```bash
python3 /Users/lvxue/work/量化/scripts/run_price_strategy_research_suite.py \
  --start-date 20260102 \
  --end-date 20260312 \
  --strategies limitup_l1l2,platform_breakout \
  --strategy-config-preset research
```

四条价格型策略的一体化研究示例：

```bash
python3 /Users/lvxue/work/量化/scripts/run_price_strategy_research_suite.py \
  --start-date 20260102 \
  --end-date 20260312 \
  --strategies limitup_l1l2,platform_breakout,double_bottom,real_breakout \
  --strategy-config-preset research_all_price \
  --max-trade-days 20
```

真实资金突破策略研究示例：

```bash
python3 /Users/lvxue/work/量化/scripts/run_price_strategy_research_suite.py \
  --start-date 20250701 \
  --end-date 20260312 \
  --strategies real_breakout \
  --strategy-config-preset real_breakout_downtrend \
  --max-trade-days 100
```

输出文件通常包括：

- `market_regime_snapshot.csv`
- `daily_results.csv`
- `strategy_summary.csv`
- `regime_summary.csv`
- `monthly_summary.csv`
- `exit_reason_summary.csv`
- `summary.json`

研究总入口通常还会补这些总表：

- `strategy_compare.csv`
- `strategy_regime_compare.csv`
- `best_regime_recommendations.csv`
- `top_trades.csv`
- `worst_trades.csv`

`daily_results.csv` 现在除了固定持有窗口收益，还会带上：

- `exit_trade_date`
- `exit_price`
- `exit_reason`
- `exit_rule`
- `exit_hold_days`
- `exit_return_pct`
- `exit_target_price`
- `exit_structure_stop`
- `exit_active_stop`
- `exit_mfe_pct`
- `exit_mae_pct`

如果启用了研究股票池过滤，还会额外带：

- `signal_count_before_filter`
- `research_filter_enabled`
- `research_filter_drop_count`
- `research_filter_drop_reasons`

如果启用了研究态后置出手开关，还会额外带：

- `research_entry_gate_enabled`
- `research_entry_gate_passed`
- `research_entry_gate_reason`
- `research_entry_gate_blocked_ts_code`
- `research_entry_gate_blocked_name`

最新一轮 preset 验证产物：

- 主回测 smoke：
  - [price_regime_backtest_20260310_20260312_121846](/Users/lvxue/work/量化/output/research_backtests/price_regime_backtest_20260310_20260312_121846)
- 研究总入口 smoke：
  - [research_suite_20260301_20260312_121847](/Users/lvxue/work/量化/output/research_backtests/research_suite_20260301_20260312_121847)
- 参数优化 smoke：
  - [optimize_limitup_l1l2_20260301_20260312_121919](/Users/lvxue/work/量化/output/research_backtests/optimize_limitup_l1l2_20260301_20260312_121919)
- walk-forward smoke：
  - [walkforward_limitup_l1l2_20260301_20260312_122027](/Users/lvxue/work/量化/output/research_backtests/walkforward_limitup_l1l2_20260301_20260312_122027)
- 四条价格型策略一体化 preset 验证：
  - [research_suite_20260102_20260312_122409](/Users/lvxue/work/量化/output/research_backtests/research_suite_20260102_20260312_122409)

### 当前卖出逻辑

研究框架里的卖出逻辑已经独立出来，后面线上如果要加“卖出提示/离场建议”，可以优先复用这套思路。

当前默认原则：

- `龙门双阶强势臻选`
  - 以 `L2` 下方为结构止损
  - 盈利达到一定幅度后上移到保本位
  - 再上涨一段后启用回撤止盈
  - 到期未触发则时间退出

- `天衡回踩转强臻选`
  - 以 `平台上沿 / 回踩低点` 为支撑止损
  - 目标位参考突破段的延伸幅度
  - 若短线转弱也会触发 close 级别退出

- `玄枢双底反转臻选`
  - `A/B` 类更重视颈线回踩失守
  - `C` 类更重视 `L2` 下方结构失效
  - 目标位参考双底量度升幅

- `星曜增持臻选`
  - 当前研究态默认按“趋势 + 持仓保护”退出
  - 先给一个固定硬止损，再参考 `MA20` 做保护止损
  - 浮盈达到一定幅度后切到保本，再启用回撤止盈
  - 若持有过程中走弱，也会触发 close 级别时间/趋势退出

注意：

- 日线回测无法知道同一根 K 线内“先打止损还是先打止盈”，当前默认采用偏保守的 `conservative` 处理
- 这套卖出规则目前是研究态默认值，不代表已经定稿
- 后续如果要上线，建议先做更长区间的 walk-forward 验证

### 参数优化器

当前已经支持对单条价格型策略做本地参数搜索，入口：

```bash
python3 /Users/lvxue/work/量化/scripts/optimize_price_strategy_params.py \
  --strategy-id platform_breakout \
  --start-date 20260301 \
  --end-date 20260312 \
  --trials 40
```

优化器特点：

- 一次只优化一条策略
- 先准备一次数据集，后面所有 trial 复用
- 默认不写 `signal_cache`，避免随机搜索把缓存目录堆满
- 支持通过 `strategy-config-file` 传入 `_research_filters`，让训练集和验证集共用同一套研究股票池

## 当前本地研究结论

下面这些是已经在本地真实跑过的阶段性结论，后面换 AI 继续做时，可以直接从这里接着推进。

### 1. `星曜增持臻选` 的纯模块与离线优化结论

已完成：

- 用 [holder_strategy_core.py](/Users/lvxue/work/量化/scripts/holder_strategy_core.py) 把 notebook 主流程抽成了纯模块
- 旧导出 `20260309 ~ 20260312` 已做离线对拍
- `apply_holder_stage1 / apply_holder_stage2` 能原样还原旧导出的 `best_pick_candidate`

当前能确认的事：

- 抽模块这一步已经稳定，后面不需要再靠执行 notebook 才能复现结果
- 目前最适合先保留的研究入口是 [optimize_holder_strategy_params.py](/Users/lvxue/work/量化/scripts/optimize_holder_strategy_params.py)
- 这条优化器当前不是全 API 历史回放，而是 `export replay`

截至 `2026-03-15` 的接口状态：

- 官方 token 校验报错
- 自定义 `TUSHARE_HTTP_URL` 目前能返回 `trade_cal / stock_basic / stk_holdertrade`
- 但整段历史重放时 `stk_holdertrade / stk_factor_pro` 仍可能偶发超时，所以这轮没有把“长区间全 API 快照生成”作为主结论来源
- 所以当前最可信的参数优化结论，仍然来自离线导出 replay

已验证的定向参数对比窗口：

- 信号日：`20260309 ~ 20260312`
- 收益口径：`T+1 开盘进场`，研究态卖出规则退出
- 当前离线行情只到 `20260313`，所以这轮属于短样本、首轮方向判断
- 研究产物目录： [holder_research_probe_20260309_20260312](/Users/lvxue/work/量化/output/research_backtests/holder_research_probe_20260309_20260312)

当前短样本里表现更好的方向：

- 默认参数：
  - 平均退出收益约 `+0.0788%`
  - 胜率 `50%`
- 更优的宽松版本：
  - `min_volume_ratio=1.1`
  - `max_price_position=0.50`
  - `max_industry_pb_pct=0.75`
  - `min_final_score=58~60`
  - `min_aggressive_score=50~52`
  - `top_n_stage1=8~10`
  - `top_n_final=3~5`
  - 平均退出收益约 `+0.2817%`
  - 主要改善来自 `20260310` 那天从 `002245.SZ 蔚蓝锂芯` 切到 `000998.SZ 隆平高科`

当前优先推荐保留的“最稳改法”是：

- [holder_strategy_research_best.json](/Users/lvxue/work/量化/configs/holder_strategy_research_best.json)
- 参数为：
  - `min_volume_ratio=1.1`
  - `max_price_position=0.50`
  - `max_industry_pb_pct=0.75`
  - `min_final_score=60`
  - `min_aggressive_score=52`
  - `top_n_stage1=10`
  - `top_n_final=5`
  - `top_n_aggressive=3`
- 选择它的原因：
  - 和默认参数相比改动最少
  - 但短样本结果已经和更激进的宽松版本打平
  - 更适合作为后续测试环境验证的候选版本

当前最重要的判断不是“已经找到最终最优参数”，而是：

- 这条策略在短样本里对 `price_position / pb 分位 / volume_ratio` 的阈值比较敏感
- 过于强调低位，会漏掉部分更强的右侧资金票
- 当前还不建议直接把这组参数推到测试环境
- 下一步应该先补更多历史快照，再用同一组参数做更长窗口 replay / walk-forward
- `stage2_cyq` 在这 4 个导出日里没有改变最终最优结果

下一步最值的事情：

- 等接口恢复后，优先把 `2025-01-01` 之后的历史日跑成真正的 holder 研究数据集
- 再决定是否把上面这组更宽松的阈值推进到测试环境

### 2. 最近 20 个交易日的未过滤基线

输出目录：

- [research_suite_20260102_20260312_165006](/Users/lvxue/work/量化/output/research_backtests/research_suite_20260102_20260312_165006)

结论：

- `天衡回踩转强臻选`
  - 样本里只有 `1` 笔信号
  - 平均退出收益约 `5.02%`
  - 当前最适合的市场状态是 `上涨趋势`
- `龙门双阶强势臻选`
  - 最近 `20` 个交易日有 `20` 个信号日
  - 但平均退出收益约 `-1.21%`
  - 在 `震荡趋势` 下明显好于 `上涨趋势 / 下跌趋势`
- 未过滤样本里已经确认混入了 `ST` 和北交所股票，所以这版只能当“原始基线”，不能直接拿来定稿

### 2A. `真实资金突破臻选` 的本地研究结论

当前新增文件：

- [real_fund_breakout_strategy.py](/Users/lvxue/work/量化/scripts/real_fund_breakout_strategy.py)
- [real_breakout_research_baseline.json](/Users/lvxue/work/量化/configs/real_breakout_research_baseline.json)
- [real_breakout_research_tuned.json](/Users/lvxue/work/量化/configs/real_breakout_research_tuned.json)
- [real_breakout_research_downtrend.json](/Users/lvxue/work/量化/configs/real_breakout_research_downtrend.json)

第一轮 60 日基线：

- 输出目录： [research_suite_20251001_20260312_230734](/Users/lvxue/work/量化/output/research_backtests/research_suite_20251001_20260312_230734)
- 口径：最近 `60` 个交易日、全市场允许当前策略出手
- 结果：
  - `filled_trades = 57`
  - `avg_exit_return_pct = -0.0552%`
  - `exit_win_rate_pct = 42.11%`
  - `max_drawdown_pct = 42.1197%`
- 结论：
  - 基线版不能进测试环境
  - 最大问题不是“完全没有 edge”，而是突破当天和震荡里的坏交易太多

第二轮 research-only 调整：

- 输出目录： [research_suite_20251001_20260312_232019](/Users/lvxue/work/量化/output/research_backtests/research_suite_20251001_20260312_232019)
- 研究调整内容：
  - 只保留 `retest_hold / follow_through`
  - 禁止 `breakout_today`
  - 限制 `breakout_volume_ratio <= 2.1`
  - 研究态 gate 只允许 `下跌趋势 + 震荡趋势`
- 结果：
  - `filled_trades = 18`
  - `avg_exit_return_pct = -0.2299%`
  - `exit_win_rate_pct = 50.00%`
  - `max_drawdown_pct = 20.8048%`
- 结论：
  - 回撤明显改善
  - 但震荡里的替补信号仍然拖后腿，仍不建议推进测试环境

当前最推荐的保守版：

- 输出目录： [research_suite_20250701_20260312_232817](/Users/lvxue/work/量化/output/research_backtests/research_suite_20250701_20260312_232817)
- 口径：
  - 最近 `100` 个交易日
  - 研究态只允许 `下跌趋势`
  - 只做 `retest_hold / follow_through`
  - 仍沿用统一研究卖出规则
- 结果：
  - `signal_days = 10`
  - `filled_trades = 8`
  - `avg_exit_return_pct = +0.3855%`
  - `exit_win_rate_pct = 75.00%`
  - `max_drawdown_pct = 9.6174%`
  - `positive_month_ratio_pct = 100%`
  - `avg_return_5d_pct = +3.7859%`
  - `win_rate_5d_pct = 75.00%`
- 当前判断：
  - 这条策略已经找到“可研究、可继续验证”的保守版本
  - 它本质上已经变成一个“弱市/回撤市里的右侧确认策略”
  - 当前还不建议直接推进测试环境
  - 更合适的下一步，是继续在本地把 `下跌趋势` 口径拉长验证，再决定是否接测试环境

### 3. 最近 20 个交易日的过滤后基线

输出目录：

- [research_suite_20260102_20260312_173941](/Users/lvxue/work/量化/output/research_backtests/research_suite_20260102_20260312_173941)

结论：

- `天衡回踩转强臻选`
  - 过滤前后结果几乎不变
  - 仍然只有 `1` 笔样本，收益约 `5.02%`
  - 说明这条策略最近至少没有靠 `ST / 北交所 / 新股` 撑结果
- `龙门双阶强势臻选`
  - 过滤后信号日仍然是 `20`
  - 但平均退出收益变成约 `-2.38%`
  - `上涨趋势` 下最差，`震荡趋势` 相对最好但仍是负收益
  - 说明问题不只是股票池噪音，当前参数和排序本身也需要继续调

### 4. 当前最有效的优化方向

- `龙门双阶强势臻选` 当前最有效的改进，不是继续盲调参数，而是研究态先学会“不出手”
- 当前已经在 [run_price_strategy_regime_backtest.py](/Users/lvxue/work/量化/scripts/run_price_strategy_regime_backtest.py) 里加了 research-only 的后置出手开关
- 这层 gate 只作用在研究入口，不会改 live / 测试环境的默认排序逻辑
- 当前保留的中等强度 gate 条件是：
  - 只允许 `震荡趋势`
  - `impulse_pct >= 6`
  - `pullback_pct <= 8`
  - `L2 高于 L1 >= 1%`
  - `L2 防守缓冲 >= 1%`
  - `trend_ok = true`
- 离线搜索还找到过更激进的 tighter gate，长样本收益更漂亮，但最近 `20` 个交易日会被压到只剩 `1` 个信号，过拟合味道太重，所以当前没有采用
- `天衡回踩转强臻选` 当前样本太少，不适合直接大规模调参，先继续累积样本或拉长历史窗口
- `玄枢双底反转臻选` 已有研究脚本，但还没有足够有效样本，建议先单独做更长区间的基线检查，再决定是否投入参数优化

### 5. `龙门双阶强势臻选` 的最新参数研究结论

小规模优化输出：

- [optimize_limitup_l1l2_20260102_20260312_183837](/Users/lvxue/work/量化/output/research_backtests/optimize_limitup_l1l2_20260102_20260312_183837)

结论：

- 在最近 `20` 个交易日、`震荡趋势` 子样本里，随机采样的 `4` 组参数都没有打败默认参数
- 也就是说，当前默认参数在这轮小样本里已经是局部最优
- 这不代表参数已经全局最优，只代表“继续在当前这几个参数上盲调，短期内收益不大”

walk-forward 输出：

- [walkforward_limitup_l1l2_20260205_20260312_191023](/Users/lvxue/work/量化/output/research_backtests/walkforward_limitup_l1l2_20260205_20260312_191023)

结论：

- `2` 个 fold 的样本外平均退出收益约 `-1.7614%`
- 样本外正收益 fold 比例 `0%`
- 随机参数打败默认参数的比例 `0%`

当前判断：

- `龙门双阶强势臻选` 的问题暂时不在参数层
- 下一步更应该优先检查：
  - 候选排序分数是否过于单一
  - `买点触发` 是否过宽，导致劣质右侧信号太多
  - `上涨趋势` 下是否应该直接降权或停用这条策略
- 默认按 `exit_return_pct` 相关指标打分
- 支持可选验证区间：

```bash
python3 /Users/lvxue/work/量化/scripts/optimize_price_strategy_params.py \
  --strategy-id limitup_l1l2 \
  --start-date 20251201 \
  --end-date 20260228 \
  --validation-start-date 20260301 \
  --validation-end-date 20260331 \
  --trials 80
```

当前优化器综合看这些维度：

- 平均卖出收益
- 卖出胜率
- 盈亏比
- 最大回撤
- 月度正收益占比
- 交易数是否过少

导出目录通常包括：

- `trial_results.csv`
- `top_results.csv`
- `summary.json`

### 6. `龙门双阶强势臻选` 的 research-only 出手开关结论

最近 `20` 个交易日 gate 版本输出：

- [research_suite_20260102_20260312_194734](/Users/lvxue/work/量化/output/research_backtests/research_suite_20260102_20260312_194734)

最近 `20` 个交易日结论：

- no-gate 基线是 `20` 个交易日 `20` 次出手，平均退出收益约 `-2.38%`
- research gate 之后收敛成 `3` 次出手
- 平均退出收益约 `+1.64%`
- 卖出胜率约 `66.67%`
- 被挡掉最多的原因是：
  - `市场状态不匹配:上涨趋势`
  - `回撤幅度>8`
  - `冲高幅度<6`

最近 `60` 个交易日 gate 版本输出：

- [research_suite_20251101_20260312_200500](/Users/lvxue/work/量化/output/research_backtests/research_suite_20251101_20260312_200500)

最近 `60` 个交易日 no-gate 对照输出：

- [research_suite_20251101_20260312_203127](/Users/lvxue/work/量化/output/research_backtests/research_suite_20251101_20260312_203127)

最近 `60` 个交易日 A/B 结论：

- gate 版本：
  - `9` 次出手
  - 平均退出收益约 `+0.70%`
  - 卖出胜率约 `55.56%`
  - 最大回撤约 `7.41%`
  - 月度正收益占比约 `66.67%`
- no-gate 版本：
  - `60` 个信号日，实际成交 `59` 笔
  - 平均退出收益约 `-0.87%`
  - 卖出胜率约 `37.29%`
  - 最大回撤约 `53.80%`
  - 月度正收益占比约 `25%`

这组对照当前说明：

- `龙门双阶强势臻选` 的核心问题是坏交易太多，不是“缺少更激进的参数”
- 研究态最有效的第一步，是先通过 gate 学会少做
- 当前这版 gate 已经明显改善了：
  - 平均退出收益
  - 卖出胜率
  - 最大回撤
  - 月度稳定性
- 但它不是 live 默认逻辑，只是当前最值得继续验证的研究方向

### 7. 2026-03-16 selective gate 最新结论

本轮精确对照记录在：

- [selective_gate_compare_20260316.md](/Users/lvxue/work/量化/output/research_backtests/selective_gate_compare_20260316.md)

`龙门双阶强势臻选`：

- 使用 preset `research` 的 `60` 个交易日基线：
  - `signal_days=9`
  - `filled_trades=9`
  - `avg_exit_return_pct=+0.7030%`
  - `exit_win_rate_pct=55.56%`
- 使用 `research_limitup_selective` 后：
  - `signal_days=5`
  - `filled_trades=5`
  - `avg_exit_return_pct=+2.4405%`
  - `exit_win_rate_pct=80.00%`
- 这版 selective 主要多挡掉了：
  - `L2高于L1<1.2`
  - `L2高于L1>7`
  - `L1到L2>24`
  - `回撤幅度>7`
  - `涨停到L1>40`

`真实资金突破臻选`：

- 使用 `real_breakout_downtrend` 的 `100` 个交易日基线：
  - `signal_days=10`
  - `filled_trades=8`
  - `avg_exit_return_pct=+0.3855%`
  - `exit_win_rate_pct=75.00%`
- 使用 `real_breakout_downtrend_selective` 后：
  - `signal_days=7`
  - `filled_trades=5`
  - `avg_exit_return_pct=+2.5359%`
  - `exit_win_rate_pct=80.00%`
- 这版 selective 不只是少做，还会替换掉原先的顶候选，例如：
  - `2025-12-04` 从 `通化金马` 切到 `赛轮轮胎`
  - `2025-12-09` 从 `英维克` 切到 `汇绿生态`

当前建议：

- `龙门双阶强势臻选` 后续优先继续围绕 `research_limitup_selective` 做更长样本验证
- `真实资金突破臻选` 后续优先继续围绕 `real_breakout_downtrend_selective` 做更长样本验证
- 两者目前都还是研究态推荐配置，尚未推动到测试环境或线上默认逻辑

### 8. 2026-03-16 selective exit 最新结论

本轮卖出优化记录在：

- [selective_exit_compare_20260316.md](/Users/lvxue/work/量化/output/research_backtests/selective_exit_compare_20260316.md)

`龙门双阶强势臻选`：

- 基线使用 `research_limitup_selective`：
  - `filled_trades=5`
  - `avg_exit_return_pct=+2.4405%`
  - `exit_win_rate_pct=80.00%`
  - `avg_exit_hold_days=6.6`
- 卖出优化最佳：
  - `filled_trades=5`
  - `avg_exit_return_pct=+7.0232%`
  - `exit_win_rate_pct=80.00%`
  - `avg_exit_hold_days=8.8`
- 当前推荐 exit 配置：
  - [exit_limitup_l1l2_research_best.json](/Users/lvxue/work/量化/configs/exit_limitup_l1l2_research_best.json)

`真实资金突破臻选`：

- 基线使用 `real_breakout_downtrend_selective`：
  - `filled_trades=5`
  - `avg_exit_return_pct=+2.5359%`
  - `exit_win_rate_pct=80.00%`
  - `avg_exit_hold_days=6.2`
- 卖出优化最佳：
  - `filled_trades=5`
  - `avg_exit_return_pct=+4.6571%`
  - `exit_win_rate_pct=80.00%`
  - `avg_exit_hold_days=5.8`
- 当前推荐 exit 配置：
  - [exit_real_breakout_research_best.json](/Users/lvxue/work/量化/configs/exit_real_breakout_research_best.json)

两条 selective 的组合 exit 配置：

- [exit_price_selective_best.json](/Users/lvxue/work/量化/configs/exit_price_selective_best.json)
- 对应 preset：`price_selective_best`

这轮结论说明：

- `龙门双阶强势臻选` 现在不是“只有入口 selective 有用”，卖出规则本身也还有明显提升空间
- `真实资金突破臻选` 的 exit 也能继续抬升收益，但提升更多来自“更完整地吃 pattern move”，不是简单地更激进止盈
- 这两组都还是研究态推荐配置，下一步先做更长样本验证，再考虑推进测试环境

### 9. 2026-03-16 selective exit 长样本验证

长样本 A/B 记录在：

- [selective_exit_validation_20260316.md](/Users/lvxue/work/量化/output/research_backtests/selective_exit_validation_20260316.md)

验证窗口：

- `2025-01-01` 到 `2026-03-12`

`龙门双阶强势臻选` 长样本结果：

- baseline（`research_limitup_selective`）：
  - `filled_trades=10`
  - `avg_exit_return_pct=-0.5469%`
  - `exit_win_rate_pct=50.00%`
  - `profit_factor=0.7490`
- best-exit（叠加 [exit_limitup_l1l2_research_best.json](/Users/lvxue/work/量化/configs/exit_limitup_l1l2_research_best.json)）：
  - `filled_trades=10`
  - `avg_exit_return_pct=+1.8983%`
  - `exit_win_rate_pct=50.00%`
  - `profit_factor=1.8758`
- 结论：
  - long window 下，exit preset 仍然有效
  - 但这条策略整体强度还不够，仍应保持“研究候选”状态

`真实资金突破臻选` 长样本结果：

- baseline（`real_breakout_downtrend_selective`）：
  - `filled_trades=20`
  - `avg_exit_return_pct=+0.8919%`
  - `exit_win_rate_pct=55.00%`
  - `profit_factor=1.8551`
- best-exit（叠加 [exit_real_breakout_research_best.json](/Users/lvxue/work/量化/configs/exit_real_breakout_research_best.json)）：
  - `filled_trades=20`
  - `avg_exit_return_pct=+1.3981%`
  - `exit_win_rate_pct=60.00%`
  - `profit_factor=2.1294`
- 结论：
  - 这条策略在长样本下更稳
  - 当前更接近测试环境候选，但仍建议先过 walk-forward

### Walk-Forward 验证

当前已经支持独立的 walk-forward 验证入口：

```bash
python3 /Users/lvxue/work/量化/scripts/run_price_strategy_walkforward.py \
  --strategy-id platform_breakout \
  --start-date 20260101 \
  --end-date 20260312 \
  --train-trade-days 60 \
  --validation-trade-days 20 \
  --step-trade-days 20 \
  --trials 40
```

这层和参数优化器的区别是：

- 参数优化器：告诉你“在某段样本里，哪组参数更好”
- walk-forward：告诉你“训练期挑出来的参数，到了下一段没见过的数据里还能不能打”

当前 walk-forward 的做法：

1. 先按总区间切出多个 fold
2. 每个 fold 里用训练窗口做随机搜索
3. 只拿训练期最优参数去验证窗口跑样本外
4. 同时保留 baseline 配置，方便和默认参数比较

导出目录通常包括：

- `fold_results.csv`
- `fold_trial_results.csv`
- `aggregate_summary.csv`
- `summary.json`

常看指标：

- `avg_best_valid_return_pct`
- `avg_best_valid_win_rate_pct`
- `total_best_valid_trades`
- `beat_baseline_ratio_pct`

推荐使用习惯：

- 先用 `run_price_strategy_research_suite.py` 看全局基线
- 先用 `optimize_price_strategy_params.py` 粗搜
- 再用 `run_price_strategy_walkforward.py` 做样本外验证
- 只有当 walk-forward 也稳定时，才考虑把参数往测试环境推进

## 当前已知情况

### 1. 双底策略当前是本地实验态

- 已完成独立形态模块和 runner
- 已能在本地基于缓存完成离线筛选
- 还没有接到 `/Users/lvxue/work/策略实验室` 的网页和调度链路

### 2. 本地缓存会复用历史策略缓存

双底 runner 当前会优先读取：

- `double_bottom_api`
- `platform_breakout_api`
- `limitup_l1l2_api`

这样做是为了：

- 本地验证更快
- 接口暂时不可用时，也能用已有缓存跑 smoke test

### 3. 离线测试可能出现 0 候选

这不一定代表代码错误。

对双底策略来说，当前版本本来就偏严格：

- 成功率优先
- 宁可少选
- 不满足标准右侧结构就返回空结果

### 4. 研究框架会暴露“当前策略真实行为”，不自动替你修正规则

例如当前 smoke test 已经暴露出：

- `龙门双阶强势臻选` 的研究回放里，仍可能挑到 `ST` 股票

这说明：

- 研究框架会尽量反映当前策略现状
- 但不会偷偷替 live 逻辑做过滤修正
- 如果后面要修规则，应单独改策略本身，再重新回测

### 5. 结构策略为了研究态卖出逻辑，已经多导出了一些关键价位字段

这些字段是向后兼容新增，主要服务于本地研究：

- `龙门双阶强势臻选`
  - `limitup_l1l2_l1_price`
  - `limitup_l1l2_l2_price`
  - `limitup_l1l2_impulse_high`
  - `limitup_l1l2_current_price`

- `天衡回踩转强臻选`
  - `platform_breakout_platform_high`
  - `platform_breakout_platform_low`
  - `platform_breakout_limit_high`
  - `platform_breakout_pullback_low_price`
  - `platform_breakout_support_floor`
  - `platform_breakout_current_price`

这样后续不管是研究层还是线上，如果要做“结构失效提醒”，都不需要再倒推这些价位。

### 6. walk-forward 当前已经在本地 smoke test 跑通

示例导出目录：

- [walkforward_platform_breakout_20260301_20260312_161448](/Users/lvxue/work/量化/output/research_backtests/walkforward_platform_breakout_20260301_20260312_161448)

这个样本还很小，主要用于验证链路，不适合直接拿来做结论。

### 7. 研究态交易日日历已经修正为优先使用 `trade_cal`

之前如果请求区间里刚好命中一部分缓存日期，研究层会把那部分缓存误当成完整交易日历。

现在的处理已经改成：

- 优先用 `trade_cal` 拿完整开市日
- 只有 `trade_cal` 失败时才退回缓存日期

这对下面几类研究尤其重要：

- 长区间基线回测
- 参数优化
- walk-forward

否则历史窗口长度会被静默截短，结果容易失真。

### 8. 2026-03-17 星曜增持 snapshot replay / 卖出 / walk-forward 最新结论

本轮 `星曜增持臻选` 的本地研究已经补齐到：

- 阶段性总结：
  - [holder_replay_validation_20260317.md](/Users/lvxue/work/量化/output/research_backtests/holder_replay_validation_20260317.md)

- snapshot replay 回测：
  - [run_holder_strategy_replay_backtest.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_replay_backtest.py)
- 卖出优化：
  - [optimize_holder_exit_rules.py](/Users/lvxue/work/量化/scripts/optimize_holder_exit_rules.py)
- snapshot walk-forward：
  - [run_holder_strategy_walkforward.py](/Users/lvxue/work/量化/scripts/run_holder_strategy_walkforward.py)

关键研究目录：

- baseline replay：
  - [holder_replay_backtest_20260309_20260312_142219](/Users/lvxue/work/量化/output/research_backtests/holder_replay_backtest_20260309_20260312_142219)
- baseline replay（补到 5 个快照）：
  - [holder_replay_backtest_20260306_20260312_143100](/Users/lvxue/work/量化/output/research_backtests/holder_replay_backtest_20260306_20260312_143100)
- 研究入口参数 replay：
  - [holder_replay_backtest_20260306_20260312_142737](/Users/lvxue/work/量化/output/research_backtests/holder_replay_backtest_20260306_20260312_142737)
- 默认入口下的卖出优化：
  - [optimize_exit_holder_increase_20260306_20260312_142414](/Users/lvxue/work/量化/output/research_backtests/optimize_exit_holder_increase_20260306_20260312_142414)
- 研究入口参数下的卖出优化：
  - [optimize_exit_holder_increase_20260306_20260312_142809](/Users/lvxue/work/量化/output/research_backtests/optimize_exit_holder_increase_20260306_20260312_142809)
- snapshot walk-forward：
  - [walkforward_holder_increase_20260306_20260312_142652](/Users/lvxue/work/量化/output/research_backtests/walkforward_holder_increase_20260306_20260312_142652)
- 研究入口参数下的 snapshot walk-forward：
  - [walkforward_holder_increase_20260306_20260312_143243](/Users/lvxue/work/量化/output/research_backtests/walkforward_holder_increase_20260306_20260312_143243)

当前可确认的短样本结论：

- baseline replay（5 个快照）：
  - `filled_trades=5`
  - `avg_exit_return_pct=+0.4320%`
  - `exit_win_rate_pct=60.00%`
  - `profit_factor=2.5334`
- 研究入口参数 [holder_strategy_research_best.json](/Users/lvxue/work/量化/configs/holder_strategy_research_best.json) replay：
  - `filled_trades=5`
  - `avg_exit_return_pct=+0.5943%`
  - `exit_win_rate_pct=60.00%`
  - `profit_factor=3.1098`
- 卖出优化：
  - 在默认入口和研究入口参数下，随机搜索都没有打败 baseline exit
  - 当前这条策略的短样本主矛盾仍然是入口与排序，不是卖出规则
- snapshot walk-forward：
  - `2` 个 fold
  - `avg_best_valid_return_pct=-0.7043%`
  - `beat_baseline_ratio_pct=0.0%`
  - 研究入口参数做 walk-forward 时，最佳配置会回到同一组入口参数，本质上没有额外样本外提升
  - 说明当前样本还太少，不足以支撑“walk-forward 已经稳定有效”的结论

当前判断：

- `星曜增持臻选` 线上逻辑先稳住，不建议因为这轮短样本去改 live 卖出
- 本地研究继续沿 `snapshot replay -> 补历史快照 -> 再做长窗口 walk-forward` 这条线推进
- 当前最值得保留的是入口参数文件：
  - [holder_strategy_research_best.json](/Users/lvxue/work/量化/configs/holder_strategy_research_best.json)
- 当前不建议额外固化单独的 holder exit preset，因为默认卖出已经打平最优

额外修正：

- [holder_replay_utils.py](/Users/lvxue/work/量化/scripts/holder_replay_utils.py) 现在会把 replay 结果里的 `chip_score_x / chip_score_y` 合并回单列 `chip_score`，避免回测导出里出现重复字段。

## 与网页项目的关系

这个目录里的策略只有在被 `/Users/lvxue/work/策略实验室` 接线后，才会出现在网页和线上调度里。

所以后续如果要把 `玄枢双底反转臻选` 真正接进系统，还需要至少做这些事：

1. 在网页项目里注册策略定义
2. 接策略运行入口
3. 接首页与策略页展示
4. 更新网页项目 README
5. 经过你明确同意后，才发布到阿里云

## 后续维护原则

以后如果改了：

- 双底结构定义
- 评分权重
- 买点分类
- runner 参数
- 研究回测口径
- 市场状态分类逻辑
- 导出字段
- 是否接网页

都要同步更新这份 README。

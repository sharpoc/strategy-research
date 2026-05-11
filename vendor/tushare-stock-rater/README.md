# Tushare 单股买入评分系统

这是一个独立项目，用 Tushare 数据对 A 股做两类研究：

- 单股买入价值评估：输出综合评分、评级、证据和风险报告
- 尾盘反抽候选筛选：收盘前找“跌得较多、但还没完全走坏，次日可能技术性反弹”的票

## 安装

```bash
cd /Users/lvxue/work/tushare-stock-rater
python3 -m pip install -e .
```

配置 token：

```bash
cp .env.example .env
# 编辑 .env，只填 TUSHARE_TOKEN
```

也可以直接使用环境变量：

```bash
export TUSHARE_TOKEN="你的token"
```

## 使用

```bash
python3 -m tushare_stock_rater analyze 600519.SH --as-of 20260417 --lookback 252 --out reports
```

也可以输入 6 位代码：

```bash
python3 -m tushare_stock_rater analyze 600519
```

尾盘反抽候选：

```bash
python3 -m tushare_stock_rater pick-rebound --top 5
```

历史回放：

```bash
python3 -m tushare_stock_rater pick-rebound --as-of 20260417 --historical --top 5
```

输出目录：

```text
reports/600519.SH_20260417/
  report.md
  result.json
  features.csv
  data_meta.json
```

尾盘反抽输出目录：

```text
reports/rebound_20260422_live/
  summary.md
  summary.json
  candidates.csv
```

## 尾盘反抽逻辑

默认只看 `主板 + 创业板`，并做这些过滤：

- 当日跌幅在 `-6.8% ~ -3.0%`
- 相对所属行业有额外超跌，默认至少 `-1.5%`
- 20 日均成交额、流通市值达到门槛，避免小票和流动性陷阱
- 不接近跌停，不碰 `ST/*ST`、次新、北交所
- 收盘前价格要明显离开日内低点
- 自动剔除近期业绩快报明显转弱、未来 30 日解禁偏大、近 60 日明显减持的票

结果里会给出：

- `股票代码 + 名称 + 得分`
- `跌幅 + 行业超跌`
- `尾盘观察区间`
- `止损位`
- `次日先看昨收附近是否有反抽`

## 实时接口说明

- 实时模式依赖 `rt_k`
- 当前实现会自动跳过 `rt_k` 的本地缓存，避免把旧实时数据当成新数据
- 如果你的 token 没有 `rt_min_daily` 或 `anns_d` 权限，命令会自动降级，不会直接报废

## 评分结构

综合分 `total_score` 为 `0-100`，同时输出 `confidence_score`：

- 财务质量：25 分
- 成长与业绩趋势：15 分
- 估值合理性：15 分
- 技术趋势与买点：20 分
- 资金流与流动性：15 分
- 事件与治理：10 分

评级：

- `A >= 80`：值得重点研究
- `B 70-79`：谨慎关注
- `C 60-69`：观望，等待更好买点
- `D < 60`：不建议买入

硬性风险会限制最高评级，例如 `ST/*ST`、退市整理、长期停牌、大额解禁、核心股东明显减持、现金流严重恶化等。

## 数据来源

所有数据都来自 Tushare，包括：

- `stock_basic`
- `trade_cal`
- `daily`
- `adj_factor`
- `daily_basic`
- `income`
- `balancesheet`
- `cashflow`
- `fina_indicator`
- `moneyflow`
- `forecast`
- `express`
- `anns_d`
- `stk_holdertrade`
- `share_float`
- `index_daily`
- `index_dailybasic`

## 注意

这是研究辅助工具，不自动下单，也不替代投资顾问建议。无论是单股评分还是尾盘反抽候选，都只适合做进一步筛查，不代表确定收益。

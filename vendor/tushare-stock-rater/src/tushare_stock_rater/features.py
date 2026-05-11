from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .models import DataBundle, FeatureSet
from .utils import latest_record, to_float, to_number


RISK_KEYWORDS = ["风险", "诉讼", "仲裁", "处罚", "立案", "问询", "监管", "退市", "亏损", "减持"]
CATALYST_KEYWORDS = ["增持", "回购", "预增", "中标", "合同", "重组", "分红", "扩产"]


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _sort_by_date(df: pd.DataFrame, column: str = "trade_date", ascending: bool = True) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame() if df is None else df.copy()
    work = df.copy()
    work[column] = work[column].astype(str)
    return work.sort_values(column, ascending=ascending).reset_index(drop=True)


def _pct_return(series: pd.Series, days: int) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= days:
        return None
    base = clean.iloc[-days - 1]
    latest = clean.iloc[-1]
    if base == 0:
        return None
    return float((latest / base - 1) * 100)


def _latest_n_unique(df: pd.DataFrame, date_col: str = "end_date", n: int = 8) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if date_col not in work.columns:
        return work.head(n)
    sort_cols = [date_col]
    if "ann_date" in work.columns:
        sort_cols.append("ann_date")
    work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return work.drop_duplicates(subset=[date_col], keep="first").head(n).reset_index(drop=True)


def _main_moneyflow_amount(row: pd.Series) -> float | None:
    net = to_float(row.get("net_mf_amount"))
    if net is not None:
        return net
    buy_lg = to_float(row.get("buy_lg_amount")) or 0.0
    buy_elg = to_float(row.get("buy_elg_amount")) or 0.0
    sell_lg = to_float(row.get("sell_lg_amount")) or 0.0
    sell_elg = to_float(row.get("sell_elg_amount")) or 0.0
    if any(col in row.index for col in ["buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"]):
        return buy_lg + buy_elg - sell_lg - sell_elg
    return None


def _text_contains(value: Any, words: list[str]) -> bool:
    text = str(value or "")
    return any(word in text for word in words)


def _add_point(points: list[str], text: str) -> None:
    if text and text not in points:
        points.append(text)


def build_features(bundle: DataBundle) -> FeatureSet:
    frames = bundle.frames
    stock = bundle.stock
    values: dict[str, Any] = {
        "ts_code": stock.ts_code,
        "name": stock.name,
        "industry": stock.industry,
        "market": stock.market,
        "as_of_date": bundle.as_of_date,
    }
    evidence: dict[str, list[str]] = {
        "financial_quality": [],
        "growth": [],
        "valuation": [],
        "technical": [],
        "moneyflow": [],
        "events": [],
    }
    positive_points: list[str] = []
    negative_points: list[str] = []
    risk_flags: list[str] = []
    hard_risks: list[str] = []
    severe_risks: list[str] = []

    missing_tables = [
        meta.endpoint
        for meta in bundle.table_meta
        if meta.row_count == 0 and meta.endpoint not in {"forecast", "express", "anns_d", "stk_holdertrade", "share_float"}
    ]

    daily = _sort_by_date(frames.get("daily", pd.DataFrame()))
    adj = _sort_by_date(frames.get("adj_factor", pd.DataFrame()))
    if not daily.empty:
        daily = _numeric(daily, ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"])
        if not adj.empty and "adj_factor" in adj.columns:
            adj = _numeric(adj, ["adj_factor"])
            daily = daily.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
            latest_factor = pd.to_numeric(daily["adj_factor"], errors="coerce").dropna()
            factor_base = float(latest_factor.iloc[-1]) if not latest_factor.empty else 1.0
            if factor_base:
                daily["close_qfq"] = daily["close"] * daily["adj_factor"] / factor_base
            else:
                daily["close_qfq"] = daily["close"]
        else:
            daily["close_qfq"] = daily["close"]

        latest_daily = daily.iloc[-1]
        latest_trade_date = str(latest_daily.get("trade_date") or "")
        values["latest_trade_date"] = latest_trade_date
        values["latest_close"] = to_number(latest_daily.get("close"))
        values["latest_close_qfq"] = to_number(latest_daily.get("close_qfq"))
        values["latest_pct_chg"] = to_number(latest_daily.get("pct_chg"))
        values["return_20d"] = to_number(_pct_return(daily["close_qfq"], 20))
        values["return_60d"] = to_number(_pct_return(daily["close_qfq"], 60))
        values["return_120d"] = to_number(_pct_return(daily["close_qfq"], 120))

        for window in [5, 10, 20, 60, 120]:
            if len(daily) >= window:
                values[f"ma_{window}"] = to_number(daily["close_qfq"].rolling(window).mean().iloc[-1])
        if len(daily) >= 252:
            low_252 = float(daily["close_qfq"].tail(252).min())
            high_252 = float(daily["close_qfq"].tail(252).max())
        else:
            low_252 = float(daily["close_qfq"].min())
            high_252 = float(daily["close_qfq"].max())
        latest_close = to_float(values.get("latest_close_qfq"))
        if latest_close is not None and high_252 > low_252:
            values["price_position_252"] = to_number((latest_close - low_252) / (high_252 - low_252), 4)
            values["drawdown_from_252_high"] = to_number((latest_close / high_252 - 1) * 100)
        returns = pd.to_numeric(daily["close_qfq"], errors="coerce").pct_change().dropna()
        if not returns.empty:
            values["annualized_volatility"] = to_number(float(returns.tail(60).std()) * math.sqrt(252) * 100)
        if "vol" in daily.columns and len(daily) >= 21:
            avg_vol_20 = daily["vol"].tail(21).iloc[:-1].mean()
            if avg_vol_20 and not pd.isna(avg_vol_20):
                values["volume_ratio"] = to_number(float(daily["vol"].iloc[-1]) / float(avg_vol_20))

        asof_ts = pd.to_datetime(bundle.as_of_date, format="%Y%m%d")
        latest_ts = pd.to_datetime(latest_trade_date, format="%Y%m%d", errors="coerce")
        if pd.notna(latest_ts):
            values["calendar_days_since_trade"] = int((asof_ts - latest_ts).days)
            if latest_trade_date != bundle.as_of_date:
                risk_flags.append(f"最近行情日期为 {latest_trade_date}，不是评估日 {bundle.as_of_date}")
            if values["calendar_days_since_trade"] >= 10:
                hard_risks.append("近 10 个自然日无日线成交记录，可能长期停牌")

        ma20 = to_float(values.get("ma_20"))
        ma60 = to_float(values.get("ma_60"))
        ma120 = to_float(values.get("ma_120"))
        if latest_close is not None and ma20 is not None:
            if latest_close > ma20:
                _add_point(positive_points, "收盘价站上 20 日均线")
            else:
                _add_point(negative_points, "收盘价仍在 20 日均线下方")
        if ma20 is not None and ma60 is not None and ma120 is not None and ma20 > ma60 > ma120:
            _add_point(positive_points, "中短期均线呈多头排列")
        if to_float(values.get("return_20d")) is not None and to_float(values.get("return_20d")) > 35:
            _add_point(negative_points, "近 20 日涨幅偏高，追高风险上升")
        evidence["technical"].append(
            f"最新收盘 {values.get('latest_close')}，20日收益 {values.get('return_20d')}%，60日收益 {values.get('return_60d')}%"
        )
    else:
        hard_risks.append("daily 行情为空，无法验证交易状态")

    daily_basic = _sort_by_date(frames.get("daily_basic", pd.DataFrame()))
    if not daily_basic.empty:
        daily_basic = _numeric(daily_basic, ["turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"])
        latest_basic = daily_basic.iloc[-1].to_dict()
        for key in ["turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"]:
            values[key] = to_number(latest_basic.get(key), 4 if key in {"turnover_rate", "volume_ratio"} else 2)
        evidence["valuation"].append(
            f"PE(TTM) {values.get('pe_ttm')}，PB {values.get('pb')}，PS(TTM) {values.get('ps_ttm')}，股息率 {values.get('dv_ttm')}"
        )

    stock_basic = frames.get("stock_basic", pd.DataFrame()).copy()
    market_basic = frames.get("daily_basic_market", pd.DataFrame()).copy()
    if not stock_basic.empty and not market_basic.empty and "industry" in stock_basic.columns:
        peer = market_basic.merge(stock_basic[["ts_code", "industry"]], on="ts_code", how="left")
        peer = peer[peer["industry"].astype(str) == stock.industry].copy()
        for col in ["pe_ttm", "pb", "ps_ttm"]:
            if col in peer.columns:
                peer[col] = pd.to_numeric(peer[col], errors="coerce")
                value = to_float(values.get(col))
                series = peer[col].replace([np.inf, -np.inf], np.nan).dropna()
                series = series[series > 0]
                if value is not None and value > 0 and len(series) >= 5:
                    percentile = float((series <= value).mean())
                    values[f"industry_{col}_pct"] = to_number(percentile, 4)

    fina = _latest_n_unique(frames.get("fina_indicator", pd.DataFrame()), n=8)
    if not fina.empty:
        fina = _numeric(
            fina,
            [
                "roe",
                "grossprofit_margin",
                "netprofit_margin",
                "debt_to_assets",
                "ocf_to_or",
                "tr_yoy",
                "or_yoy",
                "netprofit_yoy",
                "dt_netprofit_yoy",
                "q_sales_yoy",
                "q_netprofit_yoy",
            ],
        )
        latest_fina = fina.iloc[0].to_dict()
        for key in [
            "roe",
            "grossprofit_margin",
            "netprofit_margin",
            "debt_to_assets",
            "ocf_to_or",
            "tr_yoy",
            "or_yoy",
            "netprofit_yoy",
            "dt_netprofit_yoy",
            "q_sales_yoy",
            "q_netprofit_yoy",
        ]:
            values[key] = to_number(latest_fina.get(key))
        values["positive_profit_growth_quarters"] = int((pd.to_numeric(fina.get("dt_netprofit_yoy"), errors="coerce") > 0).sum()) if "dt_netprofit_yoy" in fina.columns else None
        evidence["financial_quality"].append(
            f"ROE {values.get('roe')}%，毛利率 {values.get('grossprofit_margin')}%，净利率 {values.get('netprofit_margin')}%，资产负债率 {values.get('debt_to_assets')}%"
        )
        evidence["growth"].append(
            f"营收同比 {values.get('tr_yoy') or values.get('or_yoy')}%，扣非净利同比 {values.get('dt_netprofit_yoy')}%"
        )
        if to_float(values.get("roe")) is not None and to_float(values.get("roe")) >= 12:
            _add_point(positive_points, "ROE 达到较好水平")
        if to_float(values.get("debt_to_assets")) is not None and to_float(values.get("debt_to_assets")) >= 80:
            risk_flags.append("资产负债率偏高")

    cashflow = _latest_n_unique(frames.get("cashflow", pd.DataFrame()), n=8)
    income = _latest_n_unique(frames.get("income", pd.DataFrame()), n=8)
    if not cashflow.empty and not income.empty:
        cash_latest = latest_record(cashflow, ("end_date", "ann_date"))
        income_latest = latest_record(income, ("end_date", "ann_date"))
        ocf = to_float(cash_latest.get("n_cashflow_act"))
        profit = to_float(income_latest.get("n_income_attr_p")) or to_float(income_latest.get("n_income"))
        if ocf is not None:
            values["n_cashflow_act"] = to_number(ocf)
        if profit is not None:
            values["n_income_attr_p"] = to_number(profit)
        if ocf is not None and profit is not None and abs(profit) > 1e-9:
            values["cashflow_to_profit"] = to_number(ocf / profit, 4)
            evidence["financial_quality"].append(f"经营现金流/归母净利润 {values.get('cashflow_to_profit')}")
            if ocf / profit < -0.2:
                hard_risks.append("经营现金流明显弱于利润")

    forecast = _sort_by_date(frames.get("forecast", pd.DataFrame()), "ann_date", ascending=False)
    if not forecast.empty:
        latest_forecast = forecast.iloc[0].to_dict()
        forecast_type = str(latest_forecast.get("type") or latest_forecast.get("forecast_type") or "")
        p_min = to_float(latest_forecast.get("p_change_min"))
        p_max = to_float(latest_forecast.get("p_change_max"))
        values["forecast_type"] = forecast_type
        values["forecast_p_change_avg"] = to_number(np.nanmean([v for v in [p_min, p_max] if v is not None])) if p_min is not None or p_max is not None else None
        evidence["growth"].append(f"最近业绩预告：{forecast_type or '未标明类型'}，变动均值 {values.get('forecast_p_change_avg')}%")
        if _text_contains(forecast_type, ["预亏", "首亏", "续亏", "略减", "预减"]) or (to_float(values.get("forecast_p_change_avg")) is not None and to_float(values.get("forecast_p_change_avg")) < -30):
            hard_risks.append("最近业绩预告偏负面")
        elif _text_contains(forecast_type, ["预增", "略增", "扭亏", "续盈"]):
            _add_point(positive_points, "最近业绩预告偏正面")

    moneyflow = _sort_by_date(frames.get("moneyflow", pd.DataFrame()))
    if not moneyflow.empty:
        amounts = moneyflow.apply(_main_moneyflow_amount, axis=1)
        moneyflow = moneyflow.assign(main_net_amount=amounts)
        values["main_net_amount_3d"] = to_number(pd.to_numeric(moneyflow["main_net_amount"], errors="coerce").tail(3).sum())
        values["main_net_amount_5d"] = to_number(pd.to_numeric(moneyflow["main_net_amount"], errors="coerce").tail(5).sum())
        values["main_net_amount_20d"] = to_number(pd.to_numeric(moneyflow["main_net_amount"], errors="coerce").tail(20).sum())
        values["main_net_positive_days_5d"] = int((pd.to_numeric(moneyflow["main_net_amount"], errors="coerce").tail(5) > 0).sum())
        values["main_net_positive_days_20d"] = int((pd.to_numeric(moneyflow["main_net_amount"], errors="coerce").tail(20) > 0).sum())
        evidence["moneyflow"].append(
            f"主力净流 5日 {values.get('main_net_amount_5d')}，20日 {values.get('main_net_amount_20d')}，5日流入天数 {values.get('main_net_positive_days_5d')}"
        )
        if (to_float(values.get("main_net_amount_5d")) or 0) > 0 and (values.get("main_net_positive_days_5d") or 0) >= 3:
            _add_point(positive_points, "近 5 日资金流入连续性较好")
        if (to_float(values.get("main_net_amount_20d")) or 0) < 0 and (values.get("main_net_positive_days_20d") or 0) <= 6:
            _add_point(negative_points, "近 20 日资金流偏弱")

    holder = _sort_by_date(frames.get("stk_holdertrade", pd.DataFrame()), "ann_date", ascending=False)
    if not holder.empty:
        holder = holder.copy()
        if "change_ratio" in holder.columns:
            holder["change_ratio"] = pd.to_numeric(holder["change_ratio"], errors="coerce").fillna(0.0)
        else:
            holder["change_ratio"] = 0.0
        text_cols = [col for col in ["in_de", "change_dir", "change_type", "holder_type", "holder_name"] if col in holder.columns]
        holder["_text"] = holder[text_cols].astype(str).agg(" ".join, axis=1) if text_cols else ""
        decrease = holder[holder["_text"].str.contains("DE|减", regex=True, na=False)]
        increase = holder[holder["_text"].str.contains("IN|增", regex=True, na=False)]
        core_decrease = decrease[decrease["_text"].str.contains("高管|董|监|核心|管理", regex=True, na=False)]
        values["holder_increase_events_180d"] = int(len(increase))
        values["holder_decrease_events_180d"] = int(len(decrease))
        values["core_holder_decrease_ratio_180d"] = to_number(core_decrease["change_ratio"].abs().sum()) if not core_decrease.empty else 0.0
        evidence["events"].append(
            f"近 180 日增持事件 {values.get('holder_increase_events_180d')}，减持事件 {values.get('holder_decrease_events_180d')}"
        )
        if values["holder_increase_events_180d"] > 0:
            _add_point(positive_points, "近 180 日存在增持记录")
        if values["holder_decrease_events_180d"] > 0:
            risk_flags.append("近 180 日存在减持记录")
        if (to_float(values.get("core_holder_decrease_ratio_180d")) or 0) >= 0.8:
            hard_risks.append("核心股东/高管减持比例偏高")

    share_float = _sort_by_date(frames.get("share_float", pd.DataFrame()), "float_date", ascending=True)
    if not share_float.empty:
        share_float = _numeric(share_float, ["float_ratio"])
        total_unlock_ratio = float(pd.to_numeric(share_float.get("float_ratio"), errors="coerce").fillna(0.0).sum()) if "float_ratio" in share_float.columns else 0.0
        values["unlock_ratio_60d"] = to_number(total_unlock_ratio, 4)
        nearest_ratio = to_float(share_float.iloc[0].get("float_ratio")) if "float_ratio" in share_float.columns else None
        values["nearest_unlock_date"] = str(share_float.iloc[0].get("float_date") or "")
        values["nearest_unlock_ratio"] = to_number(nearest_ratio, 4)
        evidence["events"].append(
            f"未来解禁合计 {values.get('unlock_ratio_60d')}%，最近解禁 {values.get('nearest_unlock_date')} / {values.get('nearest_unlock_ratio')}%"
        )
        if total_unlock_ratio >= 8 or (nearest_ratio is not None and nearest_ratio >= 3):
            hard_risks.append("短期解禁压力偏大")

    anns = _sort_by_date(frames.get("anns_d", pd.DataFrame()), "ann_date", ascending=False)
    if not anns.empty:
        title_col = "title" if "title" in anns.columns else ("ann_title" if "ann_title" in anns.columns else "")
        if title_col:
            titles = anns[title_col].astype(str).head(30).tolist()
            risk_titles = [title for title in titles if _text_contains(title, RISK_KEYWORDS)]
            catalyst_titles = [title for title in titles if _text_contains(title, CATALYST_KEYWORDS)]
            values["risk_announcement_count_180d"] = len(risk_titles)
            values["catalyst_announcement_count_180d"] = len(catalyst_titles)
            if risk_titles:
                risk_flags.append(f"近期风险类公告 {len(risk_titles)} 条")
                evidence["events"].append("风险公告样例：" + "；".join(risk_titles[:3]))
            if catalyst_titles:
                _add_point(positive_points, f"近期催化类公告 {len(catalyst_titles)} 条")

    if stock.name.upper().startswith("*ST") or stock.name.upper().startswith("ST") or "ST" in stock.name.upper():
        severe_risks.append("股票名称包含 ST/*ST")
    if stock.list_status in {"D", "P"}:
        severe_risks.append(f"上市状态异常：{stock.list_status}")

    if not positive_points:
        positive_points.append("暂无特别突出的正面信号")
    if not negative_points:
        negative_points.append("暂无特别突出的负面信号")
    if not risk_flags and not hard_risks and not severe_risks:
        risk_flags.append("未发现硬性风险，但仍需结合交易计划控制仓位")

    return FeatureSet(
        stock=stock,
        as_of_date=bundle.as_of_date,
        values=values,
        evidence=evidence,
        positive_points=positive_points,
        negative_points=negative_points,
        risk_flags=risk_flags,
        hard_risks=hard_risks,
        severe_risks=severe_risks,
        missing_tables=missing_tables,
        data_warnings=bundle.warnings,
    )

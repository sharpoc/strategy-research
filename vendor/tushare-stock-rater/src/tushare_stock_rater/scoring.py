from __future__ import annotations

from typing import Any

from .config import load_config
from .models import FeatureSet, RatingResult, SectionScore
from .utils import clip, score_range, to_float


SECTION_NAMES = {
    "financial_quality": "财务质量",
    "growth": "成长与业绩趋势",
    "valuation": "估值合理性",
    "technical": "技术趋势与买点",
    "moneyflow": "资金流与流动性",
    "events": "事件与治理",
}


def _score_forecast(feature_type: Any, change_avg: Any, max_score: float) -> float:
    text = str(feature_type or "")
    change = to_float(change_avg)
    if any(word in text for word in ["预增", "略增", "扭亏", "续盈"]):
        return max_score
    if any(word in text for word in ["预亏", "首亏", "续亏", "预减", "略减"]):
        return 0.0
    if change is None:
        return max_score * 0.5
    if change >= 30:
        return max_score
    if change <= -30:
        return 0.0
    return max_score * (change + 30) / 60


def score_financial(values: dict[str, Any], max_score: float) -> tuple[float, list[str]]:
    score = 0.0
    score += score_range(values.get("roe"), good=15, weak=0, max_score=7)
    score += score_range(values.get("grossprofit_margin"), good=35, weak=8, max_score=4)
    score += score_range(values.get("netprofit_margin"), good=12, weak=0, max_score=4)
    score += score_range(values.get("cashflow_to_profit"), good=1.0, weak=0.0, max_score=6)
    score += score_range(values.get("debt_to_assets"), good=45, weak=80, max_score=4, reverse=True)
    evidence = [
        f"ROE={values.get('roe')}%，毛利率={values.get('grossprofit_margin')}%，净利率={values.get('netprofit_margin')}%",
        f"经营现金流/利润={values.get('cashflow_to_profit')}，资产负债率={values.get('debt_to_assets')}%",
    ]
    return round(clip(score, 0, max_score), 2), evidence


def score_growth(values: dict[str, Any], max_score: float) -> tuple[float, list[str]]:
    revenue_growth = values.get("tr_yoy") if values.get("tr_yoy") is not None else values.get("or_yoy")
    score = 0.0
    score += score_range(revenue_growth, good=20, weak=-5, max_score=4)
    score += score_range(values.get("dt_netprofit_yoy"), good=25, weak=-15, max_score=5)
    score += score_range(values.get("q_netprofit_yoy"), good=20, weak=-15, max_score=2)
    quarters = to_float(values.get("positive_profit_growth_quarters"))
    score += 2 if quarters is not None and quarters >= 4 else (1 if quarters is not None and quarters >= 2 else 0.5)
    score += _score_forecast(values.get("forecast_type"), values.get("forecast_p_change_avg"), max_score=2)
    evidence = [
        f"营收同比={revenue_growth}%，扣非净利同比={values.get('dt_netprofit_yoy')}%，单季净利同比={values.get('q_netprofit_yoy')}%",
        f"近8期扣非净利同比为正次数={values.get('positive_profit_growth_quarters')}，业绩预告={values.get('forecast_type') or '无'}",
    ]
    return round(clip(score, 0, max_score), 2), evidence


def score_valuation(values: dict[str, Any], max_score: float) -> tuple[float, list[str]]:
    pe_pct = values.get("industry_pe_ttm_pct")
    pb_pct = values.get("industry_pb_pct")
    ps_pct = values.get("industry_ps_ttm_pct")
    score = 0.0
    score += score_range(pe_pct, good=0.35, weak=0.95, max_score=4, reverse=True) if pe_pct is not None else score_range(values.get("pe_ttm"), good=12, weak=55, max_score=4, reverse=True)
    score += score_range(pb_pct, good=0.35, weak=0.95, max_score=3, reverse=True) if pb_pct is not None else score_range(values.get("pb"), good=1.2, weak=6, max_score=3, reverse=True)
    score += score_range(ps_pct, good=0.35, weak=0.95, max_score=2, reverse=True) if ps_pct is not None else score_range(values.get("ps_ttm"), good=1.5, weak=12, max_score=2, reverse=True)
    score += score_range(values.get("pe_ttm"), good=15, weak=60, max_score=2, reverse=True)
    score += score_range(values.get("pb"), good=1.5, weak=8, max_score=2, reverse=True)
    score += score_range(values.get("dv_ttm"), good=3.0, weak=0.0, max_score=2)
    evidence = [
        f"PE(TTM)={values.get('pe_ttm')}，PB={values.get('pb')}，PS(TTM)={values.get('ps_ttm')}，股息率={values.get('dv_ttm')}",
        f"行业PE分位={values.get('industry_pe_ttm_pct')}，行业PB分位={values.get('industry_pb_pct')}，行业PS分位={values.get('industry_ps_ttm_pct')}",
    ]
    return round(clip(score, 0, max_score), 2), evidence


def score_technical(values: dict[str, Any], max_score: float) -> tuple[float, list[str]]:
    latest = to_float(values.get("latest_close_qfq"))
    ma20 = to_float(values.get("ma_20"))
    ma60 = to_float(values.get("ma_60"))
    ma120 = to_float(values.get("ma_120"))
    score = 0.0
    if latest is not None and ma20 is not None:
        score += 4 if latest > ma20 else 1
    else:
        score += 2
    if ma20 is not None and ma60 is not None:
        score += 4 if ma20 > ma60 else 1.5
    else:
        score += 2
    if ma20 is not None and ma60 is not None and ma120 is not None:
        score += 3 if ma20 > ma60 > ma120 else 1
    else:
        score += 1.5
    ret20 = to_float(values.get("return_20d"))
    if ret20 is None:
        score += 1.5
    elif -8 <= ret20 <= 18:
        score += 3
    elif 18 < ret20 <= 35:
        score += 2
    elif ret20 < -18 or ret20 > 45:
        score += 0
    else:
        score += 1
    score += score_range(values.get("drawdown_from_252_high"), good=-12, weak=-55, max_score=2)
    position = to_float(values.get("price_position_252"))
    if position is None:
        score += 1.5
    elif 0.20 <= position <= 0.75:
        score += 3
    elif position < 0.12 or position > 0.92:
        score += 0.8
    else:
        score += 1.8
    volume_ratio = to_float(values.get("volume_ratio"))
    if volume_ratio is None:
        score += 1
    elif 1.05 <= volume_ratio <= 2.5:
        score += 2
    elif volume_ratio > 4:
        score += 0.5
    else:
        score += 1
    evidence = [
        f"最新前复权收盘={values.get('latest_close_qfq')}，MA20={values.get('ma_20')}，MA60={values.get('ma_60')}，MA120={values.get('ma_120')}",
        f"20日涨跌={values.get('return_20d')}%，252日位置={values.get('price_position_252')}，量比={values.get('volume_ratio')}",
    ]
    return round(clip(score, 0, max_score), 2), evidence


def score_moneyflow(values: dict[str, Any], max_score: float) -> tuple[float, list[str]]:
    score = 0.0
    score += score_range(values.get("main_net_amount_5d"), good=0, weak=-50000, max_score=4)
    score += score_range(values.get("main_net_amount_20d"), good=0, weak=-100000, max_score=3)
    score += score_range(values.get("main_net_positive_days_5d"), good=4, weak=1, max_score=3)
    score += score_range(values.get("main_net_positive_days_20d"), good=12, weak=5, max_score=3)
    turnover = to_float(values.get("turnover_rate"))
    if turnover is None:
        score += 1
    elif 1 <= turnover <= 8:
        score += 2
    elif turnover > 15:
        score += 0.5
    else:
        score += 1
    evidence = [
        f"5日主力净流={values.get('main_net_amount_5d')}，20日主力净流={values.get('main_net_amount_20d')}",
        f"5日流入天数={values.get('main_net_positive_days_5d')}，20日流入天数={values.get('main_net_positive_days_20d')}，换手率={values.get('turnover_rate')}%",
    ]
    return round(clip(score, 0, max_score), 2), evidence


def score_events(values: dict[str, Any], feature: FeatureSet, max_score: float) -> tuple[float, list[str]]:
    score = 5.0
    if (to_float(values.get("holder_increase_events_180d")) or 0) > 0:
        score += 2
    if (to_float(values.get("holder_decrease_events_180d")) or 0) > 0:
        score -= 2
    if (to_float(values.get("unlock_ratio_60d")) or 0) >= 8:
        score -= 3
    elif (to_float(values.get("unlock_ratio_60d")) or 0) > 0:
        score -= 1
    if (to_float(values.get("catalyst_announcement_count_180d")) or 0) > 0:
        score += 2
    if (to_float(values.get("risk_announcement_count_180d")) or 0) > 0:
        score -= 2
    if feature.hard_risks:
        score -= 2
    if feature.severe_risks:
        score = 0
    evidence = [
        f"增持事件={values.get('holder_increase_events_180d')}，减持事件={values.get('holder_decrease_events_180d')}，60日解禁={values.get('unlock_ratio_60d')}%",
        f"风险公告={values.get('risk_announcement_count_180d')}，催化公告={values.get('catalyst_announcement_count_180d')}",
    ]
    return round(clip(score, 0, max_score), 2), evidence


def rating_from_score(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def verdict_from_rating(rating: str) -> str:
    return {
        "A": "值得重点研究",
        "B": "谨慎关注",
        "C": "观望，等待更好买点",
        "D": "不建议买入",
    }[rating]


def confidence_score(feature: FeatureSet) -> float:
    score = 100.0
    important_missing = [item for item in feature.missing_tables if item not in {"index_dailybasic_000001.SH"}]
    score -= min(35, len(important_missing) * 5)
    score -= min(25, len(feature.data_warnings) * 2)
    required_metrics = [
        "latest_close_qfq",
        "roe",
        "debt_to_assets",
        "pe_ttm",
        "pb",
        "return_20d",
        "main_net_amount_5d",
    ]
    missing_metrics = [metric for metric in required_metrics if feature.values.get(metric) is None]
    score -= min(25, len(missing_metrics) * 4)
    return round(clip(score, 20, 100), 2)


def score_stock(feature: FeatureSet, config_path: str | None = None) -> RatingResult:
    config = load_config(config_path)
    weights = config["section_weights"]
    values = feature.values
    calculators = {
        "financial_quality": lambda: score_financial(values, float(weights["financial_quality"])),
        "growth": lambda: score_growth(values, float(weights["growth"])),
        "valuation": lambda: score_valuation(values, float(weights["valuation"])),
        "technical": lambda: score_technical(values, float(weights["technical"])),
        "moneyflow": lambda: score_moneyflow(values, float(weights["moneyflow"])),
        "events": lambda: score_events(values, feature, float(weights["events"])),
    }
    sections: list[SectionScore] = []
    for key, calculator in calculators.items():
        score, local_evidence = calculator()
        combined_evidence = list(feature.evidence.get(key, [])) + local_evidence
        sections.append(
            SectionScore(
                key=key,
                name=SECTION_NAMES[key],
                max_score=float(weights[key]),
                score=score,
                evidence=combined_evidence,
            )
        )

    raw_total = round(sum(section.score for section in sections), 2)
    total = raw_total
    if feature.severe_risks:
        total = min(total, float(config["risk_caps"]["severe_risk_max_score"]))
    elif feature.hard_risks:
        total = min(total, float(config["risk_caps"]["hard_risk_max_score"]))
    total = round(clip(total, 0, 100), 2)
    rating = rating_from_score(total)

    return RatingResult(
        stock=feature.stock,
        as_of_date=feature.as_of_date,
        total_score=total,
        raw_total_score=raw_total,
        confidence_score=confidence_score(feature),
        rating=rating,
        verdict=verdict_from_rating(rating),
        section_scores=sections,
        positive_points=feature.positive_points,
        negative_points=feature.negative_points,
        risk_flags=feature.risk_flags,
        hard_risks=feature.hard_risks,
        severe_risks=feature.severe_risks,
        missing_tables=feature.missing_tables,
        data_warnings=feature.data_warnings,
        features=feature.values,
    )

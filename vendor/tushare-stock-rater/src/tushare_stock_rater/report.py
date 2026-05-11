from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .models import DataBundle, RatingResult, ReportPaths
from .utils import compact_join, json_safe, write_json


def _bullet(items: list[str]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item}" for item in items)


def _section_score_table(result: RatingResult) -> str:
    rows = ["| 分项 | 得分 | 满分 | 证据 |", "| --- | ---: | ---: | --- |"]
    for section in result.section_scores:
        rows.append(
            f"| {section.name} | {section.score:.2f} | {section.max_score:.0f} | {compact_join(section.evidence, limit=3)} |"
        )
    return "\n".join(rows)


def render_markdown(result: RatingResult, bundle: DataBundle) -> str:
    hard_risk_note = ""
    if result.severe_risks:
        hard_risk_note = "；触发严重风险，评级已被压低"
    elif result.hard_risks:
        hard_risk_note = "；触发硬性风险，评级最高不超过 C"

    lines = [
        f"# {result.stock.name}（{result.stock.ts_code}）买入评分报告",
        "",
        "## 一句话结论",
        "",
        f"{result.stock.name} 当前综合评分为 **{result.total_score:.2f}/100**，评级 **{result.rating}**，结论：**{result.verdict}**{hard_risk_note}。",
        "",
        "## 综合评分与评级",
        "",
        f"- 评估日期：`{result.as_of_date}`",
        f"- 原始综合分：`{result.raw_total_score:.2f}`",
        f"- 风险调整后综合分：`{result.total_score:.2f}`",
        f"- 置信度：`{result.confidence_score:.2f}`",
        f"- 行业：`{result.stock.industry or '未知'}`",
        f"- 市场：`{result.stock.market or '未知'}`",
        "",
        _section_score_table(result),
        "",
        "## 买入理由",
        "",
        _bullet(result.positive_points),
        "",
        "## 不买理由",
        "",
        _bullet(result.negative_points),
        "",
        "## 财务质量分析",
        "",
        _bullet(next((s.evidence for s in result.section_scores if s.key == "financial_quality"), [])),
        "",
        "## 估值分析",
        "",
        _bullet(next((s.evidence for s in result.section_scores if s.key == "valuation"), [])),
        "",
        "## 趋势与资金分析",
        "",
        _bullet(
            next((s.evidence for s in result.section_scores if s.key == "technical"), [])
            + next((s.evidence for s in result.section_scores if s.key == "moneyflow"), [])
        ),
        "",
        "## 事件/公告/解禁/增减持风险",
        "",
        "### 普通风险提示",
        "",
        _bullet(result.risk_flags),
        "",
        "### 硬性风险",
        "",
        _bullet(result.hard_risks),
        "",
        "### 严重风险",
        "",
        _bullet(result.severe_risks),
        "",
        "## 数据口径与接口清单",
        "",
        "| 接口 | 行数 | 缓存 | 状态 |",
        "| --- | ---: | --- | --- |",
    ]
    for meta in bundle.table_meta:
        status = meta.error or "OK"
        lines.append(f"| `{meta.endpoint}` | {meta.row_count} | {'是' if meta.cached else '否'} | {status} |")

    lines.extend(
        [
            "",
            "## 数据缺失或权限限制说明",
            "",
            _bullet(result.data_warnings or result.missing_tables),
            "",
            "## 免责声明",
            "",
            "本报告只基于 Tushare 数据做量化研究辅助，不构成投资建议，不自动下单，也不保证收益。",
            "",
        ]
    )
    return "\n".join(lines)


def result_to_dict(result: RatingResult) -> dict[str, Any]:
    data = asdict(result)
    data["section_scores"] = [asdict(section) | {"ratio": section.ratio} for section in result.section_scores]
    return json_safe(data)


def write_report_outputs(result: RatingResult, bundle: DataBundle, out_root: Path | str) -> ReportPaths:
    out_root = Path(out_root)
    output_dir = out_root / f"{result.stock.ts_code}_{result.as_of_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = ReportPaths(
        output_dir=output_dir,
        report_md=output_dir / "report.md",
        result_json=output_dir / "result.json",
        features_csv=output_dir / "features.csv",
        data_meta_json=output_dir / "data_meta.json",
    )

    paths.report_md.write_text(render_markdown(result, bundle), encoding="utf-8")
    write_json(paths.result_json, result_to_dict(result))
    pd.DataFrame([result.features]).to_csv(paths.features_csv, index=False)
    write_json(paths.data_meta_json, [asdict(meta) for meta in bundle.table_meta])
    return paths

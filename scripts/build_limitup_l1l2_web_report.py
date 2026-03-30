from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from run_tushare_limitup_l1l2_strategy import display_columns, json_safe, run_limitup_l1l2_screen
except ImportError:
    from scripts.run_tushare_limitup_l1l2_strategy import display_columns, json_safe, run_limitup_l1l2_screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the limit-up L1/L2 strategy across a date range and build a local HTML report.")
    parser.add_argument("--date-from", default="20260309", help="Start date in YYYYMMDD.")
    parser.add_argument("--date-to", default="", help="End date in YYYYMMDD. Default: today.")
    parser.add_argument("--history-bars", type=int, default=100, help="Recent open-market bars used for strategy reconstruction.")
    parser.add_argument("--moneyflow-lookback-days", type=int, default=5, help="Moneyflow lookback days.")
    parser.add_argument("--recent-buy-window", type=int, default=0, help="How many bars after a fresh pattern still count as active.")
    parser.add_argument("--min-score", type=float, default=55.0, help="Minimum strategy score to keep as a candidate.")
    parser.add_argument("--cutoff-hour", type=int, default=20, help="Use previous trading day before this hour.")
    parser.add_argument("--show-top", type=int, default=5, help="How many ranked candidates to include per day.")
    return parser.parse_args()


def build_html(payload: dict) -> str:
    rows = payload.get("days", [])
    cards = []
    for day in rows:
        best = day.get("best_pick") or {}
        top_candidates = day.get("top_candidates") or []
        badge = "有信号" if best else "无信号"
        best_name = best.get("name") or "-"
        best_code = best.get("ts_code") or "-"
        best_score = best.get("strategy_rank_score") or "-"
        best_reason = best.get("limitup_l1l2_reason") or "当日没有满足条件的股票"
        candidate_rows = "".join(
            f"""
            <tr>
              <td>{item.get("ts_code","-")}</td>
              <td>{item.get("name","-")}</td>
              <td>{item.get("strategy_rank_score","-")}</td>
              <td>{item.get("limitup_l1l2_stage","-")}</td>
              <td>{item.get("limitup_l1l2_l1_date","-")}</td>
              <td>{item.get("limitup_l1l2_l2_date","-")}</td>
            </tr>
            """
            for item in top_candidates
        )
        if not candidate_rows:
            candidate_rows = '<tr><td colspan="6">当天没有满足条件的候选</td></tr>'
        cards.append(
            f"""
            <section class="day-card">
              <div class="day-head">
                <div>
                  <p class="eyebrow">{day.get("display_date","")}</p>
                  <h2>{day.get("screen_end_date","")}</h2>
                </div>
                <span class="badge">{badge}</span>
              </div>
              <div class="hero-grid">
                <div class="hero-tile accent">
                  <p class="tile-label">最强一只</p>
                  <h3>{best_name}</h3>
                  <p class="tile-meta">{best_code}</p>
                </div>
                <div class="hero-tile">
                  <p class="tile-label">策略分</p>
                  <h3>{best_score}</h3>
                  <p class="tile-meta">{best_reason}</p>
                </div>
                <div class="hero-tile">
                  <p class="tile-label">候选数</p>
                  <h3>{day.get("strategy_candidates",0)}</h3>
                  <p class="tile-meta">L2 结构成立的股票数量</p>
                </div>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>代码</th>
                      <th>名称</th>
                      <th>排序分</th>
                      <th>阶段</th>
                      <th>L1</th>
                      <th>L2</th>
                    </tr>
                  </thead>
                  <tbody>{candidate_rows}</tbody>
                </table>
              </div>
            </section>
            """
        )

    cards_html = "\n".join(cards)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>涨停 L1/L2 策略日报</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --paper: rgba(255,255,255,0.76);
      --ink: #1d2733;
      --muted: #5d6c7b;
      --line: rgba(29,39,51,0.12);
      --accent: #d34f2a;
      --accent-soft: #f6d6bf;
      --shadow: 0 22px 50px rgba(67, 43, 25, 0.12);
      --radius: 24px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(211,79,42,0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(43,84,126,0.12), transparent 24%),
        linear-gradient(180deg, #f9f5ef 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 40px 0 72px;
    }}
    .masthead {{
      padding: 28px 30px;
      border: 1px solid var(--line);
      border-radius: 32px;
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,248,240,0.72));
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .masthead p {{
      margin: 0 0 10px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 12px;
    }}
    .masthead h1 {{
      margin: 0;
      font-size: clamp(32px, 6vw, 58px);
      line-height: 0.96;
    }}
    .masthead .sub {{
      margin-top: 14px;
      max-width: 760px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
      text-transform: none;
      letter-spacing: 0;
    }}
    .day-list {{
      display: grid;
      gap: 22px;
      margin-top: 28px;
    }}
    .day-card {{
      padding: 24px;
      border-radius: var(--radius);
      border: 1px solid var(--line);
      background: var(--paper);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .day-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
    }}
    .day-head h2 {{
      margin: 0;
      font-size: 28px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 8px 14px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .hero-tile {{
      padding: 18px;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.75);
    }}
    .hero-tile.accent {{
      background: linear-gradient(160deg, #d34f2a, #ef8354);
      color: white;
      border-color: transparent;
    }}
    .tile-label {{
      margin: 0 0 8px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      opacity: 0.75;
    }}
    .hero-tile h3 {{
      margin: 0;
      font-size: 26px;
    }}
    .tile-meta {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .hero-tile.accent .tile-meta,
    .hero-tile.accent .tile-label {{
      color: rgba(255,255,255,0.8);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    @media (max-width: 800px) {{
      .hero-grid {{
        grid-template-columns: 1fr;
      }}
      .day-head {{
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="masthead">
      <p>Local Strategy Report</p>
      <h1>涨停后两次回调<br>更高低点策略</h1>
      <p class="sub">从 {payload.get("date_from","")} 到 {payload.get("date_to","")}，每天单独跑这条策略，只保留当日策略池里排序最高的那一只。网页只展示这条策略自己的结果，不混别的策略权重。</p>
    </section>
    <section class="day-list">
      {cards_html}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    date_from = pd.Timestamp(args.date_from)
    date_to = pd.Timestamp(args.date_to) if args.date_to else pd.Timestamp.today().normalize()

    days: list[dict] = []
    for ts in pd.date_range(date_from, date_to, freq="D"):
        screen_date = ts.strftime("%Y%m%d")
        result = run_limitup_l1l2_screen(
            end_date=screen_date,
            history_bars=args.history_bars,
            moneyflow_lookback_days=args.moneyflow_lookback_days,
            recent_buy_window=args.recent_buy_window,
            min_score=args.min_score,
            cutoff_hour=args.cutoff_hour,
        )
        summary = result["summary"]
        strategy_candidates = result["strategy_candidates"].copy()
        best_pick_candidate = result["best_pick_candidate"].copy()
        top_cols = [c for c in display_columns() if c in strategy_candidates.columns]
        top_candidates = strategy_candidates[top_cols].head(args.show_top).to_dict(orient="records") if not strategy_candidates.empty else []
        best_pick = best_pick_candidate[top_cols].head(1).to_dict(orient="records")[0] if not best_pick_candidate.empty else {}
        days.append(
            {
                "display_date": ts.strftime("%Y-%m-%d"),
                "screen_end_date": summary.get("screen_end_date"),
                "strategy_candidates": summary.get("strategy_candidates", 0),
                "best_pick": json_safe(best_pick),
                "top_candidates": json_safe(top_candidates),
                "summary": json_safe(summary),
            }
        )

    payload = {
        "strategy_name": "涨停L1L2",
        "date_from": date_from.strftime("%Y-%m-%d"),
        "date_to": date_to.strftime("%Y-%m-%d"),
        "days": days,
    }

    repo_root = Path(__file__).resolve().parent.parent
    report_dir = repo_root / "output" / "web" / "limitup_l1l2_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "index.html").write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {report_dir / 'index.html'}")


if __name__ == "__main__":
    main()

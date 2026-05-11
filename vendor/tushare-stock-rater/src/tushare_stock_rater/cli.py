from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import analyze_stock
from .client import TushareClient, TushareClientError
from .rebound_picker import backtest_rebound_strategy, pick_rebound_stocks
from .utils import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tushare-stock-rater",
        description="Tushare-only single-stock buy rating system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="分析单只股票并生成报告")
    analyze.add_argument("stock", help="股票代码或名称，例如 600519.SH / 600519 / 贵州茅台")
    analyze.add_argument("--as-of", default="", help="评估日期，格式 YYYYMMDD。默认取最近完整交易日。")
    analyze.add_argument("--lookback", type=int, default=252, help="行情回看交易日数量，默认 252。")
    analyze.add_argument("--out", default=str(PROJECT_ROOT / "reports"), help="报告输出目录。")
    analyze.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "scoring_weights.yaml"), help="评分配置 YAML。")
    analyze.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "cache"), help="Tushare 缓存目录。")
    analyze.add_argument("--no-cache", action="store_true", help="禁用本地 CSV 缓存。")
    analyze.add_argument("--retries", type=int, default=2, help="接口失败重试次数。")
    analyze.add_argument("--sleep-sec", type=float, default=0.2, help="接口调用后等待秒数，降低频控风险。")
    analyze.add_argument("--cutoff-hour", type=int, default=20, help="未指定日期时，早于该小时默认用前一自然日。")

    doctor = subparsers.add_parser("doctor", help="检查 Tushare token 与基础接口")
    doctor.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "cache"), help="Tushare 缓存目录。")
    doctor.add_argument("--no-cache", action="store_true", help="禁用本地 CSV 缓存。")

    pick = subparsers.add_parser("pick-rebound", help="尾盘低吸筛选次日可能反抽的候选股")
    pick.add_argument("--as-of", default="", help="回放日期，格式 YYYYMMDD。留空时默认今天，优先走实时接口。")
    pick.add_argument("--top", type=int, default=5, help="输出候选数量，默认 5。")
    pick.add_argument("--historical", action="store_true", help="强制只用历史日线回放，不调用实时接口。")
    pick.add_argument("--out", default=str(PROJECT_ROOT / "reports"), help="结果输出目录。")
    pick.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "scoring_weights.yaml"), help="配置 YAML。")
    pick.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "cache"), help="Tushare 缓存目录。")
    pick.add_argument("--no-cache", action="store_true", help="禁用本地 CSV 缓存。")
    pick.add_argument("--retries", type=int, default=2, help="接口失败重试次数。")
    pick.add_argument("--sleep-sec", type=float, default=0.2, help="接口调用后等待秒数，降低频控风险。")

    backtest = subparsers.add_parser("backtest-rebound", help="回测尾盘低吸策略")
    backtest.add_argument("--start", required=True, help="回测开始日期，格式 YYYYMMDD。")
    backtest.add_argument("--end", required=True, help="回测结束日期，格式 YYYYMMDD。")
    backtest.add_argument("--top", type=int, default=1, help="每天取前 N 只，默认 1。")
    backtest.add_argument("--out", default=str(PROJECT_ROOT / "reports"), help="结果输出目录。")
    backtest.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "scoring_weights.yaml"), help="配置 YAML。")
    backtest.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "cache"), help="Tushare 缓存目录。")
    backtest.add_argument("--no-cache", action="store_true", help="禁用本地 CSV 缓存。")
    backtest.add_argument("--retries", type=int, default=2, help="接口失败重试次数。")
    backtest.add_argument("--sleep-sec", type=float, default=0.2, help="接口调用后等待秒数，降低频控风险。")
    return parser


def run_analyze(args: argparse.Namespace) -> int:
    result, _features, _bundle, paths = analyze_stock(
        query=args.stock,
        as_of_date=args.as_of,
        lookback_days=args.lookback,
        out_dir=Path(args.out),
        config_path=Path(args.config),
        cache_dir=Path(args.cache_dir),
        no_cache=args.no_cache,
        retries=args.retries,
        sleep_sec=args.sleep_sec,
        cutoff_hour=args.cutoff_hour,
    )
    print(f"{result.stock.name}({result.stock.ts_code}) {result.as_of_date}")
    print(f"综合评分: {result.total_score:.2f}/100  评级: {result.rating}  结论: {result.verdict}")
    print(f"置信度: {result.confidence_score:.2f}")
    if result.hard_risks or result.severe_risks:
        print("风险限制: " + "；".join(result.severe_risks + result.hard_risks))
    print(f"报告目录: {paths.output_dir}")
    print(f"Markdown: {paths.report_md}")
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    client = TushareClient(cache_dir=Path(args.cache_dir), use_cache=not args.no_cache)
    metas = client.smoke_test()
    for meta in metas:
        print(f"{meta.endpoint}: rows={meta.row_count}, cached={meta.cached}, error={meta.error or 'OK'}")
    print("Tushare 基础接口检查通过。")
    return 0


def run_pick_rebound(args: argparse.Namespace) -> int:
    candidates, paths, warnings = pick_rebound_stocks(
        as_of_date=args.as_of,
        top_n=args.top,
        out_dir=Path(args.out),
        config_path=Path(args.config),
        cache_dir=Path(args.cache_dir),
        no_cache=args.no_cache,
        retries=args.retries,
        sleep_sec=args.sleep_sec,
        historical=args.historical,
    )
    if candidates:
        head = candidates[0]
        print(f"首选: {head.name}({head.ts_code})  得分 {head.score:.2f}")
        print(
            f"跌幅 {head.pct_chg:.2f}%  行业超跌 {head.excess_drop_pct:.2f}%  "
            f"观察区间 {head.entry_low}~{head.entry_high}  止损 {head.stop_loss}"
        )
        print("入选理由: " + "；".join(head.reasons[:3]))
        if head.warnings:
            print("个股风险: " + "；".join(head.warnings))
    else:
        print("本次未筛到符合条件的候选。")
    print(f"结果目录: {paths.output_dir}")
    print(f"Markdown: {paths.summary_md}")
    if warnings:
        print("接口提示: " + "；".join(warnings[:4]))
    return 0


def run_backtest_rebound(args: argparse.Namespace) -> int:
    summary, paths = backtest_rebound_strategy(
        start_date=args.start,
        end_date=args.end,
        top_n=args.top,
        out_dir=Path(args.out),
        config_path=Path(args.config),
        cache_dir=Path(args.cache_dir),
        no_cache=args.no_cache,
        retries=args.retries,
        sleep_sec=args.sleep_sec,
    )
    print(f"回测区间: {summary.start_date} ~ {summary.end_date}")
    print(f"有效交易: {summary.selection_count}  空仓日: {summary.skipped_days}")
    print(
        f"次日开盘均值 {summary.avg_open_return_pct}%  "
        f"次日最高均值 {summary.avg_high_return_pct}%  "
        f"次日收盘均值 {summary.avg_close_return_pct}%  "
        f"策略实际均值 {summary.avg_exit_return_pct}%"
    )
    print(
        f"开盘胜率 {summary.win_rate_open_pct}%  "
        f"收盘胜率 {summary.win_rate_close_pct}%  "
        f"策略实际胜率 {summary.win_rate_exit_pct}%  "
        f"最好 {summary.best_trade_pct}%  最差 {summary.worst_trade_pct}%"
    )
    print(f"结果目录: {paths.output_dir}")
    print(f"Markdown: {paths.summary_md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            return run_analyze(args)
        if args.command == "doctor":
            return run_doctor(args)
        if args.command == "pick-rebound":
            return run_pick_rebound(args)
        if args.command == "backtest-rebound":
            return run_backtest_rebound(args)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except (TushareClientError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

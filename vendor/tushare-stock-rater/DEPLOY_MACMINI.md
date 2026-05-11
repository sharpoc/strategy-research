# Mac mini 部署清单

这个包用于在新的 Mac mini 上部署 `tushare-stock-rater` 项目，并运行尾盘反抽筛选。

## 1. 解压

```bash
cd ~
mkdir -p ai_dev
cd ai_dev
unzip 选股_macmini部署包_20260428.zip
cd 选股
```

## 2. 安装 Python 依赖

建议先确认 Python 版本：

```bash
python3 --version
```

安装项目：

```bash
python3 -m pip install -e .
```

## 3. 配置 Tushare token

不要把真实 token 写进源码。

临时方式：

```bash
export TUSHARE_TOKEN="你的token"
```

如果想长期使用，可以写进你自己的 shell 配置，例如 `~/.zshrc`：

```bash
export TUSHARE_TOKEN="你的token"
```

写完后执行：

```bash
source ~/.zshrc
```

## 4. 先做一次健康检查

```bash
python3 -m tushare_stock_rater doctor
```

## 5. 跑尾盘反抽筛选

今天直接跑：

```bash
python3 -m tushare_stock_rater pick-rebound --top 5
```

指定日期回放：

```bash
python3 -m tushare_stock_rater pick-rebound --as-of 20260428 --top 5
```

只用历史数据，不走实时接口：

```bash
python3 -m tushare_stock_rater pick-rebound --as-of 20260428 --historical --top 5
```

## 6. 结果位置

默认会写到：

```text
reports/rebound_YYYYMMDD_live/
reports/rebound_YYYYMMDD_historical/
```

主要看这几个文件：

```text
summary.md
summary.json
candidates.csv
```

## 7. 常用命令

单股分析：

```bash
python3 -m tushare_stock_rater analyze 600519 --as-of 20260428 --lookback 252 --out reports
```

回测尾盘反抽：

```bash
python3 -m tushare_stock_rater backtest-rebound --start 20251022 --end 20260422 --top 1
```

## 8. 常见问题

### 提示 `Missing TUSHARE_TOKEN`

说明当前 shell 没拿到环境变量，重新执行：

```bash
export TUSHARE_TOKEN="你的token"
```

### 提示某些接口没权限

这个项目会尽量自动降级，`pick-rebound` 通常还能继续跑。

### 想看命令帮助

```bash
python3 -m tushare_stock_rater --help
python3 -m tushare_stock_rater pick-rebound --help
```

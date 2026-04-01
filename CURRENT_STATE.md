# 当前状态

最后更新：`2026-04-01`

## 仓库角色

- 这是研究与执行仓库。
- 当前 GitHub 仓库：
  - `https://github.com/sharpoc/strategy-research`
- 本地目录：
  - `/Users/lvxue/work/量化`

## 当前职责

- 策略研究
- 回测
- 参数优化
- 单天 runner
- 缺天补数执行层
- 未来 `Mac mini` 定时执行链路

## 当前重要策略

- `星曜增持臻选`
  - 当前最接近线上主策略
- `核心高管连增臻选`
  - 当前只在本地研究
  - 最近结论是：
    - `stage1` 候选池有价值
    - `final` 更适合作为轻确认 + 去重
    - 最新 6 个月轻确认对拍里：
      - `final` 真信号 `4` 次
      - `5日均值 +9.61%`
      - `10日均值 +5.3594%`
    - 当前最值得继续研究的是：
      - 为什么 `10日` 仍弱于 `stage1`
      - 怎样继续优化轻确认 `final`
  - Mac mini 交接文档：
    - [/Users/lvxue/work/量化/docs/CORE_MANAGEMENT_LIGHT_FINAL_HANDOFF.md](/Users/lvxue/work/量化/docs/CORE_MANAGEMENT_LIGHT_FINAL_HANDOFF.md)

## 当前执行层结论

- 单天执行比整段批跑更稳
- 缺天补数应优先：
  - 单天串行
  - 失败只补缺口
  - 已完成不重复深挖

## 当前与服务仓库关系

- 服务仓库：
  - `https://github.com/sharpoc/strategy-lab`
- 当前服务仓库通过环境变量调用本仓库中的 runner
- 长期目标是：
  - `Mac mini` 跑本仓库任务
  - 再把结果推送到服务仓库对应的线上系统

## 换电脑时最重要的事

新电脑优先保证：

1. GitHub SSH 能正常拉取本仓库
2. Python 环境与 PostgreSQL 可用
3. `.env` 和路径配置按新机器重配
4. 如需断点续跑，旧电脑的 `output/` 要复制过来

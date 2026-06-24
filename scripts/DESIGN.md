# 自动 Alpha 发现系统 - 设计文档

> 日期: 2026-06-24
> 状态: 设计确认完成，进入实现

---

## 1. 系统总览

混合架构：脚本驱动的广度（参数网格探索）+ Agent 驱动的深度（论文/研报灵感提取）。
通过"批次 fuel-mine 循环"实现自动化运行，lessons.json 在挖掘统计和下一批论文提取之间形成反馈闭环。

```
论文/研报 → papers_registry.json (追踪)
  → Agent 逐篇提取 → templates/*.json (结构化模板)
    → mining_loop.py (参数展开 + 模拟 + 过滤)
      → lessons.json (经验回流)
        → 下一批论文提取时参考 lessons.json
```

## 2. 核心设计决策

### 2.1 架构：混合（脚本广度 + Agent 深度）
- 广度：纯 Python 脚本，参数网格展开，无人值守
- 深度：Agent CLI 子进程调用，逐篇读论文，产出模板 JSON

### 2.2 组织：广度是引擎，深度是燃料
- 深度产出结构化模板 JSON（骨架 + 字段对 + 参数范围 + 假设）
- 广度展开模板的参数网格进行探索
- 模板耗尽时触发深度阶段读取下一篇论文

### 2.3 循环模式：批次 fuel-mine 循环
```
mining_loop.py
  ├── 初始化：加载 templates/ + lessons.json + papers_registry.json
  ├── while True:
  │     ├─ [广度] generate_candidates.py 展开参数 → 批量模拟 → 质量过滤(方案B)
  │     │         → SUBMIT/OBSERVE/DISCARD → 更新 lessons.json
  │     ├─ [检查] 连续 3 轮无 ACTIVE? → 终止
  │     └─ [深度] 候选池空?
  │           ├─ 有 unread 论文 → fuel_one_paper() → 回到广度
  │           └─ 无 unread 论文 → 终止（燃料耗尽）
  └── 输出 mining_report.json
```

### 2.4 终止条件
- **燃料耗尽**：候选池空 AND 无 unread 论文 → 立即终止
- **效果枯竭**：连续 3 轮无新增 ACTIVE → 终止

### 2.5 深度输出：结构化模板 JSON
```json
{
  "template_id": "profitability_trend",
  "description": "盈利趋势因子：用基本面数据的趋势变化预测横截面收益",
  "skeleton": "group_rank(ts_rank({numerator} / {denominator}, {window}), {group})",
  "field_pairs": [
    {"numerator": "operating_income", "denominator": "equity"},
    {"numerator": "net_income", "denominator": "equity"},
    {"numerator": "free_cash_flow_per_share", "denominator": "close"}
  ],
  "param_ranges": {
    "window": [63, 126, 252],
    "group": ["subindustry", "industry", "sector"]
  },
  "hypothesis": "盈利持续改善的公司未来收益更高，group_rank 控制行业效应",
  "source": "src_001",
  "created": "2026-06-24"
}
```

### 2.6 字段校验：两层
1. **Agent 自查**：提取模板时对照 `references/wq_usa_top3000_delay1_data_fields.json` (4,367 字段)
2. **脚本兜底**：`generate_candidates.py` 带模糊匹配，无效字段自动跳过或建议替代

### 2.7 质量过滤：方案 B（自适应阈值 + 相关性检查）
```
SUBMIT   → Sharpe ≥ 1.5 AND Fitness ≥ 1.0 AND Turnover < 0.7
           AND 与现有 ACTIVE 相关性 < 0.7
OBSERVE  → Sharpe ≥ 1.0 (未达提交线但有潜力，留作参数调优种子)
           OR Sharpe ≥ 1.5 但与现有 ACTIVE 相关性 ≥ 0.7 (冗余降级)
           OR Sharpe 1.25-1.5 但相关性 < 0.3 (低相关升级为 SUBMIT)
DISCARD  → 其余
```

### 2.8 重试策略
- 模拟后轮询：20s 间隔，5min 最大值，1 次重试
- 自适应并发：从 4 开始，根据 429 响应扩展/收缩

### 2.9 经验存储：模式级 lessons.json
```json
{
  "patterns": {
    "profitability_trend": {
      "description": "...",
      "tested": 12, "passed": 4, "pass_rate": 0.33,
      "avg_sharpe": 1.52, "avg_fitness": 1.08,
      "best": {"alpha_id": "...", "sharpe": 2.01, "expr": "..."},
      "failure_modes": {"LOW_FITNESS": 5, "LOW_SHARPE": 3},
      "action": "expand",
      "notes": "..."
    }
  },
  "param_insights": {
    "window": {"63": {"avg_sharpe": 0.9, "verdict": "deprioritize"}, "126": {...}, "252": {...}},
    "neutralization": {...},
    "decay": {...}
  },
  "last_updated": "..."
}
```

### 2.10 素材管理：papers_registry.json
支持多格式输入：
- `pdf`：本地文件，mira_local_read_file 或 pdf skill 读取
- `markdown`：本地文件，直接读取
- `feishu_doc`：飞书文档 URL，lark-doc skill 读取
- `web`：公开网页 URL，web_builtin_fetch → fallback 浏览器

```json
{
  "sources": {
    "src_001": {
      "type": "pdf|markdown|feishu_doc|web",
      "locator": "papers/xxx.pdf | https://...",
      "hash": "sha256:...",
      "title": "...",
      "status": "unread|consumed|extraction_failed|timeout",
      "read_date": null,
      "extracted_templates": [],
      "extraction_round": null
    }
  },
  "stats": {"total": 0, "consumed": 0, "remaining": 0}
}
```

### 2.11 深度阶段稳定性
- Agent CLI 子进程调用，5 分钟超时
- 超时 → 标记 `timeout`，跳过，试下一篇
- 产出 0 模板 → 标记 `extraction_failed`，跳过
- 连续 3 篇失败 → 认为深度不可用，仅消耗剩余模板后正常退出

### 2.12 重构策略：方案 A
- 新建 `generate_candidates.py`、`mining_loop.py`
- 改造 `evolve_skill.py`：复用 API 函数，输出改为 lessons.json
- 归档 `run_alpha101*.py` 到 `scripts/archive/`
- 简化 `submit_batch.py`：只接收已过滤候选列表

## 3. 目录结构

```
world_quant/
  SKILL.md                          ← 技能文档（已存在，持续演进）
  alpha_db.json                     ← alpha 快照数据库（已存在，43 个 alpha，6 个 ACTIVE）
  alpha101_results.json             ← v1 实验结果（已存在，归档参考）
  alpha101_v2_results.json          ← v2 实验结果（已存在，归档参考）
  credential.txt                    ← BRAIN API 凭证
  
  papers/                           ← 原始论文/研报
  papers_registry.json              ← 素材追踪注册表
  
  templates/                        ← 结构化模板 JSON
    profitability_trend.json
    analyst_estimate.json
    valuation_multiple.json
    
  lessons.json                      ← 模式级经验教训
  
  references/                       ← 数据字段参考（已存在）
    wq_usa_top3000_delay1_data_fields.json
    wq_usa_top3000_delay1_data_fields_summary.json
    
  scripts/
    mining_loop.py                  ← 主循环编排
    generate_candidates.py          ← 广度引擎
    evolve_skill.py                 ← 改造：经验同步器
    submit_batch.py                 ← 简化：批量提交
    archive/
      run_alpha101.py
      run_alpha101_v2.py
    DESIGN.md                       ← 本文件
    mining_report.json              ← 挖掘报告（运行时生成）
    depth_request.json              ← 深度阶段请求（运行时生成）
```

## 4. 实现计划

| 步骤 | 任务 | 依赖 |
|------|------|------|
| 1 | 创建目录结构 + papers_registry.json | 无 |
| 2 | 从现有 ACTIVE alpha 提取初始模板 | 步骤 1 |
| 3 | 初始化 lessons.json（从 alpha_db + alpha101_results） | 无 |
| 4 | 实现 generate_candidates.py | 步骤 2 |
| 5 | 改造 evolve_skill.py（复用函数 + lessons 写入） | 步骤 3 |
| 6 | 实现 mining_loop.py | 步骤 4, 5 |
| 7 | 归档旧脚本 | 无 |
| 8 | 端到端测试（dry run） | 步骤 6 |

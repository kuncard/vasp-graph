# VASP Knowledge Graph

将 VASP Wiki 转化为结构化知识库，全链路面向 AI Agent，同时提供人工可读的静态产物与交互图谱。

## 产物

| 文件 | 用途 |
|------|------|
| kb.md | Agent 主力，单文件 Markdown，933 页面 |
| kb.json | Agent 备选，单文件结构化 JSON |
| graph.html | Agent 调用，vis.js 图谱，内嵌全量数据，搜索即用 |
| KDG API | Agent 主力运行时，支持搜索/关联/图遍历/语义检索 |

## 架构

```
VASP Wiki HTML 镜像
    │
    ▼  crawl_graph.py           优先级 BFS 爬取
    │
    ▼  nodes.json + edges.json
    │
    ├── reclassify_with_agent.py    LLM 修正分类
    ├── enrich_nodes.py             规则 + LLM 信息提取
    │
    ▼  nodes_enriched.json
    │
    ├── generate_markdown.py     kb.md + kb/
    ├── generate_json_kb.py      kb.json
    ├── graph2d.py               graph.html
    └── lookup.py                按需补漏
```

## 分类体系

`classify_page()` 用 13 级规则链逐级判定页面类型，标签优先 → 名字推断 → 正文兜底。~180 页规则拿不准的交给 LLM 重判。

| type | subtype | 含义 | 数量 |
|------|---------|------|------|
| capability | domain | 分类页、领域理论 | 114 |
| capability | parameter | INCAR 标签 | 510 |
| procedure | tutorial | 教程、HowTo | 285 |
| heuristic | best_practice | 操作建议、讨论 | 42 |
| constraint | pitfall | 已知问题、限制 | 6 |
| capability | generic | 重定向、stub | 5 |

## 信息提取

**元数据** — id, title, type, subtype, category, tags（规则）

**结构化摘要** — definition（LLM，全部页面）/ quick_facts（规则，参数页）/ options（规则，参数页）/ warnings（规则，参数页）/ tutorial_summary（LLM，教程页）

**正文** — HTML 转 Markdown，MathML 公式保护提取，LaTeX 独立清洗，wikilink 保留，导航垃圾切除，内容去重。

提取策略：规则做 95%，LLM 补 5%。504/510 参数规则自动完成，6 个非标准格式 LLM 补漏。

## 导入 know-do-graph

生成的知识库可导入 [know-do-graph](https://github.com/theAfish/know-do-graph) 数据库，提供 API + Web 界面：

```bash
python import_to_kdg.py data/enriched.json data/edges.json --db vasp_graph.db
know-do-graph serve --db vasp_graph.db
```

## 使用方式

### 安装

```bash
cd vasp-graph
pip install -r requirements.txt
```

### 环境变量

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com
```

wiki_dir 指向 HTTrack 镜像目录，如 `vasp/www.vasp.at/wiki/index.php/`。

### 流水线

```bash
python pipeline.py <wiki_dir> --api-key <YOUR_KEY>
python pipeline.py <wiki_dir> --no-reclassify --no-tutorials  # 跳过 LLM
```

### 分步

```bash
python crawl_graph.py <wiki_dir> --auto-seeds 5 -o data/nodes_raw --no-enrich
python reclassify_with_agent.py data/nodes_raw_nodes.json --only best_practice,generic,domain -o data/reclassified.json
python enrich_nodes.py data/reclassified.json -o data/enriched.json
python generate_markdown.py data/enriched.json data/nodes_raw_edges.json -o kb/ --single-file
python graph2d.py data/enriched.json data/nodes_raw_edges.json --max 0 -o graph.html
```

### 测试

```bash
pytest tests/ -v
```

## 图谱功能

- 933 节点力导向图，按类型着色，默认显示结构边
- 搜索防抖 300ms，id/title 精确匹配，自动展开邻居子图
- 匹配节点放大加粗，详情面板显示结构化内容
- 点 All Edges 切换全量 wikilink，双击展开邻居
- vis.js 内嵌，无需网络

## 文件结构

```
vasp-graph/
├── crawl_graph.py
├── wiki_parser.py
├── pipeline.py
├── reclassify_with_agent.py
├── enrich_nodes.py
├── generate_markdown.py
├── generate_json_kb.py
├── graph2d.py
├── lookup.py
├── parse_wiki.py
├── vis_bundle.js
├── requirements.txt
├── pyproject.toml
├── .github/workflows/test.yml
└── tests/
    ├── test_classify.py
    ├── test_clean.py
    └── test_dedup.py
```

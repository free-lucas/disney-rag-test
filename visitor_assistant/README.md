# 迪士尼游客 RAG 助手

面向游客的迪士尼知识库问答应用。回答只使用知识库中的公开内容，必须带来源；内部流程、员工规则和内部联系方式不会进入游客检索结果。

## 运行环境

- Python 3.12
- Windows 上可选安装 Word/PowerPoint COM，用于转换少量旧版 OLE 文档
- DeepSeek API：回答生成、图片理解和查询改写，模型为 `deepseek-flash`
- DashScope API：文本 embedding（DeepSeek 不提供 embedding 接口）

## 安装

在仓库中进入本目录：

```powershell
cd CASE-迪士尼RAG助手/visitor_assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

复制配置模板并填写真实密钥：

```powershell
Copy-Item .env.example .env
```

`.env` 中的 `DEEPSEEK_API_KEY` 和 `DASHSCOPE_API_KEY` 必须填写。不要把 `.env` 提交到仓库。

## 配置检查

```powershell
python config.py
```

检查通过时会输出模型名、密钥是否存在、知识库路径和索引路径，但不会打印密钥值。缺少密钥时会明确提示缺失的环境变量名。

## 知识库盘点与格式识别

盘点完整知识库并生成 JSON 报告（不需要 API key）：

```powershell
python -m ingest.discover --root ../迪士尼RAG知识库（完整） --output data/reports/discover_report.json
```

报告的 `summary` 包含总文件数、可索引主来源数、重复文件数、失败文件数和格式分布；`records` 记录每个文件的相对路径、扩展名、大小、SHA-256、检测格式和 OLE 子类型；`duplicates` 记录重复文件及其主来源；`errors` 记录失败文件、错误类型和错误信息。

格式按文件签名识别，不信任扩展名：

- `docx` / `pptx`：OOXML 容器，包括伪装成 `.doc` / `.ppt` 的文件
- `ole`：旧版 Office 复合文档，`ole_kind` 区分 `word`、`powerpoint`、`excel`
- `pdf`：PDF 文本层或扫描页
- `jpeg` / `image`：独立图片
- `corrupted_archive`：损坏或 0 条目的 zip 容器

常见错误类型为 `bad_zip`、`empty_zip`、`unsupported_zip`、`unknown_format`、`read_error`、`stat_error` 和 `hash_error`。单个文件失败只记录在报告中，不会中断整批盘点。

## 文本提取与结构切片

- OOXML Word/PowerPoint 使用 `python-docx` 和 `python-pptx` 直接解析，扩展名写成 `.doc` / `.ppt` 不影响识别。
- 真正的 OLE 文档通过 Windows Word/PowerPoint COM 转换到临时目录后再解析；转换在独立子进程中执行，COM 崩溃或超时不会中断整批建库。未安装 `pywin32` 时记录 `com_unavailable`。
- PDF 使用 PyMuPDF 提取文本层并保留页码；单页文本少于 20 个字符时按扫描页渲染为 PNG，交给图片描述流程。
- Word 表格、PowerPoint 表格和正文中的 Markdown 表格统一保留表头；表格作为一个整体进入 chunk，即使超过目标长度也不拆分。
- 目录名中的分类、文件名、标题层级、页码和幻灯片编号都会写入 chunk 元数据。

切片参数默认值如下，可通过环境变量覆盖：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `CHUNK_TARGET_CHARS` | 520 | 目标长度，达到目标后遇到新标题即可换块 |
| `CHUNK_MIN_CHARS` | 400 | 小于该长度的章节会继续向后合并 |
| `CHUNK_MAX_CHARS` | 600 | 普通文本块上限；表格不受此限制 |
| `CHUNK_OVERLAP_CHARS` | 100 | 相邻文本块的重叠字符数 |

短文档、单张幻灯片和 PDF 单页可能天然小于目标长度；为保证来源页码和幻灯片编号准确，这些边界不会被强行合并。公开 chunk 在切片前执行脱敏，脱敏来源文件和类型会写入建库报告。

## 构建索引

配置好两个 API key 后，从本目录执行：

```powershell
python build_index.py
```

常用调试参数：

```powershell
python build_index.py --root ../迪士尼RAG知识库（完整） --index-dir data/index --report data/reports/build_report.json
python build_index.py --no-com --force-rebuild
```

建库产物：

- `data/index/faiss.index`：对归一化向量使用 `IndexFlatIP`，内积等价于余弦相似度。
- `data/index/chunks.jsonl`：按向量位置保存 chunk 内容和全部元数据，包含 `audience`。
- `data/index/bm25.pkl`：jieba 分词后的 BM25 索引。
- `data/index/manifest.json`：embedding 模型、维度、构建版本、文件 hash 和 pipeline 签名。
- `data/reports/build_report.json`：成功、失败、重复、跳过、图片过滤、caption 错误、脱敏记录和 public/internal chunk 数。

Embedding 调用 DashScope：优先 `text-embedding-v4`，模型不可用时自动回退 `text-embedding-v1`，实际使用的模型和维度会写入 manifest。embedding 缓存键为 `(content_hash, model)`，因此同一内容重复建库不会重复调用 API，切换模型会自然产生未命中。文档级缓存还会综合内容策略、切片参数、图片阈值和视觉模型；未变化文件直接复用 chunk，变化文件重新抽取，已删除文件记录在 `removed_files`。单个文件解析失败不会中断整库，但会以明确的错误类型写入报告。

## 检索与查询理解

游客问题先经过一次 `deepseek-flash` 查询理解，返回补全指代后的独立问题、去重后的子查询和是否依赖历史。没有历史或问题本身完整时不会把旧话题拼入；模型返回非法 JSON 或接口失败时回退到原问题，复合问句仍会按问号做本地拆分。

检索参数默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `RETRIEVAL_TOP_K` | 6 | 最终交给回答阶段的证据数 |
| `DENSE_TOP_K` | 20 | 向量通道候选数 |
| `BM25_TOP_K` | 20 | BM25 通道候选数 |
| `RRF_K` | 60 | reciprocal rank fusion 的平滑常数 |
| `DENSE_MIN_SCORE` | 0.25 | 向量通道最低余弦相似度；低于该值不进入融合 |
| `RERANK_ENABLED` | false | 是否启用可选精排 |

流程如下：

- 先按 `audience=public` 构造公共 FAISS 和 BM25 子索引，内部 chunk 不进入候选集；返回前再断言一次，任何 internal chunk 混入都会直接报错。
- 向量通道使用归一化后的余弦相似度，并丢弃低于 `DENSE_MIN_SCORE` 的结果；BM25 通道使用 jieba 分词，适合门票名、项目名和价格等精确词。BM25Plus 会给所有文档一个正基线分，因此只保留与查询词项有实际重合的候选。
- 只有内部资料才能回答的问题，在公开内容中通常得不到有效证据，此时返回空证据而不是任意最近邻，由回答阶段按拒答规则处理。
- 两路结果用 RRF 融合，并按 `content_hash` 去重；结果保留融合分数、向量分数、BM25 分数、来源文件和章节位置。
- rerank 默认关闭。开启后会重排候选，但 `score` 仍保留原始融合分数，精排分数写入 `rerank_score`；精排失败时回退融合排序。
- 查询或证据与图片 caption 有足够词项重合时，会附带相关 `image_caption` 结果；普通文本查询不会无条件附带图片。

## 回答生成与引用规则

回答由 `deepseek-flash` 生成，系统提示词和证据模板位于 `generation/prompt.py`，生成与后处理位于 `generation/answer.py`。规则如下：

- 只允许使用编号后的公开证据；每条证据包含来源文件、章节路径、PDF 页码或幻灯片编号。
- 模型第一行必须输出 `ANSWER` 或 `REFUSE`。`REFUSE` 时不输出正文，应用改为标准拒答文案，并引导游客查看官方网站或官方 App。
- 事实性回答至少带一个引用。模型用 `[1]`、`[2]` 标注证据；后处理只保留被引用证据对应的来源和图片，未标注时回退到全部贡献来源。引用去重后保留文件名和章节、页码或幻灯片。
- 证据只覆盖部分问题时，先回答能确认的部分，再明确指出知识库未覆盖的缺口，不把缺失信息补成事实。
- 问题或回答涉及票价、门票、开闭园时间、演出时刻、预约、退改等易变信息时，追加“以官方渠道当日信息为准”的提示；酒店设施等稳定内容不追加。
- 生成后执行保密断言。回答中出现员工票、工号、内部流程、升级路径、内部电话、分机、责任人或员工联系方式时，整段回答替换为安全拒答。
- 只有被引用证据中的图片会随回答返回 `image_path` 和 `caption`；未引用的图片不附带，拒答时也不附带图片。
- 多轮对话会携带最近 6 条消息帮助理解追问；本轮是独立新话题时不沿用旧上下文。查询改写和回答生成都使用 `deepseek-flash`。
- 生成请求支持超时和指数退避重试；模型返回空内容或输出因长度限制被截断时，会自动扩大输出预算并重试。流式接口先缓冲第一行判断 `ANSWER` / `REFUSE`，不会把控制标记展示给游客；流式没有返回正文时回退到非流式生成。模型失败时返回带 `error` 的安全拒答，不中断会话。

## 启动游客聊天界面

完成建库并配置好密钥后，从本目录启动：

```powershell
streamlit run app.py
```

界面能力：

- 聊天输入支持流式回答，并保留多轮会话；回答区下方可展开“来源”，显示文件名及章节、页码或幻灯片。
- 命中的图片证据会随回答展示；图片文件缺失时显示“图片暂不可用”提示，不影响文本回答。
- 左侧“新对话”会清空消息和历史，下一轮不会复用旧上下文。
- 侧边栏只显示配置中的模型名和索引目录，不显示密钥；模型、路径、TopK 和密钥全部来自环境变量或 `.env`。
- 缺少 API 密钥、索引缺失、embedding 失败或生成失败时显示可理解的中文提示。失败不会清空已有消息，也不会终止会话。

## 质量评测

固定金标准问题位于 `eval/golden_questions.json`，覆盖门票、乐园项目、酒店、餐饮、应急指引、邮轮和域外问题。可回答题目记录期望来源文件，域外题目标记 `expect_refusal: true`。

建库完成后运行完整评测（检索加回答生成）：

```powershell
python evaluate.py --golden eval/golden_questions.json --output-dir data/reports
```

只评测检索、不消耗生成调用：

```powershell
python evaluate.py --no-generation --limit 5
```

常用参数：

- `--top-k`：评估 hit@k 和 MRR 的截断位置，默认 6。
- `--limit`：只运行前 N 题，便于调试。
- `--no-generation`：只计算检索指标，跳过 DeepSeek 回答生成。

每次运行会在 `data/reports/` 写出同名的 JSON 和 Markdown 报告。报告包含：

- 总体和分类的 hit@k、MRR、引用率、拒答正确率、内部泄露数、通过题数和错误数。
- 未命中题目的期望来源与实际 TopK，便于定位切片或检索参数问题。
- 单题失败记录；某一题的检索或生成异常不会中断其余题目。

对比两版报告：

```powershell
python evaluate.py --compare data/reports/eval_report_旧.json data/reports/eval_report_新.json
```

对比输出 hit@k、MRR、引用率、拒答正确率和内部泄露数的总体及分类差值，并在 `data/reports/eval_comparison.md` 写出 Markdown 结果。内部内容泄露按零容忍检查：回答中出现内部 chunk 片段或员工规则、内部电话等敏感表述即判为失败。

## 配置项

路径默认以本目录为根；知识库默认位于 `../迪士尼RAG知识库（完整）`。可通过环境变量覆盖：

- `DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`
- `EMBEDDING_MODEL`、`EMBEDDING_FALLBACK_MODEL`
- `DISNEY_KB_ROOT`、`RAG_DATA_DIR`、`RAG_INDEX_DIR`
- `RETRIEVAL_TOP_K`、`DENSE_TOP_K`、`BM25_TOP_K`、`RRF_K`、`DENSE_MIN_SCORE`
- `RERANK_ENABLED`
- `CHUNK_TARGET_CHARS`、`CHUNK_MIN_CHARS`、`CHUNK_MAX_CHARS`、`CHUNK_OVERLAP_CHARS`
- `IMAGE_MIN_WIDTH`、`IMAGE_MIN_HEIGHT`、`IMAGE_MIN_AREA`、`IMAGE_MAX_PER_DOCUMENT`
- `MODEL_TIMEOUT_SECONDS`、`MODEL_MAX_RETRIES`、`ANSWER_MAX_TOKENS`、`REWRITE_MAX_TOKENS`

知识库是静态资料，不是实时票价或排队数据源。涉及票价、开闭园、演出时刻和政策时，回答会提示以官方渠道当日信息为准。

## 图片抽取、过滤与描述

图片不是直接拿去做跨模态向量检索，而是先由 `deepseek-flash` 生成简体中文描述，再以 `image_caption` chunk 进入文本索引。这样图片和文本共享同一套引用、受众过滤和拒答规则，命中图片时仍会返回原始图片路径。

- 独立 JPEG、Word/PPT 内嵌图片、扫描版 PDF 页面都会被抽取；内嵌图片保留来源文档、页码或幻灯片编号以及邻近文本。
- 图片按 SHA-256 去重；宽、高或面积低于阈值的装饰性图片会被跳过，单个文档最多保留 `IMAGE_MAX_PER_DOCUMENT` 张，所有过滤原因写入建库报告。
- 描述请求使用图片数据 URL 调用 `deepseek-flash`，提示词要求只描述可见内容，并优先提取标题、价格、日期、地点、时间和表格表头。
- 描述缓存键由图片 hash、视觉模型名和 `CAPTION_PROMPT_VERSION` 共同组成。重复建库会直接复用缓存；更换模型或提示词版本会使旧缓存失效。
- 空描述、接口错误和重试耗尽都会记录到报告，不会生成空 caption chunk；原图元数据仍保留，便于人工复核。

成本控制主要依靠尺寸阈值、全局 hash 去重、单文档上限和描述缓存。扫描页、地图和价格表是最需要人工抽查的类型：OCR 或视觉描述可能漏字、错行或误读小字号价格，因此图片描述只能作为检索线索，事实性回答仍应结合同一来源的文本并提示以官方渠道为准。

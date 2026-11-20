# OCR 修订 · 实体标签迁移与行图训练清单工具

语料工程师修订 OCR 文本后，把原文上的实体标签迁移到修订文，供训练使用；
确认结果页还可上传切图 JSON，生成**行图训练清单**：判断哪些行图在所有
合法切法下都能配上唯一且非空的修订文字。
后端 FastAPI + SQLite，前端为单页网页，交付含 Docker Compose 一键启动。

## 快速启动

### Docker（推荐）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

SQLite 数据库落在名为 `appdata` 的卷里（容器内 `/data/app.db`），重启不丢。

### 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt     # 运行只需 requirements.txt
uvicorn app.main:app --reload --port 8000
# 打开 http://localhost:8000
```

### 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

## 使用流程

1. **输入**：粘贴原文、修订文、标签 JSON（或点"填入示例"）。
   约束：每段文本 ≤ 120 个 Unicode 码点；标签 ≤ 10 个、非空、互不重叠，
   区间为原文码点的左闭右开区间 `[start, end)`。
2. **保存并分析**：服务端按最小编辑距离（插/删/换各计 1）枚举全部最优对齐：
   - **自动迁移**：所有最优对齐都保留该标签且落点唯一；
   - **待选择**：有多个落点，或落点唯一但并非所有最优对齐都保留——
     逐项选择落点或丢弃；
   - **无法保留**：任何最优对齐都不能让原跨度字符完整连续地出现，只能丢弃。
3. **确认并生成结果**：整份选择必须被**同一个最优对齐**支持，
   否则拒绝并指出冲突标签；确认成功后才能下载
   `修订文 + 完整标签 JSON`。失败不会留下任何结果。
4. **行图训练清单**（确认成功后）：上传切图 JSON，包含切图时的 OCR
   原文（须与会话原文逐码点一致）以及至多 20 张行图的唯一编号和原文
   Unicode 码点半开区间，区间须按行序无缝覆盖原文。服务端把修订文
   按行切开，要求所有分界落在**同一条**最小编辑对齐上，且该对齐还须
   支持全部已确认保留的实体落点；插在行间的字可归相邻任一行。
   - 所有合法切法给某张行图分到的文字**完全相同且非空** → 图号+文字
     写入可下载 JSON（位置不同但文字相同也可入选）；
   - 文字有多种可能 → 歧义行，界面展示全部可能文字（常见恰为两种），
     不写入下载文件；
   - 所有合法切法下文字都为空 → 确定空行，编号单列于 `empty_lines`。
5. **多窗口安全**：另一窗口修改任一输入后，旧选择、在途确认与已生成的
   行图清单都会被服务端拒绝（409 stale）或删除；把输入改回原样也不会
   恢复，需要重新选择确认、重新生成清单。

标签 JSON 格式（`id` 可省略，自动补 `L1..Ln`）：

```json
{"labels": [{"id": "ent1", "start": 0, "end": 4}]}
```

下载的结果 JSON：

```json
{
  "revised": "修订后文本",
  "labels": [{"id": "ent1", "start": 0, "end": 4, "text": "对应文本"}],
  "meta": {"session_id": "…", "revision": 3, "edit_distance": 2,
           "migrated": 1, "discarded": ["ent2"]}
}
```

行图上传 JSON（`id` 可改用 `[id, start, end]` 数组写法；顶层也兼容
`original`/`text` 字段名）：

```json
{
  "ocr_original": "切图时的 OCR 原文，须与会话原文完全一致",
  "lines": [
    {"id": "line-001", "start": 0, "end": 12},
    {"id": "line-002", "start": 12, "end": 25}
  ]
}
```

下载的行图训练清单 JSON（只含文字唯一且非空的行图，确定空行单列）：

```json
{
  "items": [{"id": "line-001", "text": "该行确定对应的修订文字"}],
  "empty_lines": ["line-002"],
  "meta": {"session_id": "…", "revision": 3, "edit_distance": 2,
           "legal_cut_count": 4, "line_count": 2, "ambiguous_count": 0}
}
```

## 重复文字示例（examples/）

| 文件 | 场景 | 现象 |
|---|---|---|
| `01_auto.json` | 句尾追加两字 | 两标签落点唯一，自动迁移 |
| `02_repeat_choice.json` | `abcabcabc → abcabc` | 中间那段 `abc` 可落在 `[0,3)` 或 `[3,6)`，需选择 |
| `03_impossible.json` | `银行 → 银号` | 原跨度字符被替换，无法保留，只能丢弃 |
| `04_conflict.json` | `abab → ab` 两个 `ab` 标签 | 各自唯一落点都是 `[0,2)`，同一对齐不可兼得，确认报冲突 |
| `05_line_manifest.json` | 行间插入 `X` 的两行切图 | 插入字可归相邻任一行，两行均歧义、各两种文字，清单不入选 |

网页左上角"示例"下拉可一键填入前四组数据；第 4 步面板的
"填入示例切图 JSON"按钮按当前会话原文生成无缝覆盖的两行切图。

## 算法说明

设原文 `s`、修订文 `t`，`d1[i][j] = ed(s[:i], t[:j])`，
`d2[i][j] = ed(s[i:], t[j:])`，`D = ed(s, t)`。

- **落点判定**：标签 `[a,b)` 在某最优对齐中完整连续落在 `[p,p+L)`
  ⟺ `t[p:p+L] == s[a:b]` 且 `d1[a][p] + d2[b][p+L] == D`。
  （前缀最优 + 跨度逐字匹配 + 后缀最优，即拼出一个最优对齐。）
- **自动迁移判定**：对最优对齐计数（`c1/c2` 路径计数，大整数），
  保留该标签的对齐数 `Σp c1[a][p]·c2[b][p+L]` 等于总数且落点唯一。
- **联合一致性**：把保留的跨度按源序排列，目标区间必须同序不重叠，
  且各段（前缀/间隙/后缀）最小编辑代价之和等于 `D`；
  冲突时给出目标次序冲突的标签对，或搜索一个极小的不一致标签子集。

正确性由 `tests/test_bruteforce.py` 保证：对小字符串穷举全部最优对齐，
逐跨度对照落点集合、"所有对齐都保留"标志与联合一致性判定。

### 行图切分算法（`app/line_manifest.py`）

- **可行点**：把已确认实体跨度内的逐字匹配边标为必经边；用前向
  `fwd[i][j]`（从起点到该点最多按序经过几条必经边）与反向
  `back[i][j]`（离开该点后最多还能经过几条）两次 DP，配合
  `d1+d2==D` 判定网格点是否属于某条**支持全部实体**的最优对齐。
- **分界传播**：在可行点子图上只沿最优边（匹配 0 代价，插/删/换 1 代价）
  按源分界列逐列 BFS。到达分界列后仍可沿插入边横向移动，对应
  "插在行间的字归前一行或后一行"两种选择；末分界固定为 `(m,n)`。
- **结论分类**：传播过程中累计每行在全部合法切法下分到的修订文字
  （按路径数计数，故切法数也是精确值）。文字集合唯一且非空 →
  `certain`（入下载清单），唯一且为空 → `empty`（单列），多于一个 →
  `ambiguous`（展示全部可能文字，不入选）。
- 正确性由 `tests/test_line_manifest.py` 保证：对小字符串穷举全部
  最优**网格路径**（含插入段内点），逐行对照文字集合、切法总数与
  分类；带实体的随机用例同样暴力对照。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/sessions` | 创建会话（三件套），返回分析结果与 `revision` |
| GET | `/api/sessions/{sid}` | 当前输入 + 分析 + 已存选择 + 结果/清单状态 |
| PUT | `/api/sessions/{sid}/inputs` | 修改任一输入：`revision` 必变，旧选择/结果/清单同事务清除 |
| PUT | `/api/sessions/{sid}/selections` | 保存选择（整表替换），`revision` 过期返回 409 |
| POST | `/api/sessions/{sid}/confirm` | 事务内校验版本与联合一致性，成功才原子写入结果 |
| GET | `/api/sessions/{sid}/download` | 仅当前 `revision` 已确认成功时可下载 |
| POST | `/api/sessions/{sid}/line-manifest` | 上传切图 JSON 生成行图清单（须先确认；旧 revision 返回 409） |
| GET | `/api/sessions/{sid}/line-manifest/download` | 下载行图训练清单；带过期 `revision` 参数返回 409 |

错误码：`400` 输入非法（含清单原文不一致、区间不无缝、未确认先上传）；
`404` 会话/结果/清单不存在；`409 stale_revision` 输入已被他处修改或
下载旧版本清单；`409 conflict` 选择互相冲突（附冲突标签）；
`422 unresolved` 存在未处理的待选标签。

## 目录结构

```
app/
  alignment.py    # 最小编辑距离、落点集合、对齐计数、联合一致性
  line_manifest.py# 行图切分：上传校验 + 实体必经边 DP + 分界传播
  main.py         # FastAPI 路由、确认事务、行图清单接口
  db.py           # SQLite（sessions / selections / results / line_manifests）
  schemas.py      # 输入校验
  static/         # 单页网页（原生 JS）
tests/            # 单元、API、暴力对照（对齐 + 行图切分）测试
examples/         # 五组示例（含重复文字与行图清单）
Dockerfile  docker-compose.yml  requirements*.txt
```

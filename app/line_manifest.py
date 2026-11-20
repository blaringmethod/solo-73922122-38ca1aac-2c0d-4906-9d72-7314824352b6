"""行图训练清单：校验上传切图信息，并在最优对齐图上枚举全部合法切法。

背景：
切图工序给出原文 s（OCR 文本）按行切分的半开码点区间
[b0,b1), [b1,b2), ..., [b_{k-1},b_k)，b0=0、b_k=m，每张行图带唯一编号。
修订文 t 要按行切成 k 段，要求：

1. 所有分界 (b_r, j_r) 落在**同一条**最小编辑对齐（Levenshtein 最优路径）上；
2. 该对齐还必须支持全部已确认保留的实体（实体跨度逐字匹配边是必经边）；
3. 插在两条源分界之间的插入字可归前一行或后一行，即同一源列 b_r 上
   可行目标列 j_r 可取“进入列的匹配/替换边落点”到“离开列的删除边起点”
   之间任意值，天然覆盖两种归属。

实现：
- 先在 d1/d2 上只保留最优边，构造最优对齐 DAG；
- 把实体匹配边标记为必经边（required），用两次 DP（fwd/back）求每个
  网格点上“经过该点且包含全部必经边的最优路径”是否存在，即该点是否
  属于某条**支持全部实体**的最优对齐（可行点）；
- 在可行点子图上，按源分界列逐列做 BFS 传播可达行，第 r 次传播得到
  第 r 张行图在全部合法切法下能分到的修订文文字集合；
- 文字集合恰好一个且非空 -> 可入清单；恰好一个且为空 -> 确定空行；
  多于一个 -> 歧义行（展示全部可能文字，常见情况恰为两种）。
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Sequence

from .alignment import compute_tables

MAX_LINES = 20


class ManifestValidationError(ValueError):
    """行图上传 JSON 校验失败，message 面向用户。"""


def normalize_line_payload(raw, session_original: str) -> dict:
    """校验并规范化上传的行图 JSON。

    接受形如
        {"ocr_original": "...",
         "lines": [{"id": "L1", "start": 0, "end": 3}, ...]}
    顶层兼容 original/text 字段名，行内兼容 [id, start, end] 数组。
    返回 {"ocr_original": str, "lines": [{"id","start","end"}, ...]}（按行序）。
    """
    if not isinstance(raw, dict):
        raise ManifestValidationError("行图 JSON 必须是对象")

    ocr = raw.get("ocr_original")
    if ocr is None:
        ocr = raw.get("original")
    if ocr is None:
        ocr = raw.get("text")
    if not isinstance(ocr, str):
        raise ManifestValidationError(
            "缺少 ocr_original 字段：须为切图时的 OCR 原文字符串")
    if ocr != session_original:
        raise ManifestValidationError(
            "上传的 OCR 原文与会话原文不一致（须逐码点完全相同），"
            "请使用本会话当前原文对应的切图文件")

    lines = raw.get("lines")
    if lines is None:
        lines = raw.get("line_images")
    if not isinstance(lines, list) or not lines:
        raise ManifestValidationError("lines 必须是非空数组")
    if len(lines) > MAX_LINES:
        raise ManifestValidationError(
            f"行图数量 {len(lines)} 超过上限 {MAX_LINES}")

    m = len(ocr)
    out = []
    seen_ids = set()
    for idx, item in enumerate(lines, 1):
        if isinstance(item, dict):
            lid = item.get("id")
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            lid, start, end = item
        else:
            raise ManifestValidationError(
                f"第 {idx} 张行图不是对象，也不是 [id, start, end] 数组")
        if not isinstance(lid, str) or not lid.strip():
            raise ManifestValidationError(f"第 {idx} 张行图的 id 必须是非空字符串")
        lid = lid.strip()
        if lid in seen_ids:
            raise ManifestValidationError(f"行图 id 重复：{lid!r}")
        seen_ids.add(lid)
        for name, v in (("start", start), ("end", end)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ManifestValidationError(
                    f"行图 {lid!r} 的 {name} 必须是整数（Unicode 码点下标）")
        if not (0 <= start < end <= m):
            raise ManifestValidationError(
                f"行图 {lid!r} 区间 [{start}, {end}) 非法：须满足 "
                f"0 <= start < end <= 原文长度 {m}（行区间不能为空）")
        out.append({"id": lid, "start": start, "end": end})

    out.sort(key=lambda x: (x["start"], x["end"]))
    # 无缝覆盖：按行序首尾相接，从 0 到 m，不重不漏
    if out[0]["start"] != 0 or out[-1]["end"] != m:
        raise ManifestValidationError(
            f"行区间须无缝覆盖整个原文 [0, {m})：实际起点 {out[0]['start']}，"
            f"终点 {out[-1]['end']}")
    for prev, cur in zip(out, out[1:]):
        if cur["start"] != prev["end"]:
            where = "重叠" if cur["start"] < prev["end"] else "留有空隙"
            raise ManifestValidationError(
                f"行区间须按行序无缝覆盖原文：{prev['id']!r} 的终点 "
                f"{prev['end']} 与 {cur['id']!r} 的起点 {cur['start']} 之间{where}")
    return {"ocr_original": ocr, "lines": out}


def _feasible_points(s: str, t: str, kept: Sequence[tuple[int, int, int]]):
    """计算属于“某条支持全部实体的最优对齐”的网格点集合。

    kept: 已确认保留实体 (a, b, p)，实体跨度 s[a:b] 逐字匹配到 t[p:p+L]。
    返回 (D, d1, feasible)，feasible 为 (m+1) x (n+1) 的 bool 表。
    """
    m, n = len(s), len(t)
    d1, d2, D = compute_tables(s, t)

    # 必经边：实体跨度内的每条逐字匹配边 (i,j)->(i+1,j+1)，按源位置索引
    required = {}
    for a, b, p in kept:
        for k in range(b - a):
            i, j = a + k, p + k
            # 实体在确认时已校验过文字与最优性，这里做防御性检查
            if not (i < m and j < n and s[i] == t[j]):
                raise ValueError(f"实体边 ({i},{j}) 不是匹配边")
            if i in required and required[i] != j:
                raise ValueError(f"实体边在源位置 {i} 冲突")
            required[i] = j
    req_sources = sorted(required)                 # 必经边源位置，升序

    def req_rank(i: int, j: int) -> int:
        """边 (i,j)->(i+1,j+1) 若是必经边返回其序号，否则 -1。"""
        if required.get(i) == j:
            return bisect_left(req_sources, i)
        return -1

    R = len(req_sources)

    # fwd[i][j]：从 (0,0) 到 (i,j) 的最优路径上最多能按序经过多少条必经边。
    # 格点递推：i、j 均递增，三类前驱 (i-1,j)/(i,j-1)/(i-1,j-1) 都已算出。
    fwd = [[-1] * (n + 1) for _ in range(m + 1)]
    fwd[0][0] = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            best = -1
            if i > 0 and d1[i][j] == d1[i - 1][j] + 1:          # 删除边
                best = max(best, fwd[i - 1][j])
            if j > 0 and d1[i][j] == d1[i][j - 1] + 1:          # 插入边
                best = max(best, fwd[i][j - 1])
            if i > 0 and j > 0:                                 # 匹配/替换边
                cost = 0 if s[i - 1] == t[j - 1] else 1
                if d1[i][j] == d1[i - 1][j - 1] + cost:
                    cand = fwd[i - 1][j - 1]
                    if cand >= 0:
                        rk = req_rank(i - 1, j - 1)
                        if rk >= 0:
                            if rk == cand:
                                cand += 1
                            else:
                                cand = -1
                    best = max(best, cand)
            fwd[i][j] = best

    # back[i][j]：从 (i,j) 到 (m,n) 的最优路径上，离开 (i,j) 后最多还能
    # 按序经过多少条必经边（back[m][n]=0）。格点递推：i、j 均递减，
    # 三类后继 (i+1,j)/(i,j+1)/(i+1,j+1) 都已算出。
    back = [[-1] * (n + 1) for _ in range(m + 1)]
    back[m][n] = 0
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m and j == n:
                continue
            best = -1
            if i < m and d2[i][j] == d2[i + 1][j] + 1:          # 删除边
                best = max(best, back[i + 1][j])
            if j < n and d2[i][j] == d2[i][j + 1] + 1:          # 插入边
                best = max(best, back[i][j + 1])
            if i < m and j < n:                                 # 匹配/替换边
                cost = 0 if s[i] == t[j] else 1
                if d2[i][j] == d2[i + 1][j + 1] + cost:
                    cand = back[i + 1][j + 1]
                    if cand >= 0:
                        rk = req_rank(i, j)
                        if rk >= 0:
                            if rk != R - 1 - cand:
                                cand = -1
                            else:
                                cand += 1
                    best = max(best, cand)
            back[i][j] = best

    # 可行点：存在经过该点的最优路径，且该路径能按序收齐全部 R 条必经边
    feasible = [[False] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if d1[i][j] + d2[i][j] == D \
                    and fwd[i][j] >= 0 and back[i][j] >= 0 \
                    and fwd[i][j] + back[i][j] == R:
                feasible[i][j] = True
    return D, d1, feasible


def _reachable_rows(s, t, d1, feasible, n: int, start_i: int, start_j: int,
                    target_i: int) -> set[int]:
    """在“支持全部实体的最优对齐”子图上，从 (start_i,start_j) 单调走到
    源列 target_i 时能到达的全部目标列 j（即该分界列上的可行 j_r）。

    只沿最优边行走，并要求每个落点都是可行点；到达 target_i 列后仍
    允许沿插入边在列内横向移动（插入字归本行/下一行的两种选择正体现
    为 j 的取值区间），但不能再跨到下一源列。
    """
    m = len(feasible) - 1
    seen = [[False] * (n + 1) for _ in range(m + 1)]
    queue = [(start_i, start_j)]
    seen[start_i][start_j] = True
    head = 0
    while head < len(queue):
        i, j = queue[head]
        head += 1
        base = d1[i][j]
        # 删除边：代价恒为 1
        if i < target_i and i + 1 <= m \
                and feasible[i + 1][j] and d1[i + 1][j] == base + 1 \
                and not seen[i + 1][j]:
            seen[i + 1][j] = True
            queue.append((i + 1, j))
        # 匹配/替换边：代价 0（同字符）或 1（不同字符）
        if i < target_i and i + 1 <= m and j + 1 <= n \
                and feasible[i + 1][j + 1] and not seen[i + 1][j + 1]:
            cost = 0 if s[i] == t[j] else 1
            if d1[i + 1][j + 1] == base + cost:
                seen[i + 1][j + 1] = True
                queue.append((i + 1, j + 1))
        # 插入边：代价恒为 1；任何列（含目标列）内都可继续向右
        if j + 1 <= n and feasible[i][j + 1] \
                and d1[i][j + 1] == base + 1 and not seen[i][j + 1]:
            seen[i][j + 1] = True
            queue.append((i, j + 1))
    return {j for j in range(n + 1) if seen[target_i][j]}


def build_manifest(s: str, t: str, lines: Sequence[dict],
                   kept: Sequence[tuple[int, int, int]]) -> dict:
    """完整流水线：校验后的行图 + 已确认实体 -> 分析结果与可下载清单。

    lines: normalize_line_payload 的输出（含 id/start/end，按行序）。
    kept:  [(a, b, p), ...] 已确认保留实体（源区间 -> 修订文落点）。
    返回：
      {
        "distance": int,
        "legal_cut_count": int,                 # 合法切法（j 元组）数量
        "lines": [{"id","index","source_range","status","texts"}],
        "manifest": {"items": [{"id","text"}], "empty_lines": [id, ...]}
      }
    """
    m, n = len(s), len(t)
    boundaries = [0] + [ln["end"] for ln in lines]
    D, d1, feasible = _feasible_points(s, t, kept)
    if not feasible[0][0] or not feasible[m][n]:
        # 实体联合一致性在确认环节已校验，理论上不可达
        raise ValueError("不存在支持全部已确认实体的最优对齐")

    k = len(lines)
    line_texts: list[set[str]] = [set() for _ in range(k)]

    # 当前源列 boundaries[r] 上，所有合法前缀切法可达的目标列集合，
    # 以及到达每个目标列的合法前缀切法（j 元组）数量。
    current = {0}
    counts = {0: 1}
    legal_cut_count = 0
    for r in range(k):
        target_i = boundaries[r + 1]
        next_rows = set()
        next_counts: dict[int, int] = {}
        for start_j in current:
            rows = _reachable_rows(s, t, d1, feasible, n,
                                   boundaries[r], start_j, target_i)
            if r == k - 1:
                # 最后一条分界固定为 (m, n)
                rows = {n} if n in rows else set()
            for j in rows:
                line_texts[r].add(t[start_j:j])
                next_rows.add(j)
                next_counts[j] = next_counts.get(j, 0) + counts[start_j]
        if not next_rows:
            raise ValueError("找不到合法切法（理论上不可达）")
        if r == k - 1:
            legal_cut_count = sum(next_counts.values())
        current = next_rows
        counts = next_counts

    lines_out = []
    certain_items = []
    empty_ids = []
    for r, texts in enumerate(line_texts):
        ordered = sorted(texts)
        if len(ordered) == 1 and ordered[0] != "":
            status = "certain"
            certain_items.append({"id": lines[r]["id"], "text": ordered[0]})
        elif len(ordered) == 1:
            status = "empty"
            empty_ids.append(lines[r]["id"])
        else:
            status = "ambiguous"
        lines_out.append({
            "id": lines[r]["id"],
            "index": r,
            "source_range": [lines[r]["start"], lines[r]["end"]],
            "status": status,
            "texts": ordered,
        })

    return {
        "distance": D,
        "legal_cut_count": legal_cut_count,
        "lines": lines_out,
        "manifest": {"items": certain_items, "empty_lines": empty_ids},
    }

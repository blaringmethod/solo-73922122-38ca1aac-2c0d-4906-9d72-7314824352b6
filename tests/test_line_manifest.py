"""行图切分的暴力对照测试。

对小字符串穷举全部最优对齐（沿用 test_bruteforce 的枚举方式），
逐条对齐按源分界列取出目标列 j 元组作为一种合法切法（插入边允许
j 落在进入列/离开列之间的任意值，与产品规则一致），再与
line_manifest.build_manifest 的结论逐项对照：

- 合法切法总数；
- 每张行图在全部合法切法下的文字集合（去重）；
- certain / empty / ambiguous 分类；
- 可下载清单 items 与 empty_lines。

另外覆盖实体约束：只保留支持全部已确认实体的对齐。
"""

import sys
from itertools import product
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.alignment import edit_distance
from app.line_manifest import (MAX_LINES, ManifestValidationError,
                               build_manifest, normalize_line_payload)


# ---------------- 暴力枚举 ----------------

def brute_optimal_paths(s: str, t: str):
    """枚举全部最优对齐，每条对齐返回“匹配对 (i,j)”集合
    （替换不算匹配，与 test_bruteforce 一致）。"""
    m, n = len(s), len(t)
    D = edit_distance(s, t)
    found = set()

    def rec(i, j, cost, pairs):
        if cost > D:
            return
        if i == m and j == n:
            if cost == D:
                found.add(tuple(pairs))
            return
        if i < m and j < n:
            if s[i] == t[j]:
                rec(i + 1, j + 1, cost, pairs + ((i, j),))
            else:
                rec(i + 1, j + 1, cost + 1, pairs)
        if i < m:
            rec(i + 1, j, cost + 1, pairs)
        if j < n:
            rec(i, j + 1, cost + 1, pairs)

    rec(0, 0, 0, ())
    return [set(p) for p in found]


def path_supports_entities(pairs, kept):
    d = dict(pairs)
    for a, b, p in kept:
        if any(d.get(i) != p + (i - a) for i in range(a, b)):
            return False
    return True


def brute_optimal_grid_paths(s: str, t: str):
    """枚举全部最优网格路径，每条路径是 (i,j) 网格点序列（含插入内点）。

    与 brute_optimal_paths 的区别：这里保留插入段在源列内的每一步，
    因此同一匹配对集合的多条网格路径（插入归属不同）会分别出现。
    """
    from app.alignment import edit_distance
    m, n = len(s), len(t)
    D = edit_distance(s, t)
    paths = []

    def rec(i, j, cost, path):
        if cost > D:
            return
        if i == m and j == n:
            if cost == D:
                paths.append(tuple(path))
            return
        if i < m and j < n:
            rec(i + 1, j + 1,
                cost + (0 if s[i] == t[j] else 1),
                path + ((i + 1, j + 1),))
        if i < m:
            rec(i + 1, j, cost + 1, path + ((i + 1, j),))
        if j < n:
            rec(i, j + 1, cost + 1, path + ((i, j + 1),))

    rec(0, 0, 0, ((0, 0),))
    return paths


def _path_supports_entities_grid(path, s, t, kept):
    pairs = {(i, j) for (i, j), (pi, pj) in zip(path, path[1:])
             if pi == i + 1 and pj == j + 1 and i < len(s) and j < len(t)
             and s[i] == t[j]}
    d = dict(pairs)
    for a, b, p in kept:
        if any(d.get(i) != p + (i - a) for i in range(a, b)):
            return False
    return True


def brute_line_texts(s, t, boundaries, kept):
    """与 build_manifest 同口径的暴力结果：(切法数, 每行文字集合)。

    直接枚举全部最优网格路径，取每条路径在源分界列上经过的目标列
    （同列插入段的每个内点都是合法分界取值）。
    """
    m, n = len(s), len(t)
    line_texts = [set() for _ in range(len(boundaries) - 1)]
    cut_keys = set()
    bset = set(boundaries)
    for path in brute_optimal_grid_paths(s, t):
        if not _path_supports_entities_grid(path, s, t, kept):
            continue
        # 每个分界列在该路径上经过的全部目标列
        col_values = {b: sorted({j for (i, j) in path if i == b})
                      for b in boundaries}
        col_values[0] = [0]
        col_values[m] = [n]
        for combo in product(*[col_values[b] for b in boundaries]):
            if any(x > y for x, y in zip(combo, combo[1:])):
                continue
            cut_keys.add(combo)
            for r, (j0, j1) in enumerate(zip(combo, combo[1:])):
                line_texts[r].add(t[j0:j1])
    return len(cut_keys), line_texts


def all_strings(alphabet, max_len):
    out = [""]
    for L in range(1, max_len + 1):
        out += ["".join(c) for c in product(alphabet, repeat=L)]
    return out


def make_lines(s, k):
    """把长度 m 的 s 近似均分成 k 个非空源行，返回行图描述与分界列。"""
    m = len(s)
    if k > m:
        k = m
    edges = sorted({0, m} | {
        round(m * r / k) for r in range(1, k) if 0 < round(m * r / k) < m})
    lines = [{"id": f"L{i+1}", "start": a, "end": b}
             for i, (a, b) in enumerate(zip(edges, edges[1:]))]
    return lines, edges


# ---------------- 对照测试 ----------------

@pytest.mark.parametrize("alphabet,max_len", [("ab", 4), ("abc", 3)])
def test_manifest_matches_bruteforce_no_entities(alphabet, max_len):
    strs = all_strings(alphabet, max_len)
    for s in strs:
        if not s:
            continue
        for t in strs:
            for k in range(1, min(len(s), 3) + 1):
                lines, boundaries = make_lines(s, k)
                got = build_manifest(s, t, lines, [])
                want_count, want_texts = brute_line_texts(
                    s, t, boundaries, [])
                assert got["legal_cut_count"] == want_count, (s, t, k)
                for r, want in enumerate(want_texts):
                    assert set(got["lines"][r]["texts"]) == want, (s, t, k, r)
                    status = got["lines"][r]["status"]
                    if want == {""}:
                        assert status == "empty"
                    elif len(want) == 1:
                        assert status == "certain"
                    else:
                        assert status == "ambiguous"
                # 下载清单内容
                want_items = [{"id": lines[r]["id"], "text": next(iter(w))}
                              for r, w in enumerate(want_texts) if len(w) == 1 and w != {""}]
                assert got["manifest"]["items"] == want_items
                want_empty = [lines[r]["id"] for r, w in enumerate(want_texts)
                              if w == {""}]
                assert got["manifest"]["empty_lines"] == want_empty


def test_manifest_with_entities_matches_bruteforce():
    # 带实体约束的定向小字符串对照
    cases = [
        ("甲乙丙丁", "甲乙X丙丁", [(2, 4, 3)]),
        ("甲乙丙丁", "X甲乙丙丁", [(1, 3, 2)]),
        ("abcabcabc", "abcabc", [(3, 6, 3)]),
        ("abcabcabc", "abcabc", [(0, 3, 0)]),
        ("abab", "ab", [(0, 2, 0)]),
        ("aab", "ab", [(0, 1, 0)]),
        ("中国人民银行", "中国人民银号", [(0, 4, 0)]),
    ]
    for s, t, kept in cases:
        for k in range(1, min(len(s), 3) + 1):
            lines, boundaries = make_lines(s, k)
            got = build_manifest(s, t, lines, kept)
            want_count, want_texts = brute_line_texts(s, t, boundaries, kept)
            assert got["legal_cut_count"] == want_count, (s, t, k, kept)
            for r, want in enumerate(want_texts):
                assert set(got["lines"][r]["texts"]) == want, (s, t, k, r)


def test_random_cases_with_entities():
    import random
    rng = random.Random(20261120)
    for _ in range(120):
        m = rng.randint(1, 6)
        n = rng.randint(0, 6)
        s = "".join(rng.choice("aab") for _ in range(m))
        t = "".join(rng.choice("aab") for _ in range(n))
        # 随机取 1~2 个不重叠源跨度，落点只选真实可行位置
        from app.alignment import compute_tables, possible_positions
        d1, d2, D = compute_tables(s, t)
        kept = []
        a = rng.randint(0, m - 1)
        b = rng.randint(a + 1, m)
        poss = possible_positions(s, t, a, b, d1, d2, D)
        if poss:
            kept.append((a, b, rng.choice(poss)))
        lines, boundaries = make_lines(s, rng.randint(1, min(3, m)))
        got = build_manifest(s, t, lines, kept)
        want_count, want_texts = brute_line_texts(s, t, boundaries, kept)
        assert got["legal_cut_count"] == want_count, (s, t, kept)
        for r, want in enumerate(want_texts):
            assert set(got["lines"][r]["texts"]) == want, (s, t, kept, r)


# ---------------- 可行点 DP 对照 ----------------

def test_feasible_points_match_bruteforce():
    from app.alignment import compute_tables, possible_positions
    from app.line_manifest import _feasible_points
    rng = __import__("random").Random(20261121)
    for _ in range(200):
        m = rng.randint(0, 5)
        n = rng.randint(0, 5)
        s = "".join(rng.choice("ab") for _ in range(m))
        t = "".join(rng.choice("ab") for _ in range(n))
        d1, d2, D = compute_tables(s, t)
        # 随机挑 0~2 个源区间不重叠的可行实体跨度作为必经边
        kept = []
        used_src = []
        if m > 0:
            for _e in range(rng.randint(0, 2)):
                a = rng.randint(0, m - 1)
                b = rng.randint(a + 1, m)
                if any(not (b <= x or a >= y) for x, y in used_src):
                    continue
                poss = possible_positions(s, t, a, b, d1, d2, D)
                if poss:
                    kept.append((a, b, rng.choice(poss)))
                    used_src.append((a, b))
        # 暴力：枚举支持全部实体的最优网格路径上出现过的网格点
        brute_pts = set()
        for path in brute_optimal_grid_paths(s, t):
            if _path_supports_entities_grid(path, s, t, kept):
                brute_pts.update(path)
        if not brute_pts:
            # 实体联合不相容（无一条最优路径能同时支持），跳过：
            # 该情形在确认环节已被 check_joint 拦截，不会进入本模块
            continue
        _D, _d1, feasible = _feasible_points(s, t, kept)
        algo_pts = {(i, j) for i in range(m + 1) for j in range(n + 1)
                    if feasible[i][j]}
        assert algo_pts == brute_pts, (s, t, kept)


# ---------------- 上传校验 ----------------

class TestNormalize:
    def test_basic_and_sorted(self):
        raw = {"ocr_original": "abcde",
               "lines": [{"id": "b", "start": 2, "end": 5},
                         {"id": "a", "start": 0, "end": 2}]}
        out = normalize_line_payload(raw, "abcde")
        assert [x["id"] for x in out["lines"]] == ["a", "b"]
        assert out["ocr_original"] == "abcde"

    def test_original_mismatch_rejected(self):
        with pytest.raises(ManifestValidationError, match="不一致"):
            normalize_line_payload(
                {"ocr_original": "abc", "lines": [{"id": "a", "start": 0, "end": 3}]},
                "abd")

    def test_gap_and_overlap_rejected(self):
        with pytest.raises(ManifestValidationError, match="空隙"):
            normalize_line_payload(
                {"ocr_original": "abcde",
                 "lines": [{"id": "a", "start": 0, "end": 2},
                           {"id": "b", "start": 3, "end": 5}]}, "abcde")
        with pytest.raises(ManifestValidationError, match="重叠"):
            normalize_line_payload(
                {"ocr_original": "abcde",
                 "lines": [{"id": "a", "start": 0, "end": 3},
                           {"id": "b", "start": 2, "end": 5}]}, "abcde")

    def test_endpoints_must_cover_whole_text(self):
        with pytest.raises(ManifestValidationError, match="无缝覆盖"):
            normalize_line_payload(
                {"ocr_original": "abcde",
                 "lines": [{"id": "a", "start": 1, "end": 5}]}, "abcde")

    def test_duplicate_id_and_empty_interval(self):
        with pytest.raises(ManifestValidationError, match="id 重复"):
            normalize_line_payload(
                {"ocr_original": "ab",
                 "lines": [{"id": "x", "start": 0, "end": 1},
                           {"id": "x", "start": 1, "end": 2}]}, "ab")
        with pytest.raises(ManifestValidationError):
            normalize_line_payload(
                {"ocr_original": "ab",
                 "lines": [{"id": "x", "start": 0, "end": 0},
                           {"id": "y", "start": 0, "end": 2}]}, "ab")

    def test_too_many_lines(self):
        lines = [{"id": f"L{i}", "start": i, "end": i + 1}
                 for i in range(MAX_LINES + 1)]
        with pytest.raises(ManifestValidationError, match="20"):
            text = "x" * (MAX_LINES + 1)
            normalize_line_payload(
                {"ocr_original": text, "lines": lines}, text)

    def test_array_form_accepted(self):
        out = normalize_line_payload(
            {"ocr_original": "abc", "lines": [["a", 0, 1], ["b", 1, 3]]},
            "abc")
        assert [x["id"] for x in out["lines"]] == ["a", "b"]

    def test_missing_fields(self):
        with pytest.raises(ManifestValidationError):
            normalize_line_payload({"lines": []}, "abc")
        with pytest.raises(ManifestValidationError):
            normalize_line_payload({"ocr_original": "abc"}, "abc")


# ---------------- 业务语义定向用例 ----------------

class TestSemantics:
    def test_insertion_between_two_lines_ambiguous(self):
        # 插入 X 在两行之间：两种归属，两行都歧义且各恰有两种文字
        r = build_manifest(
            "甲乙丙丁", "甲乙X丙丁",
            [{"id": "L1", "start": 0, "end": 2},
             {"id": "L2", "start": 2, "end": 4}], [])
        assert r["legal_cut_count"] == 2
        assert r["lines"][0]["texts"] == ["甲乙", "甲乙X"]
        assert r["lines"][1]["texts"] == ["X丙丁", "丙丁"]
        assert all(l["status"] == "ambiguous" for l in r["lines"])
        # 歧义行不进下载清单，也不是确定空行
        assert r["manifest"] == {"items": [], "empty_lines": []}

    def test_head_and_tail_insertion_unambiguous(self):
        lines = [{"id": "L1", "start": 0, "end": 2},
                 {"id": "L2", "start": 2, "end": 4}]
        r = build_manifest("甲乙丙丁", "X甲乙丙丁", lines, [])
        assert r["manifest"]["items"] == [
            {"id": "L1", "text": "X甲乙"}, {"id": "L2", "text": "丙丁"}]
        r = build_manifest("甲乙丙丁", "甲乙丙丁X", lines, [])
        assert r["manifest"]["items"] == [
            {"id": "L1", "text": "甲乙"}, {"id": "L2", "text": "丙丁X"}]

    def test_determined_empty_line(self):
        # 第二行源字符全部被删且没有插入字可归 -> 确定空行，单列
        r = build_manifest(
            "甲乙丙丁", "甲乙",
            [{"id": "L1", "start": 0, "end": 2},
             {"id": "L2", "start": 2, "end": 4}], [])
        assert r["lines"][1]["status"] == "empty"
        assert r["lines"][1]["texts"] == [""]
        assert r["manifest"]["empty_lines"] == ["L2"]
        assert r["manifest"]["items"] == [{"id": "L1", "text": "甲乙"}]

    def test_empty_line_vs_deleted_with_kept_char(self):
        # 乙丙被删但丁保留：第二行非空，文字唯一为“丁”
        r = build_manifest(
            "甲乙丙丁", "甲丁",
            [{"id": "L1", "start": 0, "end": 2},
             {"id": "L2", "start": 2, "end": 4}], [])
        assert r["lines"][1]["status"] == "certain"
        assert r["manifest"]["items"] == [
            {"id": "L1", "text": "甲"}, {"id": "L2", "text": "丁"}]
        assert r["manifest"]["empty_lines"] == []

    def test_crossing_entity_pins_boundary(self):
        r = build_manifest(
            "甲乙丙丁", "X甲乙丙丁",
            [{"id": "L1", "start": 0, "end": 2},
             {"id": "L2", "start": 2, "end": 4}],
            [(1, 3, 2)])
        assert r["legal_cut_count"] == 1
        assert [l["status"] for l in r["lines"]] == ["certain", "certain"]

    def test_identical_texts_one_cut(self):
        lines = [{"id": f"L{i}", "start": i, "end": i + 1}
                 for i in range(4)]
        r = build_manifest("abcd", "abcd", lines, [])
        assert r["legal_cut_count"] == 1
        assert [l["texts"] for l in r["lines"]] == [[c] for c in "abcd"]

    def test_astral_codepoints(self):
        # emoji 单码点：插入一个 emoji 在行间，两种归属
        r = build_manifest(
            "😀😃😄", "😀😎😃😄",
            [{"id": "L1", "start": 0, "end": 1},
             {"id": "L2", "start": 1, "end": 3}], [])
        assert r["legal_cut_count"] == 2
        assert r["lines"][0]["texts"] == ["😀", "😀😎"]
        assert set(r["lines"][1]["texts"]) == {"😎😃😄", "😃😄"}

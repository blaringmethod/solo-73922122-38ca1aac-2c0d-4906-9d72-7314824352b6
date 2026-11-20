"""FastAPI 入口：OCR 修订后实体标签迁移工具。

流程：上传三件套 -> 分析（自动迁移/多落点/无法保留）-> 逐项选择 ->
确认（整份选择须被同一最优对齐支持，且 revision 未过期）-> 下载结果。
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .alignment import analyze_labels, check_joint
from .line_manifest import (ManifestValidationError, build_manifest,
                            normalize_line_payload)
from .schemas import (ValidationError, normalize_labels, normalize_selections,
                      validate_text)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="OCR 标签迁移工具")


@app.exception_handler(ValidationError)
async def validation_handler(_req: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ManifestValidationError)
async def manifest_validation_handler(_req: Request, exc: ManifestValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------- 请求模型 ----------------

class InputsIn(BaseModel):
    original: str
    revised: str
    labels: object  # list 或 {"labels": [...]}，由 normalize_labels 校验


class SelectionsIn(BaseModel):
    revision: int
    selections: object


class ConfirmIn(BaseModel):
    revision: int


class LineManifestIn(BaseModel):
    revision: int
    payload: object  # 行图切分 JSON，由 normalize_line_payload 校验


# ---------------- 工具 ----------------

def _validate_inputs(payload: InputsIn):
    original = validate_text(payload.original, "原文")
    revised = validate_text(payload.revised, "修订文")
    labels = normalize_labels(payload.labels, original)
    return original, revised, labels


def _must_get_session(sid: str) -> dict:
    sess = db.get_session(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return sess


def _kept_entities(sess: dict, result_data: dict) -> list[tuple[int, int, int]]:
    """从确认结果反推已保留实体 (a, b, p)：结果标签 id 与会话标签一一对应。"""
    src_by_id = {lab["id"]: (lab["start"], lab["end"]) for lab in sess["labels"]}
    kept = []
    for out in result_data["labels"]:
        a, b = src_by_id[out["id"]]
        kept.append((a, b, out["start"]))
    kept.sort()
    return kept


def _state(sess: dict) -> dict:
    analysis = analyze_labels(sess["original"], sess["revised"], sess["labels"])
    result = db.get_result(sess["id"], sess["revision"])
    manifest = db.get_line_manifest(sess["id"], sess["revision"])
    return {
        "session_id": sess["id"],
        "revision": sess["revision"],
        "original": sess["original"],
        "revised": sess["revised"],
        "labels": sess["labels"],
        "analysis": analysis,
        "selections": db.get_selections(sess["id"], sess["revision"]),
        "has_result": result is not None,
        "download_url": f"/api/sessions/{sess['id']}/download" if result else None,
        "line_manifest": manifest["data"] if manifest else None,
        "manifest_download_url":
            f"/api/sessions/{sess['id']}/line-manifest/download"
            f"?revision={sess['revision']}"
            if manifest else None,
    }


# ---------------- 页面 ----------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ---------------- API ----------------

@app.post("/api/sessions", status_code=201)
def create_session(payload: InputsIn):
    original, revised, labels = _validate_inputs(payload)
    sess = db.create_session(original, revised, labels)
    return _state(sess)


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return _state(_must_get_session(sid))


@app.put("/api/sessions/{sid}/inputs")
def update_inputs(sid: str, payload: InputsIn):
    original, revised, labels = _validate_inputs(payload)
    sess = db.update_inputs(sid, original, revised, labels)
    if sess is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _state(sess)


@app.put("/api/sessions/{sid}/selections")
def put_selections(sid: str, payload: SelectionsIn):
    """保存选择。版本检查与写入在同一事务：若另一窗口正修改输入，
    要么本请求先落库（对方随后使其失效），要么直接看到新版本返回 409。"""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="会话不存在")
        if row["revision"] != payload.revision:
            conn.rollback()
            raise HTTPException(
                status_code=409,
                detail={"code": "stale_revision",
                        "message": "输入已在别处修改，旧选择已失效，请基于最新分析重新选择",
                        "current_revision": row["revision"]})
        labels = json.loads(row["labels"])
        sels = normalize_selections(payload.selections, labels)
        # 落点必须真的在可能集合里，防止绕过前端伪造
        analysis = analyze_labels(row["original"], row["revised"], labels)
        possible_by_id = {l["id"]: l["possible"] for l in analysis["labels"]}
        for lid, sel in sels.items():
            if sel["choice"] == "position" and sel["position"] not in possible_by_id[lid]:
                conn.rollback()
                raise ValidationError(
                    f"标签 {lid!r} 的落点 {sel['position']} 不在可能位置集合中")
        conn.execute(
            "INSERT INTO selections(session_id, revision, data, updated_at)"
            " VALUES (?,?,?,?)"
            " ON CONFLICT(session_id, revision) DO UPDATE SET"
            " data=excluded.data, updated_at=excluded.updated_at",
            (sid, row["revision"], json.dumps(sels, ensure_ascii=False),
             db._now()))
        conn.commit()
        return {"ok": True, "revision": row["revision"]}
    except (HTTPException, ValidationError):
        raise
    finally:
        conn.close()


@app.post("/api/sessions/{sid}/confirm")
def confirm(sid: str, payload: ConfirmIn):
    """确认发布。整个"校验 revision + 校验联合一致 + 写结果"在一个
    IMMEDIATE 事务里：输入一旦被别的窗口改过（哪怕又改回去），revision
    必然不同，旧选择/在途确认都会被拒绝；任何失败都不会留下结果行。"""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="会话不存在")
        if row["revision"] != payload.revision:
            conn.rollback()
            raise HTTPException(
                status_code=409,
                detail={"code": "stale_revision",
                        "message": "输入已在别处修改，本次确认已作废，请重新分析并选择",
                        "current_revision": row["revision"]})

        original, revised = row["original"], row["revised"]
        labels = json.loads(row["labels"])
        srow = conn.execute(
            "SELECT data FROM selections WHERE session_id=? AND revision=?",
            (sid, row["revision"])).fetchone()
        sels = json.loads(srow["data"]) if srow else {}

        # 幂等：同一 revision 已成功确认过，直接返回既有结果
        existing = conn.execute(
            "SELECT data FROM results WHERE session_id=? AND revision=?",
            (sid, row["revision"])).fetchone()
        if existing is not None:
            conn.commit()
            return {"ok": True, "already": True,
                    "result": json.loads(existing["data"]),
                    "download_url": f"/api/sessions/{sid}/download"}

        analysis = analyze_labels(original, revised, labels)
        info_by_id = {l["id"]: l for l in analysis["labels"]}

        kept, discarded, unresolved = [], [], []
        for lab in labels:
            lid = lab["id"]
            info = info_by_id[lid]
            sel = sels.get(lid)
            if info["status"] == "impossible":
                discarded.append(lid)          # 无法保留：只能丢弃
                continue
            if info["status"] == "auto":
                if sel and sel.get("choice") == "discard":
                    discarded.append(lid)
                else:
                    kept.append((lab["start"], lab["end"],
                                 info["possible"][0], lid))
                continue
            # choice：必须显式选择
            if sel is None:
                unresolved.append(lid)
            elif sel.get("choice") == "discard":
                discarded.append(lid)
            elif sel.get("choice") == "position" and \
                    sel.get("position") in info["possible"]:
                kept.append((lab["start"], lab["end"], sel["position"], lid))
            else:
                conn.rollback()
                raise HTTPException(
                    status_code=400,
                    detail=f"标签 {lid!r} 的选择无效：落点不在可能位置集合中")

        if unresolved:
            conn.rollback()
            raise HTTPException(
                status_code=422,
                detail={"code": "unresolved",
                        "message": "存在未处理的标签，请逐项选择位置或丢弃",
                        "labels": unresolved})

        kept.sort(key=lambda x: x[0])
        ok, conflict_idx = check_joint(
            original, revised, [(a, b, p) for a, b, p, _ in kept],
            analysis["distance"])
        if not ok:
            conn.rollback()
            conflict_ids = [kept[i][3] for i in conflict_idx]
            raise HTTPException(
                status_code=409,
                detail={"code": "conflict",
                        "message": "整份选择无法被同一个最优对齐支持，请调整下列冲突标签",
                        "conflicting_labels": conflict_ids})

        # 成功：原子写入结果，随后才允许下载
        out_labels = [{
            "id": lid,
            "start": p,
            "end": p + (b - a),
            "text": revised[p:p + (b - a)],
        } for a, b, p, lid in kept]
        result = {
            "revised": revised,
            "labels": out_labels,
            "meta": {
                "session_id": sid,
                "revision": row["revision"],
                "edit_distance": analysis["distance"],
                "migrated": len(out_labels),
                "discarded": discarded,
            },
        }
        conn.execute(
            "INSERT INTO results(session_id, revision, data, created_at)"
            " VALUES (?,?,?,?)",
            (sid, row["revision"], json.dumps(result, ensure_ascii=False),
             db._now()))
        conn.commit()
        return {"ok": True, "already": False, "result": result,
                "download_url": f"/api/sessions/{sid}/download"}
    except HTTPException:
        raise
    finally:
        conn.close()


@app.get("/api/sessions/{sid}/download")
def download(sid: str):
    """只有当前 revision 已成功确认才提供下载；输入改过后旧结果不可下载。"""
    sess = _must_get_session(sid)
    result = db.get_result(sid, sess["revision"])
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="当前输入尚未确认成功，无法下载；失败不会留下任何结果")
    return JSONResponse(
        content=result["data"],
        headers={
            "Content-Disposition":
                f'attachment; filename="labels_revised_r{sess["revision"]}.json"'
        })


# ---------------- 行图训练清单 ----------------

@app.post("/api/sessions/{sid}/line-manifest")
def upload_line_manifest(sid: str, payload: LineManifestIn):
    """上传切图 JSON 并生成行图训练清单分析。

    前置：当前 revision 已确认成功（清单必须基于确认版本）。
    - revision 过期 -> 409 stale_revision（会话改版后拒绝旧清单）；
    - 原文不一致 / 区间不无缝 / 超过 20 张等 -> 400；
    - 成功后幂等落库，GET 会话即可取回。
    """
    sess = _must_get_session(sid)
    if sess["revision"] != payload.revision:
        current = db.get_session(sid)
        raise HTTPException(
            status_code=409,
            detail={"code": "stale_revision",
                    "message": "会话已改版，旧版本的行图清单请求已拒绝，"
                               "请基于最新确认版本重新上传",
                    "current_revision": current["revision"]})
    result = db.get_result(sid, sess["revision"])
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="当前版本尚未确认成功，请先完成标签确认再生成行图训练清单")

    normalized = normalize_line_payload(payload.payload, sess["original"])
    kept = _kept_entities(sess, result["data"])
    analysis = build_manifest(sess["original"], sess["revised"],
                              normalized["lines"], kept)
    # 带上上传原文与行区间，便于回显与审计
    analysis["upload"] = normalized
    db.save_line_manifest(sid, sess["revision"], analysis)
    return {"ok": True, "revision": sess["revision"], "analysis": analysis,
            "download_url":
            f"/api/sessions/{sid}/line-manifest/download"
            f"?revision={sess['revision']}"}


@app.get("/api/sessions/{sid}/line-manifest/download")
def download_line_manifest(sid: str, revision: int | None = None):
    """下载行图训练清单。仅当前 revision 已生成清单时可下载；
    显式携带旧 revision 时返回 409，防止会话改版后下载旧清单。"""
    sess = _must_get_session(sid)
    if revision is not None and revision != sess["revision"]:
        raise HTTPException(
            status_code=409,
            detail={"code": "stale_revision",
                    "message": "会话已改版，旧版本的行图清单不可下载，"
                               "请基于最新确认版本重新生成",
                    "current_revision": sess["revision"]})
    stored = db.get_line_manifest(sid, sess["revision"])
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail="当前版本尚未生成行图训练清单，请先上传切图 JSON")
    data = stored["data"]
    out = {
        "items": data["manifest"]["items"],
        "empty_lines": data["manifest"]["empty_lines"],
        "meta": {
            "session_id": sid,
            "revision": sess["revision"],
            "edit_distance": data["distance"],
            "legal_cut_count": data["legal_cut_count"],
            "line_count": len(data["lines"]),
            "ambiguous_count":
                sum(1 for ln in data["lines"] if ln["status"] == "ambiguous"),
        },
    }
    return JSONResponse(
        content=out,
        headers={
            "Content-Disposition":
                f'attachment; filename="line_manifest_r{sess["revision"]}.json"'
        })


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

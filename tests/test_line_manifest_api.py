"""行图训练清单的端到端 API 测试。

覆盖：上传校验、确认前置、清单生成与下载、会话改版失效（409）、
实体约束在清单中的作用、轮询状态回显。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def create_confirmed(client, original, revised, labels, selections=None):
    r = client.post("/api/sessions", json={
        "original": original, "revised": revised, "labels": labels})
    assert r.status_code == 201, r.text
    st = r.json()
    sid, rev = st["session_id"], st["revision"]
    if selections:
        r = client.put(f"/api/sessions/{sid}/selections", json={
            "revision": rev, "selections": selections})
        assert r.status_code == 200, r.text
    r = client.post(f"/api/sessions/{sid}/confirm", json={"revision": rev})
    assert r.status_code == 200, r.text
    return sid, rev


def payload(original, lines):
    return {"ocr_original": original, "lines": lines}


class TestUploadValidation:
    def test_requires_confirmation_first(self, client):
        r = client.post("/api/sessions", json={
            "original": "abc", "revised": "abc", "labels": []})
        st = r.json()
        r = client.post(f"/api/sessions/{st['session_id']}/line-manifest",
                        json={"revision": st["revision"],
                              "payload": payload("abc", [["L1", 0, 3]])})
        assert r.status_code == 400
        assert "确认" in r.json()["detail"]

    def test_original_mismatch(self, client):
        sid, rev = create_confirmed(client, "abc", "abc", [])
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("abd", [["L1", 0, 3]])})
        assert r.status_code == 400
        assert "不一致" in r.json()["detail"]

    def test_gap_rejected(self, client):
        sid, rev = create_confirmed(client, "abcde", "abcde", [])
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("abcde", [
                                  {"id": "a", "start": 0, "end": 2},
                                  {"id": "b", "start": 3, "end": 5}])})
        assert r.status_code == 400 and "空隙" in r.json()["detail"]

    def test_overlap_rejected(self, client):
        sid, rev = create_confirmed(client, "abcde", "abcde", [])
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("abcde", [
                                  {"id": "a", "start": 0, "end": 3},
                                  {"id": "b", "start": 2, "end": 5}])})
        assert r.status_code == 400 and "重叠" in r.json()["detail"]

    def test_duplicate_id(self, client):
        sid, rev = create_confirmed(client, "ab", "ab", [])
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("ab", [
                                  {"id": "x", "start": 0, "end": 1},
                                  {"id": "x", "start": 1, "end": 2}])})
        assert r.status_code == 400 and "重复" in r.json()["detail"]

    def test_too_many_lines(self, client):
        text = "x" * 21
        sid, rev = create_confirmed(client, text, text, [])
        lines = [{"id": f"L{i}", "start": i, "end": i + 1} for i in range(21)]
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload(text, lines)})
        assert r.status_code == 400 and "20" in r.json()["detail"]

    def test_bad_json_shape(self, client):
        sid, rev = create_confirmed(client, "abc", "abc", [])
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev, "payload": ["nope"]})
        assert r.status_code == 400

    def test_session_404(self, client):
        r = client.post("/api/sessions/nope/line-manifest",
                        json={"revision": 1, "payload": {}})
        assert r.status_code == 404


class TestGenerateAndDownload:
    def test_ambiguous_insertion_full_flow(self, client):
        # 插入 X 在两行之间：两种合法切法，两行都歧义，清单 items 为空
        sid, rev = create_confirmed(client, "甲乙丙丁", "甲乙X丙丁", [])
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("甲乙丙丁", [
                                  {"id": "L1", "start": 0, "end": 2},
                                  {"id": "L2", "start": 2, "end": 4}])})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["revision"] == rev
        lines = {l["id"]: l for l in body["analysis"]["lines"]}
        assert body["analysis"]["legal_cut_count"] == 2
        assert lines["L1"]["texts"] == ["甲乙", "甲乙X"]
        assert lines["L2"]["texts"] == ["X丙丁", "丙丁"]
        assert body["analysis"]["manifest"] == {
            "items": [], "empty_lines": []}

        # 下载：只有确定项（这里没有），meta 带歧义计数
        d = client.get(f"/api/sessions/{sid}/line-manifest/download")
        assert d.status_code == 200
        assert "attachment" in d.headers["content-disposition"]
        data = d.json()
        assert data["items"] == []
        assert data["meta"]["revision"] == rev
        assert data["meta"]["ambiguous_count"] == 2
        assert data["meta"]["legal_cut_count"] == 2

    def test_certain_and_empty_lines(self, client):
        # 第二行源字符全删 -> 确定空行；第一行文字确定
        sid, rev = create_confirmed(client, "甲乙丙丁", "甲乙", [])
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("甲乙丙丁", [
                                  {"id": "L1", "start": 0, "end": 2},
                                  {"id": "L2", "start": 2, "end": 4}])})
        assert r.status_code == 200
        data = r.json()["analysis"]
        assert data["manifest"]["items"] == [{"id": "L1", "text": "甲乙"}]
        assert data["manifest"]["empty_lines"] == ["L2"]
        d = client.get(f"/api/sessions/{sid}/line-manifest/download").json()
        assert d["items"] == [{"id": "L1", "text": "甲乙"}]
        assert d["empty_lines"] == ["L2"]

    def test_entity_pins_ambiguity(self, client):
        # 跨边界实体把分界钉死，歧义消失
        st = client.post("/api/sessions", json={
            "original": "甲乙丙丁", "revised": "X甲乙丙丁",
            "labels": [{"id": "e", "start": 1, "end": 3}]}).json()
        sid, rev = st["session_id"], st["revision"]
        r = client.post(f"/api/sessions/{sid}/confirm", json={"revision": rev})
        assert r.status_code == 200
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("甲乙丙丁", [
                                  {"id": "L1", "start": 0, "end": 2},
                                  {"id": "L2", "start": 2, "end": 4}])})
        assert r.status_code == 200
        data = r.json()["analysis"]
        assert data["legal_cut_count"] == 1
        assert [l["status"] for l in data["lines"]] == ["certain", "certain"]
        assert data["manifest"]["items"] == [
            {"id": "L1", "text": "X甲乙"}, {"id": "L2", "text": "丙丁"}]

    def test_state_includes_manifest(self, client):
        sid, rev = create_confirmed(client, "abc", "abc", [])
        assert client.get(f"/api/sessions/{sid}").json()["line_manifest"] is None
        client.post(f"/api/sessions/{sid}/line-manifest",
                    json={"revision": rev,
                          "payload": payload("abc", [["only", 0, 3]])})
        st = client.get(f"/api/sessions/{sid}").json()
        assert st["line_manifest"] is not None
        assert "/line-manifest/download?revision=" in st["manifest_download_url"]

    def test_reupload_is_idempotent_replace(self, client):
        sid, rev = create_confirmed(client, "abc", "abc", [])
        body = {"revision": rev,
                "payload": payload("abc", [["only", 0, 3]])}
        assert client.post(f"/api/sessions/{sid}/line-manifest",
                           json=body).status_code == 200
        assert client.post(f"/api/sessions/{sid}/line-manifest",
                           json=body).status_code == 200
        d = client.get(f"/api/sessions/{sid}/line-manifest/download").json()
        assert d["meta"]["line_count"] == 1

    def test_download_without_upload_404(self, client):
        sid, _ = create_confirmed(client, "abc", "abc", [])
        r = client.get(f"/api/sessions/{sid}/line-manifest/download")
        assert r.status_code == 404


class TestStaleness:
    def test_stale_revision_upload_rejected(self, client):
        sid, rev = create_confirmed(client, "甲乙丙丁", "甲乙X丙丁", [])
        # 另一窗口改版
        r = client.put(f"/api/sessions/{sid}/inputs", json={
            "original": "甲乙丙丁", "revised": "甲乙丙丁", "labels": []})
        assert r.status_code == 200
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": rev,
                              "payload": payload("甲乙丙丁", [
                                  ["L1", 0, 2], ["L2", 2, 4]])})
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "stale_revision"

    def test_manifest_removed_after_input_change(self, client):
        sid, rev = create_confirmed(client, "abc", "abc", [])
        client.post(f"/api/sessions/{sid}/line-manifest",
                    json={"revision": rev,
                          "payload": payload("abc", [["only", 0, 3]])})
        assert client.get(
            f"/api/sessions/{sid}/line-manifest/download").status_code == 200
        client.put(f"/api/sessions/{sid}/inputs", json={
            "original": "abc", "revised": "ab", "labels": []})
        # 旧清单随 revision 一起删除
        assert client.get(
            f"/api/sessions/{sid}/line-manifest/download").status_code == 404
        # 显式携带旧 revision 下载 -> 409
        r = client.get(
            f"/api/sessions/{sid}/line-manifest/download?revision={rev}")
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "stale_revision"

    def test_regenerate_on_new_revision(self, client):
        sid, rev = create_confirmed(client, "abc", "abc", [])
        client.post(f"/api/sessions/{sid}/line-manifest",
                    json={"revision": rev,
                          "payload": payload("abc", [["only", 0, 3]])})
        client.put(f"/api/sessions/{sid}/inputs", json={
            "original": "abc", "revised": "ab", "labels": []})
        st = client.get(f"/api/sessions/{sid}").json()
        new_rev = st["revision"]
        # 新版本未确认 -> 拒绝
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": new_rev,
                              "payload": payload("abc", [["only", 0, 3]])})
        assert r.status_code == 400
        # 重新确认后可生成
        client.post(f"/api/sessions/{sid}/confirm",
                    json={"revision": new_rev})
        r = client.post(f"/api/sessions/{sid}/line-manifest",
                        json={"revision": new_rev,
                              "payload": payload("abc", [["only", 0, 3]])})
        assert r.status_code == 200
        assert client.get(
            f"/api/sessions/{sid}/line-manifest/download").status_code == 200

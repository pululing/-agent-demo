"""
Lumina Wealth (澜景财富) 多 Agent 智能投顾服务网关.

支持 FastAPI + Uvicorn 服务模式、内置 Python HTTPServer 降级模式与 CLI 终端自检模式。
"""

from __future__ import annotations
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

import agents
import storage
import tools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
DOCKER_STATUS: Dict[str, Any] = {}


def init_backends() -> Dict[str, Any]:
    global DOCKER_STATUS
    if DOCKER_STATUS:
        return DOCKER_STATUS

    redis_ok = storage.redis_ping()

    pg_ok = False
    try:
        seed_users = [tools.get_user_account("U_10086"), tools.get_user_account("U_88888")]
        pg_ok = storage._run_async(storage._init_pg_tables_async(seed_users))
    except Exception:
        pg_ok = False

    milvus_info: Dict[str, Any] = {"ready": False}
    try:
        embedder = tools.LocalBGEM3Engine.get_instance()
        seeded = storage.MilvusVectorStore.ensure_collections_seeded(
            products=tools.PRODUCT_UNIVERSE,
            regulations=tools.FINANCE_REGULATION_CLAUSES,
            embedder=embedder,
        )
        milvus_info = {"ready": True, **seeded}
    except Exception as e:
        milvus_info = {"ready": False, "error": str(e)}

    DOCKER_STATUS = {
        "postgres": {
            "connected": pg_ok,
            "endpoint": f"{storage.CFG['DB_HOST']}:{storage.CFG['DB_PORT']}/{storage.CFG['DB_NAME']}",
            "tables": ["lumina_checkpoints", "lumina_user_profiles", "lumina_audit_logs"],
        },
        "milvus": {
            "connected": milvus_info.get("ready", False),
            "endpoint": f"{storage.CFG['MILVUS_HOST']}:{storage.CFG['MILVUS_PORT']}",
            "collections": [storage.PRODUCT_COLL, storage.REG_COLL],
            "details": milvus_info,
        },
        "redis": {
            "connected": redis_ok,
            "endpoint": f"{storage.CFG['REDIS_HOST']}:{storage.CFG['REDIS_PORT']}",
        },
    }
    return DOCKER_STATUS


def process_api_payload(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    init_backends()
    thread_id = payload.get("thread_id", "T_20261104_01")
    user_id = payload.get("user_id", "U_10086")

    if path == "/api/chat":
        query = payload.get("query", "最近大盘怎么样")
        scenario_key = payload.get("scenario_key", "")
        return agents.handle_user_request(
            thread_id=thread_id,
            user_id=user_id,
            query=query,
            scenario_key=scenario_key,
        )

    if path == "/api/resume":
        target_agent = payload.get("agent", "RiskAgent")
        action = payload.get("action", "confirm")
        note = payload.get("note", "")
        if target_agent == "ReportAgent":
            res = agents.resume_report_agent(thread_id=thread_id, action=action, advisor_comment=note)
        else:
            res = agents.resume_risk_agent(thread_id=thread_id, action=action, reassessment_note=note)
        res["memory_snapshot"] = agents.get_dual_memory_snapshot(thread_id, user_id)
        return res

    if path == "/api/state":
        engine = tools.LocalBGEM3Engine.get_instance()
        return {
            "bge_m3_configured": bool(tools.BGE_M3_MODEL_PATH),
            "bge_m3_mode": engine.backend_mode,
            "backends": DOCKER_STATUS,
            "memory_snapshot": agents.get_dual_memory_snapshot(thread_id, user_id),
        }

    return {"error": f"Unknown endpoint: {path}"}


def run_cli_selftest() -> None:
    print("=" * 80)
    print("Lumina Wealth · 多 Agent 智能投顾全链路自检 (CLI)")
    print("=" * 80)

    print("\n[Step 0] 初始化 PostgreSQL / Milvus / Redis 存储后端...")
    status = init_backends()
    print(f"  -> PostgreSQL ({status['postgres']['endpoint']}): connected={status['postgres']['connected']} | tables={status['postgres']['tables']}")
    print(f"  -> Milvus     ({status['milvus']['endpoint']}): connected={status['milvus']['connected']} | collections={status['milvus']['collections']}")
    print(f"  -> Redis      ({status['redis']['endpoint']}): connected={status['redis']['connected']}")

    print("\n[Step 1] 校验 BGE-M3 + Milvus 双路混合检索 (WeightedRanker 0.75:0.25)...")
    kb_test = tools.search_product_kb("低回撤稳健纯债与红利平衡基金", max_risk_code="R2")
    print(f"  -> Engine: {kb_test['tool_layer']}")
    print(f"  -> Dense preview: {kb_test['query_dense_preview']} | Sparse tokens: {kb_test['query_sparse_tokens']}")
    for c in kb_test["candidates"]:
        print(f"  -> Recalled: {c['name']} ({c['risk_level']}) | score={c['bge_m3_score']}")

    print("\n" + "-" * 80)
    print("[Flow 1] 市场行情问答 (Redis 缓存 + 结构化行情接口，跳过向量检索)")
    r1 = agents.handle_user_request("T_CLI_01", "U_10086", "最近大盘怎么样")
    for step in r1["trace"]:
        print(f"  [{step['agent']} · {step['node_name']}] vector={step['vector_used']} | {step['detail']}")

    print("\n" + "-" * 80)
    print("[Flow 2] 50万资产配置 + 节点 P7 适当性合规拦截 (P7 -> P3 重选)")
    r2 = agents.handle_user_request("T_CLI_02", "U_10086", "我有50万闲钱，稳健型，帮我做个资产配置")
    for step in r2["trace"]:
        loop_tag = f" [LOOP: {step['branch_loop']}]" if step.get("branch_loop") else ""
        print(f"  [{step['agent']} · {step['node_name']}]{loop_tag} ({step['latency']}) -> {step['detail']}")

    print("\n" + "-" * 80)
    print("[Flow 3] 风险测评 R8 断点暂停 -> 确认恢复 -> 自动串联资产配置")
    r3 = agents.handle_user_request("T_CLI_03", "U_10086", "从风险评估到资产配置全流程走一遍", scenario_key="pipeline_full")
    print(f"  -> Status: {r3['status']} @ {r3.get('interrupted_node')} ({r3.get('interrupt_title')})")
    r3_resume = agents.resume_risk_agent("T_CLI_03", action="confirm")
    print(f"  -> Resumed status: {r3_resume['status']} | total trace steps: {len(r3_resume['trace'])}")
    snap_after = agents.get_dual_memory_snapshot("T_CLI_03", "U_10086")
    print(f"  -> Long-term profile updated: risk={snap_after['long_term_profile']['data']['风险等级']}")


try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="Lumina Wealth - 智能投顾多 Agent 协同平台", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def serve_index():
        return FileResponse(INDEX_HTML_PATH)

    @app.get("/assets/{filename}")
    async def serve_asset(filename: str):
        asset_path = os.path.join(BASE_DIR, "assets", filename)
        if os.path.exists(asset_path):
            return FileResponse(asset_path)
        return JSONResponse({"error": "asset not found"}, status_code=404)

    @app.post("/api/chat")
    async def api_chat(req: Request):
        body = await req.json()
        return JSONResponse(process_api_payload("/api/chat", body))

    @app.post("/api/resume")
    async def api_resume(req: Request):
        body = await req.json()
        return JSONResponse(process_api_payload("/api/resume", body))

    @app.get("/api/state")
    async def api_state(thread_id: str = "T_20261104_01", user_id: str = "U_10086"):
        return JSONResponse(process_api_payload("/api/state", {"thread_id": thread_id, "user_id": user_id}))

    HAS_FASTAPI = True
except ImportError:
    app = None
    HAS_FASTAPI = False


class FallbackHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            self._send_json(process_api_payload("/api/state", {}))
            return
        if self.path.startswith("/assets/"):
            fname = os.path.basename(self.path.split("?")[0])
            asset_path = os.path.join(BASE_DIR, "assets", fname)
            if os.path.exists(asset_path):
                with open(asset_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
        if os.path.exists(INDEX_HTML_PATH):
            with open(INDEX_HTML_PATH, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send_json({"error": "index.html not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        payload = json.loads(raw)
        self._send_json(process_api_payload(self.path, payload))

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli_selftest()
        sys.exit(0)

    port = int(os.environ.get("PORT", "8001"))
    if HAS_FASTAPI:
        import uvicorn
        print(f"[SERVER] Lumina Wealth FastAPI server running on http://127.0.0.1:{port}")
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    else:
        print(f"[SERVER] Lumina Wealth HTTP server running on http://127.0.0.1:{port}")
        server = HTTPServer(("127.0.0.1", port), FallbackHTTPHandler)
        server.serve_forever()

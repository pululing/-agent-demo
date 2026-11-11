"""
Lumina Wealth Storage Layer (PostgreSQL + Milvus + Redis).

- Milvus: BGE-M3 1024-dim dense + sparse hybrid search with risk_rank scalar pre-filtering.
- PostgreSQL: thread-scoped checkpoints, user-scoped risk profiles, and 5-year compliance audit logs.
- Redis: lightweight RESP TCP client for intent routing and market quote caching.
"""

from __future__ import annotations
import asyncio
import json
import os
import socket
import time
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env_config() -> Dict[str, str]:
    """Load runtime config from defaults, optional .env/.env.local files, and process env."""
    cfg = {
        "DB_HOST": "127.0.0.1",
        "DB_PORT": "5432",
        "DB_NAME": "lumina_wealth",
        "DB_USER": "postgres",
        "DB_PASSWORD": "",
        "MILVUS_HOST": "127.0.0.1",
        "MILVUS_PORT": "19530",
        "MILVUS_PRODUCT_COLLECTION": "lumina_products_kb",
        "MILVUS_REG_COLLECTION": "lumina_regulations_kb",
        "REDIS_HOST": "127.0.0.1",
        "REDIS_PORT": "6379",
        "BGE_M3_MODEL_PATH": "",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY": "",
        "DEEPSEEK_MODEL": "deepseek-chat",
    }
    for env_name in (".env", ".env.local"):
        env_path = os.path.join(BASE_DIR, env_name)
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    ln = line.strip()
                    if not ln or ln.startswith("#") or "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    cfg[k.strip()] = v.strip()

    for key in list(cfg.keys()):
        if key in os.environ and os.environ[key] != "":
            cfg[key] = os.environ[key]
    return cfg


CFG = load_env_config()
PG_DSN = (
    f"postgresql://{CFG['DB_USER']}:{CFG['DB_PASSWORD']}"
    f"@{CFG['DB_HOST']}:{CFG['DB_PORT']}/{CFG['DB_NAME']}"
)
MILVUS_URI = f"http://{CFG['MILVUS_HOST']}:{CFG['MILVUS_PORT']}"
PRODUCT_COLL = CFG["MILVUS_PRODUCT_COLLECTION"]
REG_COLL = CFG["MILVUS_REG_COLLECTION"]


# Redis client (raw RESP over TCP)

def _redis_cmd(*args: str) -> Optional[str]:
    try:
        host = CFG["REDIS_HOST"]
        port = int(CFG["REDIS_PORT"])
        chunks = [f"*{len(args)}\r\n".encode("utf-8")]
        for a in args:
            b = str(a).encode("utf-8")
            chunks.append(f"${len(b)}\r\n".encode("utf-8") + b + b"\r\n")
        payload = b"".join(chunks)

        with socket.create_connection((host, port), timeout=1.5) as sock:
            sock.sendall(payload)
            first = sock.recv(4096)
            if not first:
                return None
            prefix = chr(first[0])
            if prefix == "+":
                return first[1:].decode("utf-8", errors="ignore").strip()
            if prefix == "$":
                header, _, rest = first.partition(b"\r\n")
                length = int(header[1:])
                if length < 0:
                    return None
                while len(rest) < length + 2:
                    more = sock.recv(4096)
                    if not more:
                        break
                    rest += more
                return rest[:length].decode("utf-8", errors="ignore")
            return None
    except Exception:
        return None


def redis_ping() -> bool:
    return _redis_cmd("PING") == "PONG"


def redis_get_json(key: str) -> Optional[Dict[str, Any]]:
    raw = _redis_cmd("GET", key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def redis_set_json(key: str, val: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
    raw = json.dumps(val, ensure_ascii=False)
    return _redis_cmd("SETEX", key, str(ttl_seconds), raw) == "OK"


# PostgreSQL persistence (checkpoints, user profiles, audit logs)

def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=8)
    return asyncio.run(coro)


async def _init_pg_tables_async(seed_users: List[Dict[str, Any]]) -> bool:
    import asyncpg
    conn = await asyncpg.connect(PG_DSN, timeout=4)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lumina_checkpoints (
                thread_id VARCHAR(128) PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL,
                agent VARCHAR(64),
                interrupted_at VARCHAR(32),
                state_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS lumina_user_profiles (
                user_id VARCHAR(64) PRIMARY KEY,
                cust_name VARCHAR(64) NOT NULL,
                age INT NOT NULL,
                investor_type VARCHAR(32) NOT NULL,
                risk_code VARCHAR(16) NOT NULL,
                risk_level VARCHAR(32) NOT NULL,
                risk_score NUMERIC(5, 1) DEFAULT 58.5,
                acct_bal_yuan NUMERIC(16, 2) NOT NULL,
                profile_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS lumina_audit_logs (
                id SERIAL PRIMARY KEY,
                record_id VARCHAR(96) NOT NULL,
                user_id VARCHAR(64) NOT NULL,
                event_type VARCHAR(64) NOT NULL,
                detail_json JSONB NOT NULL,
                retention_years INT DEFAULT 5,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        for u in seed_users:
            await conn.execute(
                """
                INSERT INTO lumina_user_profiles
                    (user_id, cust_name, age, investor_type, risk_code, risk_level, risk_score, acct_bal_yuan, profile_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                ON CONFLICT (user_id) DO NOTHING;
                """,
                u["user_id"],
                u["用户姓名"],
                int(u["年龄"]),
                u["投资者类型"],
                u["risk_code"],
                u["风险等级"],
                74.5 if u["user_id"] == "U_88888" else 58.5,
                float(u["账户总资产_数值"]),
                json.dumps(u, ensure_ascii=False),
            )
        return True
    finally:
        await conn.close()


async def _pg_save_checkpoint_async(thread_id: str, user_id: str, state: Dict[str, Any]) -> bool:
    import asyncpg
    # 剔除不可序列化的字段副本
    safe_state = json.dumps(state, ensure_ascii=False, default=str)
    conn = await asyncpg.connect(PG_DSN, timeout=3)
    try:
        await conn.execute(
            """
            INSERT INTO lumina_checkpoints (thread_id, user_id, status, agent, interrupted_at, state_json, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
            ON CONFLICT (thread_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                status = EXCLUDED.status,
                agent = EXCLUDED.agent,
                interrupted_at = EXCLUDED.interrupted_at,
                state_json = EXCLUDED.state_json,
                updated_at = NOW();
            """,
            thread_id,
            user_id,
            state.get("status", "IDLE"),
            state.get("agent") or state.get("last_agent") or "",
            state.get("interrupted_at"),
            safe_state,
        )
        return True
    finally:
        await conn.close()


async def _pg_load_checkpoint_async(thread_id: str) -> Optional[Dict[str, Any]]:
    import asyncpg
    conn = await asyncpg.connect(PG_DSN, timeout=3)
    try:
        row = await conn.fetchrow(
            "SELECT state_json, updated_at FROM lumina_checkpoints WHERE thread_id = $1;",
            thread_id,
        )
        if not row:
            return None
        data = json.loads(row["state_json"])
        data["pg_updated_at"] = str(row["updated_at"])[:19]
        return data
    finally:
        await conn.close()


async def _pg_save_user_profile_async(
    user_id: str,
    risk_code: str,
    risk_level: str,
    score: float,
    profile_dict: Dict[str, Any],
    record_id: str,
) -> bool:
    import asyncpg
    conn = await asyncpg.connect(PG_DSN, timeout=3)
    try:
        await conn.execute(
            """
            UPDATE lumina_user_profiles
            SET risk_code = $2,
                risk_level = $3,
                risk_score = $4,
                profile_json = $5::jsonb,
                updated_at = NOW()
            WHERE user_id = $1;
            """,
            user_id,
            risk_code,
            risk_level,
            float(score),
            json.dumps(profile_dict, ensure_ascii=False),
        )
        await conn.execute(
            """
            INSERT INTO lumina_audit_logs (record_id, user_id, event_type, detail_json, retention_years)
            VALUES ($1, $2, 'RISK_ASSESSMENT_CONFIRMED', $3::jsonb, 5);
            """,
            record_id,
            user_id,
            json.dumps({"risk_code": risk_code, "risk_level": risk_level, "score": score}, ensure_ascii=False),
        )
        return True
    finally:
        await conn.close()


async def _pg_load_user_profile_async(user_id: str) -> Optional[Dict[str, Any]]:
    import asyncpg
    conn = await asyncpg.connect(PG_DSN, timeout=3)
    try:
        row = await conn.fetchrow(
            "SELECT risk_code, risk_level, risk_score, profile_json, updated_at FROM lumina_user_profiles WHERE user_id = $1;",
            user_id,
        )
        if not row:
            return None
        prof = json.loads(row["profile_json"])
        prof["risk_code"] = row["risk_code"]
        prof["风险等级"] = row["risk_level"]
        prof["risk_score"] = float(row["risk_score"])
        prof["pg_updated_at"] = str(row["updated_at"])[:19]
        return prof
    finally:
        await conn.close()


def pg_save_checkpoint(thread_id: str, user_id: str, state: Dict[str, Any]) -> bool:
    try:
        return _run_async(_pg_save_checkpoint_async(thread_id, user_id, state))
    except Exception:
        return False


def pg_load_checkpoint(thread_id: str) -> Optional[Dict[str, Any]]:
    try:
        return _run_async(_pg_load_checkpoint_async(thread_id))
    except Exception:
        return None


def pg_save_user_profile(
    user_id: str,
    risk_code: str,
    risk_level: str,
    score: float,
    profile_dict: Dict[str, Any],
    record_id: str,
) -> bool:
    try:
        return _run_async(_pg_save_user_profile_async(user_id, risk_code, risk_level, score, profile_dict, record_id))
    except Exception:
        return False


def pg_load_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        return _run_async(_pg_load_user_profile_async(user_id))
    except Exception:
        return None


# Milvus hybrid vector store (dense + sparse retrieval)

class MilvusVectorStore:
    _client = None
    _ready = False

    @classmethod
    def get_client(cls):
        if cls._client is None:
            from pymilvus import MilvusClient
            cls._client = MilvusClient(uri=MILVUS_URI)
        return cls._client

    @classmethod
    def ensure_collections_seeded(
        cls,
        products: List[Dict[str, Any]],
        regulations: List[Dict[str, Any]],
        embedder,
    ) -> Dict[str, Any]:
        client = cls.get_client()
        from pymilvus import DataType

        # Initialize product hybrid vector collection
        if not client.has_collection(PRODUCT_COLL):
            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("risk_rank", DataType.INT64)
            schema.add_field("risk_level", DataType.VARCHAR, max_length=16)
            schema.add_field("category", DataType.VARCHAR, max_length=64)
            schema.add_field("name", DataType.VARCHAR, max_length=128)
            schema.add_field("description", DataType.VARCHAR, max_length=1024)

            ip = client.prepare_index_params()
            ip.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 128})
            ip.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.1})
            ip.add_index(field_name="risk_rank", index_type="STL_SORT")
            ip.add_index(field_name="risk_level", index_type="INVERTED")

            client.create_collection(collection_name=PRODUCT_COLL, schema=schema, index_params=ip)

        client.load_collection(PRODUCT_COLL)
        rows = []
        rank_map = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
        for p in products:
            dense, sparse = embedder.encode_single(p["description"])
            rows.append({
                "id": p["product_id"],
                "embedding": dense,
                "sparse_embedding": sparse if sparse else {1: 0.01},
                "risk_rank": rank_map.get(p["risk_level"], 2),
                "risk_level": p["risk_level"],
                "category": p["category"],
                "name": p["name"],
                "description": p["description"],
                "annual_return_3y": float(p["annual_return_3y"]),
                "max_drawdown": float(p["max_drawdown"]),
                "sharpe": float(p["sharpe"]),
                "min_buy": int(p["min_buy"]),
                "tags": p["tags"],
            })
        client.upsert(collection_name=PRODUCT_COLL, data=rows)

        # Initialize regulation hybrid vector collection
        if not client.has_collection(REG_COLL):
            schema_r = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema_r.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema_r.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)
            schema_r.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
            schema_r.add_field("title", DataType.VARCHAR, max_length=256)
            schema_r.add_field("text", DataType.VARCHAR, max_length=2048)

            ip_r = client.prepare_index_params()
            ip_r.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 128})
            ip_r.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.1})
            client.create_collection(collection_name=REG_COLL, schema=schema_r, index_params=ip_r)

        client.load_collection(REG_COLL)
        reg_rows = []
        for c in regulations:
            dense, sparse = embedder.encode_single(f"{c['title']} {c['text']}")
            reg_rows.append({
                "id": c["clause_id"],
                "embedding": dense,
                "sparse_embedding": sparse if sparse else {1: 0.01},
                "title": c["title"],
                "text": c["text"],
            })
        client.upsert(collection_name=REG_COLL, data=reg_rows)
        cls._ready = True
        return {
            "milvus_uri": MILVUS_URI,
            "product_collection": PRODUCT_COLL,
            "product_count": len(rows),
            "reg_collection": REG_COLL,
            "reg_count": len(reg_rows),
        }

    @classmethod
    def hybrid_search_products(
        cls,
        q_dense: List[float],
        q_sparse: Dict[int, float],
        max_risk_rank: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        from pymilvus import AnnSearchRequest, WeightedRanker
        client = cls.get_client()
        filter_expr = f"risk_rank <= {max_risk_rank}"

        dense_req = AnnSearchRequest(
            data=[q_dense],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[q_sparse if q_sparse else {1: 0.01}],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k,
            expr=filter_expr,
        )
        res = client.hybrid_search(
            collection_name=PRODUCT_COLL,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(0.75, 0.25),
            limit=top_k,
            output_fields=["id", "name", "category", "risk_level", "description", "annual_return_3y", "max_drawdown", "sharpe", "min_buy", "tags"],
        )
        items: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent = hit["entity"]
            items.append({
                "product_id": ent.get("id"),
                "name": ent.get("name"),
                "category": ent.get("category"),
                "risk_level": ent.get("risk_level"),
                "description": ent.get("description"),
                "annual_return_3y": ent.get("annual_return_3y"),
                "max_drawdown": ent.get("max_drawdown"),
                "sharpe": ent.get("sharpe"),
                "min_buy": ent.get("min_buy"),
                "tags": ent.get("tags", []),
                "bge_m3_score": round(float(hit.get("distance", 0.0)), 4),
            })
        return items

    @classmethod
    def hybrid_search_regulations(
        cls,
        q_dense: List[float],
        q_sparse: Dict[int, float],
        top_k: int = 2,
    ) -> List[str]:
        from pymilvus import AnnSearchRequest, WeightedRanker
        client = cls.get_client()
        dense_req = AnnSearchRequest(
            data=[q_dense],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 32}},
            limit=top_k,
        )
        sparse_req = AnnSearchRequest(
            data=[q_sparse if q_sparse else {1: 0.01}],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k,
        )
        res = client.hybrid_search(
            collection_name=REG_COLL,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(0.75, 0.25),
            limit=top_k,
            output_fields=["title", "text"],
        )
        clauses: List[str] = []
        for hit in res[0]:
            ent = hit["entity"]
            score = round(float(hit.get("distance", 0.0)), 4)
            clauses.append(f"{ent.get('title')}（Milvus Hybrid 得分 {score}）：{ent.get('text')}")
        return clauses

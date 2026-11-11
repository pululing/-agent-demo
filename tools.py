"""
Lumina Wealth Decoupled Tool Layer (19 tools across 4 layers).

1. Deterministic calculation & compliance rule engines (<10ms).
2. Market quote, macro database, and BGE-M3 hybrid vector search tools.
3. Account, product catalog, order, holdings, and profile persistence APIs.
4. LLM semantic profiling, commentary, matching, and attribution tools.
"""

from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import storage


BGE_M3_MODEL_PATH = (
    storage.CFG.get("BGE_M3_MODEL_PATH")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-m3")
)


def load_llm_config() -> Dict[str, str]:
    """Load LLM endpoint, credentials, and model name from environment / local config."""
    base_url = (storage.CFG.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    api_key = (storage.CFG.get("DEEPSEEK_API_KEY") or "").strip()
    model_name = (storage.CFG.get("DEEPSEEK_MODEL") or "deepseek-chat").strip()
    api_model = "deepseek-flash" if "flash" in model_name.lower() else model_name
    return {
        "base_url": base_url,
        "api_key": api_key,
        "display_model": model_name,
        "api_model": api_model,
    }


LLM_CONFIG = load_llm_config()


def call_deepseek_llm(
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
    json_mode: bool = False,
    max_tokens: int = 320,
    timeout: float = 12.0,
) -> Optional[str]:
    """Call OpenAI-compatible chat completions endpoint; returns None if API key is unset or unavailable."""
    cfg = load_llm_config()
    if not cfg["api_key"]:
        return None

    url = f"{cfg['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    candidates = [cfg["api_model"]]
    if cfg["display_model"] != cfg["api_model"]:
        candidates.append(cfg["display_model"])

    for model_id in candidates:
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if not enable_thinking:
            payload["thinking"] = {"type": "disabled"}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if content:
                    return content
        except Exception:
            continue
    return None


# Local BGE-M3 dense + sparse embedding engine


class LocalBGEM3Engine:
    """BGE-M3 encoder producing 1024-dim normalized dense vectors and sparse lexical weights."""

    _instance: Optional["LocalBGEM3Engine"] = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
        self._sparse_linear = None
        self._device = "cpu"
        self.backend_mode = "uninitialized"
        self._corpus_cache: Dict[str, Tuple[List[float], Dict[int, float]]] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None or self.backend_mode == "fallback":
            return
        if not os.path.exists(self.model_path):
            self.backend_mode = "fallback"
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModel.from_pretrained(self.model_path).to(self._device)
            self._model.eval()

            # 加载 BGE-M3 稀疏线性层权重 sparse_linear.pt
            sparse_pt = os.path.join(self.model_path, "sparse_linear.pt")
            if os.path.exists(sparse_pt):
                self._sparse_linear = torch.nn.Linear(1024, 1).to(self._device)
                state_dict = torch.load(sparse_pt, map_location=self._device, weights_only=True)
                self._sparse_linear.load_state_dict(state_dict)
                self._sparse_linear.eval()

            self.backend_mode = f"本地 BGE-M3 真实权重 ({self._device.upper()}, 1024-dim Dense + Sparse)"
        except Exception:
            self.backend_mode = "fallback"

    @classmethod
    def get_instance(cls) -> "LocalBGEM3Engine":
        if cls._instance is None:
            cls._instance = LocalBGEM3Engine(BGE_M3_MODEL_PATH)
        return cls._instance

    def encode_single(self, text: str) -> Tuple[List[float], Dict[int, float]]:
        if text in self._corpus_cache:
            return self._corpus_cache[text]

        self._ensure_loaded()
        if self._model is not None and self._tokenizer is not None:
            import torch

            inputs = self._tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                out = self._model(**inputs)
                hidden = out.last_hidden_state
                # BGE-M3 CLS 向量 L2 归一化
                dense_tensor = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=-1)
                dense_vec = [round(float(x), 6) for x in dense_tensor[0].cpu().tolist()]

                sparse_vec: Dict[int, float] = {}
                if self._sparse_linear is not None:
                    token_weights = torch.relu(self._sparse_linear(hidden)).squeeze(-1)[0]
                    input_ids = inputs["input_ids"][0].cpu().tolist()
                    weights_list = token_weights.cpu().tolist()
                    special_ids = set(self._tokenizer.all_special_ids)
                    for tid, w in zip(input_ids, weights_list):
                        if tid not in special_ids and w > 0:
                            sparse_vec[int(tid)] = max(sparse_vec.get(int(tid), 0.0), round(float(w), 4))

            self._corpus_cache[text] = (dense_vec, sparse_vec)
            return dense_vec, sparse_vec

        # 轻量回退 (未安装 torch 时保证脚本可运行)
        raw = [math.sin(i * 0.13 + len(text)) for i in range(1024)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        dense_vec = [round(x / norm, 6) for x in raw]
        sparse_vec = {ord(ch): 0.25 for ch in text[:12]}
        self._corpus_cache[text] = (dense_vec, sparse_vec)
        return dense_vec, sparse_vec

    @staticmethod
    def hybrid_score(
        q_dense: List[float],
        q_sparse: Dict[int, float],
        d_dense: List[float],
        d_sparse: Dict[int, float],
        dense_weight: float = 0.75,
        sparse_weight: float = 0.25,
    ) -> float:
        dense_sim = sum(a * b for a, b in zip(q_dense, d_dense))
        sparse_sim = sum(v * d_sparse.get(k, 0.0) for k, v in q_sparse.items())
        sparse_norm = min(1.0, sparse_sim / 1.5) if sparse_sim > 0 else 0.0
        return round(dense_weight * dense_sim + sparse_weight * sparse_norm, 4)


# Sample customer account profiles & fund product catalog (amounts in cents internally)

RAW_MIDDLE_PLATFORM_USERS: Dict[str, Dict[str, Any]] = {
    "U_10086": {
        "uid": "U_10086",
        "cust_name": "林知远",
        "age": 32,
        "invst_type": 1,          # 1=普通投资者, 2=高净值客户
        "risk_lvl": 2,            # 1=R1谨慎型, 2=R2稳健型, 3=R3平衡型, 4=R4进取型, 5=R5激进型
        "acct_bal": 50000000,     # 500,000.00 元 (底层按分存储)
        "annual_inc": 30000000,   # 年收入 30万
        "debt_ratio": 0.28,       # 房贷收支比 28%
        "kyc_verified": True,
        "aml_Abnormal": False,
        "invest_exp_years": 2,
        "max_loss_tolerance": 0.10,
        "horizon_months": 36,
    },
    "U_88888": {
        "uid": "U_88888",
        "cust_name": "周明轩",
        "age": 45,
        "invst_type": 2,          # 2=高净值客户 (触发 ReportAgent 节点6 理财师人工审阅 interrupt)
        "risk_lvl": 3,            # 3=R3平衡型
        "acct_bal": 650000000,    # 650.00 万元
        "annual_inc": 180000000,  # 年收入 180万
        "debt_ratio": 0.12,
        "kyc_verified": True,
        "aml_Abnormal": False,
        "invest_exp_years": 9,
        "max_loss_tolerance": 0.15,
        "horizon_months": 60,
    },
}

INVESTOR_TYPE_MAP = {1: "普通投资者", 2: "高净值客户"}
RISK_LEVEL_MAP = {
    1: "R1-谨慎型",
    2: "R2-稳健型",
    3: "R3-平衡型",
    4: "R4-进取型",
    5: "R5-激进型",
}
RISK_RANK = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}

PRODUCT_UNIVERSE: List[Dict[str, Any]] = [
    {
        "product_id": "FD_101",
        "name": "招商安泽纯债 A",
        "category": "债券型基金",
        "risk_level": "R2",
        "annual_return_3y": 4.2,
        "max_drawdown": -1.8,
        "sharpe": 1.45,
        "min_buy": 100,
        "tags": ["纯债底仓", "低回撤", "机构重仓"],
        "description": "招商安泽纯债基金A，专注高等级国债与央企金融债，低回撤稳健纯债底仓产品。",
    },
    {
        "product_id": "FD_102",
        "name": "易方达稳健双利债券",
        "category": "债券型基金",
        "risk_level": "R2",
        "annual_return_3y": 4.6,
        "max_drawdown": -2.4,
        "sharpe": 1.32,
        "min_buy": 100,
        "tags": ["固收+", "稳健收益"],
        "description": "易方达稳健双利债券，以优质利率债打底辅以可转债增强，适合稳健型投资者。",
    },
    {
        "product_id": "FD_201",
        "name": "中欧红利智选平衡混合",
        "category": "混合型基金",
        "risk_level": "R2",
        "annual_return_3y": 6.8,
        "max_drawdown": -4.9,
        "sharpe": 1.18,
        "min_buy": 1000,
        "tags": ["高股息红利", "股债平衡"],
        "description": "中欧红利智选平衡混合基金，精选高股息央企红利资产与高等级信用债，股债平衡控制回撤。",
    },
    {
        "product_id": "FD_301",
        "name": "天弘云商宝货币 A",
        "category": "货币基金",
        "risk_level": "R1",
        "annual_return_3y": 2.1,
        "max_drawdown": -0.05,
        "sharpe": 2.10,
        "min_buy": 1,
        "tags": ["T+0高流动性", "现金管理"],
        "description": "天弘云商宝货币市场基金，投向高流动性银行大额存单与短期国债逆回购，T+0随时申赎。",
    },
    {
        "product_id": "FD_401",
        "name": "招行90天看涨鲨鱼鳍结构性存款",
        "category": "结构性存款",
        "risk_level": "R1",
        "annual_return_3y": 3.5,
        "max_drawdown": 0.0,
        "sharpe": 1.90,
        "min_buy": 10000,
        "tags": ["本金保障型", "90天锁定期"],
        "description": "招商银行90天结构性存款，本金100%纳入存款保险保障，挂钩沪深300指数浮动增厚收益。",
    },
    {
        "product_id": "FD_999",
        "name": "华夏科创板半导体锐进股票",
        "category": "股票型基金",
        "risk_level": "R4",
        "annual_return_3y": 14.5,
        "max_drawdown": -28.4,
        "sharpe": 0.76,
        "min_buy": 1000,
        "tags": ["高波动成长", "R4高风险"],
        "description": "华夏科创板半导体锐进股票型基金，高仓位集中配置科创板算力芯片，属于R4高波动进取型产品。",
    },
]

FINANCE_REGULATION_CLAUSES = [
    {
        "clause_id": "REG_16",
        "title": "《证券期货投资者适当性管理办法》第十六条",
        "text": "禁止向普通投资者主动推介风险等级高于其风险承受能力的产品或者服务。",
    },
    {
        "clause_id": "REG_23",
        "title": "《证券期货投资者适当性管理办法》第二十三条",
        "text": "向普通投资者销售产品或者提供服务前，应当告知风险评估结果、适当性匹配意见，并由投资者本人确认留痕。",
    },
]


# Layer 1: Deterministic Python calculation & compliance rules (<10ms)

def risk_level_calc(risk_dimensions: Dict[str, int], force_capital_preserve: bool = False) -> Dict[str, Any]:
    weights = {
        "收入稳定性": Decimal("0.20"),
        "投资经验": Decimal("0.15"),
        "亏损容忍度": Decimal("0.30"),
        "流动性需求": Decimal("0.15"),
        "投资期限": Decimal("0.20"),
    }
    weighted_sum = sum(
        Decimal(str(risk_dimensions.get(k, 5))) * w for k, w in weights.items()
    )
    score_100 = float((weighted_sum * Decimal("10")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

    if force_capital_preserve or risk_dimensions.get("亏损容忍度", 5) <= 2:
        code = "R1"
    elif score_100 < 35:
        code = "R1"
    elif score_100 < 65:
        code = "R2"
    elif score_100 < 78:
        code = "R3"
    elif score_100 < 88:
        code = "R4"
    else:
        code = "R5"

    label_map = {
        "R1": "R1-谨慎型",
        "R2": "R2-稳健型",
        "R3": "R3-平衡型",
        "R4": "R4-进取型",
        "R5": "R5-激进型",
    }
    return {
        "tool_layer": "Python 规则引擎 (<10ms)",
        "risk_code": code,
        "risk_level": label_map[code],
        "score": score_100,
        "dimensions": risk_dimensions,
    }


def asset_allocation_calc(
    risk_code: str,
    investment_amount: float,
    market_trend: str = "震荡温和修复",
    horizon_months: int = 36,
) -> Dict[str, Any]:
    base_templates = {
        "R1": {"债券型基金": Decimal("35"), "混合型基金": Decimal("0"), "货币基金": Decimal("40"), "结构性存款": Decimal("25")},
        "R2": {"债券型基金": Decimal("50"), "混合型基金": Decimal("25"), "货币基金": Decimal("15"), "结构性存款": Decimal("10")},
        "R3": {"债券型基金": Decimal("40"), "混合型基金": Decimal("40"), "货币基金": Decimal("10"), "结构性存款": Decimal("10")},
        "R4": {"债券型基金": Decimal("25"), "混合型基金": Decimal("60"), "货币基金": Decimal("10"), "结构性存款": Decimal("5")},
        "R5": {"债券型基金": Decimal("10"), "混合型基金": Decimal("80"), "货币基金": Decimal("10"), "结构性存款": Decimal("0")},
    }
    weights = dict(base_templates.get(risk_code, base_templates["R2"]))

    if horizon_months < 12 and weights["混合型基金"] >= Decimal("10"):
        weights["混合型基金"] -= Decimal("5")
        weights["货币基金"] += Decimal("5")

    total_pct = sum(weights.values())
    if total_pct != Decimal("100"):
        weights["货币基金"] += Decimal("100") - total_pct

    amt_dec = Decimal(str(investment_amount))
    allocation_detail = []
    for category, pct in weights.items():
        if pct <= 0:
            continue
        sub_amount = (amt_dec * pct / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        allocation_detail.append({
            "category": category,
            "ratio_pct": float(pct),
            "ratio_str": f"{int(pct)}%",
            "amount_yuan": float(sub_amount),
            "amount_wan": f"{float(sub_amount / Decimal('10000')):g}万",
        })

    return {
        "tool_layer": "Python 规则引擎 (Decimal 100% 强约束)",
        "risk_code": risk_code,
        "total_amount": investment_amount,
        "sum_check_pct": float(sum(weights.values())),
        "allocation": allocation_detail,
    }


def performance_calc(
    allocation: List[Dict[str, Any]],
    holdings: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if holdings:
        total_cost = sum(Decimal(str(h["cost"])) for h in holdings)
        total_val = sum(Decimal(str(h["market_value"])) for h in holdings)
        profit = total_val - total_cost
        total_ret_pct = float((profit / total_cost * Decimal("100")).quantize(Decimal("0.01")))
        return {
            "tool_layer": "Python 规则引擎 (Decimal/NumPy)",
            "total_cost": float(total_cost),
            "total_market_value": float(total_val),
            "total_profit": float(profit),
            "portfolio_return_pct": total_ret_pct,
            "benchmark_hs300_pct": 3.12,
            "excess_alpha_pct": round(total_ret_pct - 3.12, 2),
            "max_drawdown_pct": -3.45,
            "sharpe_ratio": 1.28,
        }

    return {
        "tool_layer": "Python 规则引擎 (NumPy 组合测算)",
        "expected_return_range": "4.5% - 5.2%",
        "expected_max_drawdown": "-3.0% 至 -4.8%",
        "sharpe_ratio": 1.24,
        "volatility_pct": 3.6,
    }


def compliance_check(
    user_risk_code: str,
    matched_products: List[Dict[str, Any]],
    advice_text: str = "",
    aml_flag: bool = False,
) -> Dict[str, Any]:
    user_rank = RISK_RANK.get(user_risk_code, 2)
    violations: List[str] = []

    for prod in matched_products:
        p_code = prod.get("risk_level", "R2")
        p_rank = RISK_RANK.get(p_code, 2)
        if p_rank > user_rank:
            violations.append(
                f"适当性超标拦截：产品《{prod['name']}》({p_code}) 超出客户风险承受等级 ({user_risk_code})"
            )

    if aml_flag:
        violations.append("反洗钱(AML)预警：检测到近期快进快出异常申赎特征，阻断自动下单")

    for word in ["保本无忧", "稳赚不赔", "保证年化", "零风险包赚"]:
        if word in advice_text:
            violations.append(f"合规违禁词拦截：检测到违规承诺表述“{word}”")

    return {
        "tool_layer": "Python 合规规则门禁 (<3ms)",
        "is_compliant": len(violations) == 0,
        "violations": violations,
        "mandatory_disclosure": "过往业绩不预示未来表现，市场有风险，投资需谨慎。（《证券期货投资者适当性管理办法》合规留痕已归档，保存期 5 年）",
    }


# Layer 2: Market data & hybrid vector knowledge-base tools


def query_market_data(symbol: str = "000300.SH", time_range: str = "近期") -> Dict[str, Any]:
    cache_key = f"lumina:market:{symbol}"
    cached = storage.redis_get_json(cache_key)
    if cached:
        cached["tool_layer"] = "Redis 5min Market Cache"
        return cached

    data = {
        "tool_layer": "Structured Market Data API + Redis Cache",
        "symbol": symbol,
        "time_range": time_range,
        "indices": [
            {"name": "沪深300 (000300.SH)", "close": "3,968.42", "change_5d": "+1.85%", "pe_percentile": "34% (历史中低位)"},
            {"name": "上证指数 (000001.SH)", "close": "3,312.18", "change_5d": "+1.42%", "turnover": "1.18 万亿元"},
            {"name": "中证全债 (H11001.CSI)", "close": "248.65", "change_5d": "+0.22%", "yield_10y": "2.12%"},
        ],
    }
    storage.redis_set_json(cache_key, data, ttl_seconds=300)
    return data


def query_macro_data(time_range: str = "本季度") -> Dict[str, Any]:
    return {
        "tool_layer": "Macroeconomic Indicator API",
        "time_range": time_range,
        "cpi_yoy": "+0.4%",
        "pmi_manufacturing": "50.2% (重回扩张区间)",
        "m2_yoy": "+7.5%",
        "bond_10y_yield": "2.12%",
    }


def search_product_kb(
    query_text: str,
    max_risk_code: str = "R2",
    inject_unsuitable_candidate: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = LocalBGEM3Engine.get_instance()
    q_dense, q_sparse = engine.encode_single(query_text)
    max_rank = RISK_RANK.get(max_risk_code, 2)

    milvus_used = False
    scored_candidates: List[Dict[str, Any]] = []
    try:
        if not storage.MilvusVectorStore._ready:
            storage.MilvusVectorStore.ensure_collections_seeded(
                PRODUCT_UNIVERSE, FINANCE_REGULATION_CLAUSES, engine
            )
        scored_candidates = storage.MilvusVectorStore.hybrid_search_products(
            q_dense=q_dense,
            q_sparse=q_sparse,
            max_risk_rank=max_rank,
            top_k=5,
        )
        milvus_used = len(scored_candidates) > 0
    except Exception:
        milvus_used = False

    if not scored_candidates:
        for p in PRODUCT_UNIVERSE:
            if RISK_RANK.get(p["risk_level"], 2) <= max_rank:
                d_dense, d_sparse = engine.encode_single(p["description"])
                sim = engine.hybrid_score(q_dense, q_sparse, d_dense, d_sparse)
                scored_candidates.append({**p, "bge_m3_score": sim})
        scored_candidates.sort(key=lambda x: x["bge_m3_score"], reverse=True)

    if inject_unsuitable_candidate:
        r4_prod = next(p for p in PRODUCT_UNIVERSE if p["risk_level"] == "R4")
        d_dense, d_sparse = engine.encode_single(r4_prod["description"])
        sim = engine.hybrid_score(q_dense, q_sparse, d_dense, d_sparse)
        scored_candidates = scored_candidates[:3] + [{**r4_prod, "bge_m3_score": sim}]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    milvus_tag = (
        f"Milvus ({storage.PRODUCT_COLL}) HybridSearch"
        if milvus_used
        else "In-Memory Hybrid Fallback"
    )
    return {
        "tool_layer": f"{engine.backend_mode} + {milvus_tag}",
        "query_dense_preview": [round(v, 4) for v in q_dense[:6]],
        "query_sparse_tokens": len(q_sparse),
        "scalar_filter": f"risk_rank <= {max_rank} ({max_risk_code})",
        "elapsed_ms": elapsed_ms,
        "recalled_count": len(scored_candidates),
        "candidates": scored_candidates,
    }


def query_finance_regulation(query: str) -> Dict[str, Any]:
    engine = LocalBGEM3Engine.get_instance()
    q_dense, q_sparse = engine.encode_single(query)
    scored: List[str] = []
    try:
        if not storage.MilvusVectorStore._ready:
            storage.MilvusVectorStore.ensure_collections_seeded(
                PRODUCT_UNIVERSE, FINANCE_REGULATION_CLAUSES, engine
            )
        scored = storage.MilvusVectorStore.hybrid_search_regulations(q_dense, q_sparse, top_k=2)
    except Exception:
        scored = []

    if not scored:
        for c in FINANCE_REGULATION_CLAUSES:
            d_dense, d_sparse = engine.encode_single(f"{c['title']} {c['text']}")
            sim = engine.hybrid_score(q_dense, q_sparse, d_dense, d_sparse)
            scored.append(f"{c['title']}（BGE-M3 混合得分 {sim}）：{c['text']}")

    return {
        "tool_layer": f"Milvus ({storage.REG_COLL}) + BGE-M3",
        "query": query,
        "clauses": scored,
    }


# Layer 3: Account, order, and holdings APIs

def get_user_account(user_id: str) -> Dict[str, Any]:
    raw = RAW_MIDDLE_PLATFORM_USERS.get(user_id, RAW_MIDDLE_PLATFORM_USERS["U_10086"])
    return {
        "tool_layer": "httpx 异步账户中台 API (含分转元与编码中文映射)",
        "user_id": raw["uid"],
        "用户姓名": raw["cust_name"],
        "年龄": raw["age"],
        "投资者类型": INVESTOR_TYPE_MAP[raw["invst_type"]],
        "风险等级": RISK_LEVEL_MAP[raw["risk_lvl"]],
        "risk_code": f"R{raw['risk_lvl']}",
        "账户总资产": f"{raw['acct_bal'] / 100:,.2f}元",
        "账户总资产_数值": raw["acct_bal"] / 100,
        "年收入": f"{raw['annual_inc'] / 1000000:g}万元",
        "房贷收支比": f"{int(raw['debt_ratio'] * 100)}%",
        "投资经验年限": f"{raw['invest_exp_years']}年",
        "最大亏损容忍度": f"{int(raw['max_loss_tolerance'] * 100)}%",
        "计划投资期限": f"{raw['horizon_months']}个月",
        "实名KYC认证": raw["kyc_verified"],
        "反洗钱异常标记": raw["aml_Abnormal"],
    }


def get_product_detail(product_ids: List[str]) -> List[Dict[str, Any]]:
    id_set = set(product_ids)
    return [p for p in PRODUCT_UNIVERSE if p["product_id"] in id_set]


def place_order(user_id: str, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"order_batch_id": "ORD_20261104_8821", "user_id": user_id, "status": "SUBMITTED", "count": len(orders)}


def get_portfolio_history(user_id: str) -> List[Dict[str, Any]]:
    if user_id == "U_88888":
        return [
            {"product_id": "FD_101", "name": "招商安泽纯债 A", "category": "债券型基金", "cost": 3000000, "market_value": 3138000, "return_pct": 4.60},
            {"product_id": "FD_201", "name": "中欧红利智选平衡混合", "category": "混合型基金", "cost": 2500000, "market_value": 2695000, "return_pct": 7.80},
            {"product_id": "FD_999", "name": "华夏科创板半导体锐进股票", "category": "股票型基金", "cost": 1000000, "market_value": 942000, "return_pct": -5.80},
        ]
    return [
        {"product_id": "FD_101", "name": "招商安泽纯债 A", "category": "债券型基金", "cost": 250000, "market_value": 260500, "return_pct": 4.20},
        {"product_id": "FD_201", "name": "中欧红利智选平衡混合", "category": "混合型基金", "cost": 150000, "market_value": 160200, "return_pct": 6.80},
        {"product_id": "FD_301", "name": "天弘云商宝货币 A", "category": "货币基金", "cost": 100000, "market_value": 102100, "return_pct": 2.10},
    ]


def get_return_detail(user_id: str) -> Dict[str, Any]:
    holdings = get_portfolio_history(user_id)
    return {"user_id": user_id, "holdings_count": len(holdings), "holdings": holdings}


def save_risk_assessment(user_id: str, risk_code: str, score: float, dimensions: Dict[str, int]) -> Dict[str, Any]:
    record_id = f"RA_{user_id}_20261104"
    acct = get_user_account(user_id)
    acct["risk_code"] = risk_code
    acct["风险等级"] = RISK_LEVEL_MAP.get(RISK_RANK.get(risk_code, 2), "R2-稳健型")
    acct["dimensions"] = dimensions
    pg_ok = storage.pg_save_user_profile(
        user_id=user_id,
        risk_code=risk_code,
        risk_level=acct["风险等级"],
        score=score,
        profile_dict=acct,
        record_id=record_id,
    )
    return {
        "tool_layer": "PostgreSQL Profile & Audit Store",
        "pg_persisted": pg_ok,
        "success": True,
        "record_id": record_id,
        "saved_risk_code": risk_code,
        "saved_score": score,
        "audit_retention_years": 5,
    }


# Layer 4: LLM semantic analysis tools

def llm_risk_profiling(account_info: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    if account_info.get("user_id") == "U_88888":
        dims = {"收入稳定性": 9, "投资经验": 8, "亏损容忍度": 6, "流动性需求": 7, "投资期限": 8}
    else:
        dims = {"收入稳定性": 8, "投资经验": 3, "亏损容忍度": 5, "流动性需求": 6, "投资期限": 7}

    llm_summary = call_deepseek_llm(
        system_prompt="你是专业的财富管理智能投顾风险评估助手。请用一句话总结客户财务与风险偏好特征，不超過70字。",
        user_prompt=f"客户信息：{account_info}，用户提问：{user_message}，五维打分：{dims}",
        enable_thinking=False,
        max_tokens=120,
    )
    return {
        "tool_layer": f"{LLM_CONFIG['display_model']} + Pydantic 强类型结构化提取",
        "dimensions": dims,
        "summary": llm_summary or (
            f"客户年收入 {account_info['年收入']}，房贷收支比 {account_info['房贷收支比']}，"
            f"可承受最大本金回撤 {account_info['最大亏损容忍度']}，计划投资期限 {account_info['计划投资期限']}。"
        ),
    }


def llm_market_analysis(market_data: Dict[str, Any], macro_data: Dict[str, Any]) -> Dict[str, Any]:
    llm_summary = call_deepseek_llm(
        system_prompt="你是专业财富管理宏观策略师。请根据传入的结构化行情与宏观数据，用2句话给出通俗精炼的市场研判，控制在120字以内。",
        user_prompt=f"实时行情数据：{json.dumps(market_data, ensure_ascii=False)}；宏观数据：{json.dumps(macro_data, ensure_ascii=False)}",
        enable_thinking=False,
        max_tokens=180,
    )
    return {
        "tool_layer": f"{LLM_CONFIG['display_model']} 宏观与市场定性研判",
        "market_trend": "震荡温和修复（结构性红利与稳健固收占优）",
        "summary": llm_summary or (
            "近期 A 股大盘呈现温和修复态势：沪深 300 近 5 日上涨 +1.85%（点位 3,968.42，PE 分位处于历史 34% 中低估值区间），"
            "上证指数上涨 +1.42%，两市成交额维持在 1.18 万亿元活跃水平；制造业 PMI 回升至 50.2% 扩张区间，10 年期国债收益率 2.12%。"
            "当前策略建议以纯债与固收+作为稳健底仓，辅以高股息红利平衡混合产品增厚收益。"
        ),
    }


def llm_product_match(
    allocation: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    r4_item = next((c for c in candidates if c["risk_level"] == "R4"), None)
    if r4_item:
        matched.append({
            **r4_item,
            "allocated_ratio": "25%",
            "allocated_amount": "12.5万",
            "reason": "高弹性科技成长方向（待合规审查校验）",
        })

    for item in allocation:
        cat = item["category"]
        prod = next((c for c in candidates if c["category"] == cat), None)
        if prod and not any(m["product_id"] == prod["product_id"] for m in matched):
            sim_tag = f"，BGE-M3 检索得分 {prod.get('bge_m3_score', 0.86)}" if "bge_m3_score" in prod else ""
            matched.append({
                **prod,
                "allocated_ratio": item["ratio_str"],
                "allocated_amount": item["amount_wan"],
                "reason": f"契合{cat}配置目标，近3年年化 {prod['annual_return_3y']}%，最大回撤 {prod['max_drawdown']}%{sim_tag}",
            })
    return matched


def llm_invest_advice(context_type: str, data: Dict[str, Any]) -> str:
    if context_type == "risk_report":
        llm_text = call_deepseek_llm(
            system_prompt="你是持牌智能投顾助手。请根据客户的五维风险评分结果生成一段通俗易懂的配置方向建议（不超过90字，严禁出现保本、稳赚词汇）。",
            user_prompt=f"风险等级结果：{json.dumps(data, ensure_ascii=False)}",
            enable_thinking=False,
            max_tokens=140,
        )
        return llm_text or (
            f"已完成五维度风险画像评定：您的综合评分为 {data['score']} 分，对应风险等级为【{data['risk_level']}】。"
            "建议以中低波动的债券型基金（50%-60%）作为核心底仓，搭配高股息平衡混合基金（20%-30%）与高流动性货币/结构性存款（20%-25%）。"
        )
    return "已为您生成总和严格等于 100% 的专属资产配置方案，全部候选产品均已通过 R1-R5 适当性合规门禁与反洗钱核查。"


def llm_report_generate(metrics: Dict[str, Any], holdings: List[Dict[str, Any]], reviewer_note: str = "") -> Dict[str, Any]:
    winners = [h for h in holdings if h["return_pct"] > 0]
    laggards = [h for h in holdings if h["return_pct"] <= 0]
    win_str = "、".join(f"《{w['name']}》(+{w['return_pct']}%)" for w in winners)
    lag_str = "、".join(f"《{l['name']}》({l['return_pct']}%)" for l in laggards) if laggards else "无亏损拖累产品，全线正收益"

    adj_note = f"（已按持牌理财师审阅意见「{reviewer_note}」完成调仓比例精修）" if reviewer_note else ""
    llm_advice = call_deepseek_llm(
        system_prompt="你是资深投后资产配置分析师。请结合持仓指标与归因数据，生成一段简明调仓建议（不超过100字）。",
        user_prompt=f"绩效指标：{metrics}，持仓：{holdings}，理财师附加意见：{reviewer_note}",
        enable_thinking=False,
        max_tokens=160,
    )
    return {
        "tool_layer": f"{LLM_CONFIG['display_model']} 收益归因与投后报告生成",
        "attribution_summary": f"收益贡献主力：{win_str}；拖累项分析：{lag_str}。",
        "rebalance_advice": (llm_advice + adj_note) if llm_advice else (
            f"组合累计收益率 +{metrics['portfolio_return_pct']}%，跑赢同期沪深300基准（+3.12%）达 "
            f"+{metrics['excess_alpha_pct']} 个百分点，最大回撤控制在 {metrics['max_drawdown_pct']}%。"
            f"建议继续持有纯债与红利底仓，将高波动科技仓位止盈置换为稳健固收+产品{adj_note}。"
        ),
    }

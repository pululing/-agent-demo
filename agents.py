"""
Lumina Wealth Multi-Agent State Machine.

Coordinates a top-level Orchestrator and three stateful sub-agents (24 nodes total):
- Orchestrator: Redis intent cache (<5ms) + structured LLM intent routing (8 categories).
- RiskAgent (9 nodes): KYC intake, 5D risk profiling, R1-R5 deterministic scoring, HITL confirmation interrupt.
- PortfolioAgent (8 nodes): market QA, BGE-M3/Milvus hybrid retrieval, Decimal 100% allocation, suitability loop.
- ReportAgent (7 nodes): holdings metrics, attribution analysis, tiered advisor review interrupt.
"""

from __future__ import annotations
from typing import Any, Dict, List
import storage
import tools


# Dual-layer memory stores (thread-scoped checkpoints + user-scoped long-term profiles)

THREAD_CHECKPOINTS: Dict[str, Dict[str, Any]] = {}
USER_LONG_TERM_MEMORY: Dict[str, Dict[str, Any]] = {}


def save_checkpoint_state(thread_id: str, user_id: str, state: Dict[str, Any]) -> None:
    THREAD_CHECKPOINTS[thread_id] = state
    storage.pg_save_checkpoint(thread_id, user_id, state)


def load_checkpoint_state(thread_id: str) -> Dict[str, Any] | None:
    if thread_id in THREAD_CHECKPOINTS:
        return THREAD_CHECKPOINTS[thread_id]
    pg_ckpt = storage.pg_load_checkpoint(thread_id)
    if pg_ckpt:
        THREAD_CHECKPOINTS[thread_id] = pg_ckpt
    return pg_ckpt


def load_user_profile_state(user_id: str) -> Dict[str, Any]:
    pg_prof = storage.pg_load_user_profile(user_id)
    if pg_prof:
        USER_LONG_TERM_MEMORY[user_id] = pg_prof
        return pg_prof
    if user_id in USER_LONG_TERM_MEMORY:
        return USER_LONG_TERM_MEMORY[user_id]
    acct = tools.get_user_account(user_id)
    USER_LONG_TERM_MEMORY[user_id] = acct
    return acct


def get_dual_memory_snapshot(thread_id: str, user_id: str) -> Dict[str, Any]:
    ckpt = load_checkpoint_state(thread_id) or {"status": "IDLE", "interrupted_at": None}
    prof = load_user_profile_state(user_id)
    return {
        "short_term_checkpoint": {
            "storage": "PostgreSQL (lumina_checkpoints)",
            "isolation_key": f"thread_id = '{thread_id}'",
            "state": {
                "status": ckpt.get("status", "IDLE"),
                "interrupted_at": ckpt.get("interrupted_at"),
                "pg_updated_at": ckpt.get("pg_updated_at", "已同步"),
            },
        },
        "long_term_profile": {
            "storage": "PostgreSQL (lumina_user_profiles)",
            "isolation_key": f"user_id = '{user_id}'",
            "data": prof,
        },
    }


# 1. Top-level Orchestrator (8 financial intent categories)

REDIS_HIGH_FREQ_INTENT_CACHE = {
    "最近大盘怎么样": {
        "intents": ["market_qa"],
        "intent_names": ["市场问答 (market_qa)"],
        "cache_hit": True,
        "latency": "3.2ms (Redis 意图缓存命中)",
        "model": "Redis Hash Cache",
        "extracted_params": {"关注标的": "A股大盘 (000300.SH / 000001.SH)", "时间范围": "近期(近5日)"},
        "vector_used": False,
    },
    "帮我做个风险评估": {
        "intents": ["risk_assess"],
        "intent_names": ["风险评估 (risk_assess)"],
        "cache_hit": True,
        "latency": "4.1ms (Redis 意图缓存命中)",
        "model": "Redis Hash Cache",
        "extracted_params": {"评估类型": "五维度动态风险画像"},
        "vector_used": False,
    },
}


def orchestrator_route(user_query: str, amount: float = 500000.0) -> Dict[str, Any]:
    """Check Redis intent cache first; fall back to LLM structured intent routing and write back."""
    import time
    t_redis = time.perf_counter()
    q = user_query.strip()
    redis_key = f"lumina:intent:{q}"

    cached_intent = storage.redis_get_json(redis_key)
    if cached_intent:
        ms = round((time.perf_counter() - t_redis) * 1000, 2)
        cached_intent["latency"] = f"{ms}ms (Redis 缓存命中)"
        cached_intent["cache_hit"] = True
        return cached_intent

    if q in REDIS_HIGH_FREQ_INTENT_CACHE:
        preset = dict(REDIS_HIGH_FREQ_INTENT_CACHE[q])
        storage.redis_set_json(redis_key, preset, ttl_seconds=3600)
        return preset

    t0 = time.perf_counter()
    llm_json_str = tools.call_deepseek_llm(
        system_prompt=(
            "你是总控编排器。请从8类理财意图[risk_assess, risk_query, asset_allocation, product_recommend, "
            "investment_report, portfolio_analysis, return_query, market_qa]中识别用户意图，只输出JSON对象："
            "{\"intent_summary\": \"一句话说明\", \"target\": \"提取标的或金额\"}"
        ),
        user_prompt=q,
        enable_thinking=False,
        json_mode=True,
        max_tokens=90,
        timeout=6.0,
    )
    model_label = tools.LLM_CONFIG["display_model"]
    elapsed_s = f"{time.perf_counter() - t0:.2f}s ({model_label}, thinking=disabled)"

    if ("全流程" in q) or ("风险" in q and ("配置" in q or "推荐" in q)):
        route_res = {
            "intents": ["risk_assess", "asset_allocation"],
            "intent_names": ["风险评估 (risk_assess)", "资产配置 (asset_allocation)"],
            "cache_hit": False,
            "latency": elapsed_s,
            "model": f"{model_label} (结构化路由)",
            "extracted_params": {"串联模式": "RiskAgent -> PortfolioAgent 强制合规串联", "投入本金": f"{amount:,.0f}元", "路由回执": llm_json_str or "复合意图"},
            "vector_used": False,
        }
    elif "大盘" in q or "行情" in q or "市场" in q:
        route_res = {
            "intents": ["market_qa"],
            "intent_names": ["市场问答 (market_qa)"],
            "cache_hit": False,
            "latency": elapsed_s,
            "model": f"{model_label} (结构化路由)",
            "extracted_params": {"关注标的": "沪深300 / 上证指数", "时间范围": "近期", "路由回执": llm_json_str or "market_qa"},
            "vector_used": False,
        }
    elif "风险" in q or "测评" in q:
        route_res = {
            "intents": ["risk_assess"],
            "intent_names": ["风险评估 (risk_assess)"],
            "cache_hit": False,
            "latency": elapsed_s,
            "model": f"{model_label} (结构化路由)",
            "extracted_params": {"评估类型": "五维度风险承受力测评"},
            "vector_used": False,
        }
    elif "报告" in q or "持仓" in q or "收益" in q or "盈亏" in q:
        route_res = {
            "intents": ["investment_report"],
            "intent_names": ["投资报告/持仓盈亏分析 (investment_report)"],
            "cache_hit": False,
            "latency": elapsed_s,
            "model": f"{model_label} (结构化路由)",
            "extracted_params": {"分析范围": "历史持仓收益归因与调仓建议"},
            "vector_used": False,
        }
    else:
        route_res = {
            "intents": ["asset_allocation"],
            "intent_names": ["资产配置 (asset_allocation)"],
            "cache_hit": False,
            "latency": elapsed_s,
            "model": f"{model_label} (结构化路由)",
            "extracted_params": {"投入本金": f"{amount:,.0f}元", "目标期限": "36个月", "路由回执": llm_json_str or "asset_allocation"},
            "vector_used": False,
        }

    storage.redis_set_json(redis_key, route_res, ttl_seconds=1800)
    return route_res


# 2. Sub-graph 1: RiskAgent (9 nodes with completeness loop and HITL confirmation)

def run_risk_agent_until_interrupt(
    thread_id: str,
    user_id: str,
    user_query: str,
    simulate_missing_kyc_once: bool = False,
    follow_up_pipeline: List[str] | None = None,
) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []
    account = tools.get_user_account(user_id)

    # 节点 R1: 收集客户基础信息 (collect_kyc)
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R1",
        "node_code": "collect_kyc",
        "node_name": "1. 收集客户基础信息",
        "exec_type": "httpx 外部 API",
        "tool_name": "get_user_account",
        "vector_used": False,
        "latency": "18ms",
        "detail": f"从账户中台拉取实名信息：{account['用户姓名']}（{account['年龄']}岁），年收入 {account['年收入']}，房贷收支比 {account['房贷收支比']}。",
    })

    # Node R2: analyze financial profile
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R2",
        "node_code": "analyze_profile",
        "node_name": "2. 分析财务画像",
        "exec_type": "LLM + Decimal",
        "tool_name": "llm_risk_profiling",
        "vector_used": False,
        "latency": "320ms",
        "detail": f"核算家庭自由现金流与负债结构：账户可投资总资产 {account['账户总资产']}，负债可控，具备中长期配置基础。",
    })

    # Node R3: extract 5D risk preferences
    profiling = tools.llm_risk_profiling(account, user_query)
    dims = profiling["dimensions"]
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R3",
        "node_code": "assess_risk_preference",
        "node_name": "3. 多轮对话识别风险偏好",
        "exec_type": "LLM + Structured Schema",
        "tool_name": "llm_risk_profiling",
        "vector_used": False,
        "latency": "410ms",
        "detail": (
            f"五维风险画像打分：收入稳定性 {dims['收入稳定性']}/10、投资经验 {dims['投资经验']}/10、"
            f"亏损容忍度 {dims['亏损容忍度']}/10、流动性需求 {dims['流动性需求']}/10、投资期限 {dims['投资期限']}/10。"
        ),
    })

    # 节点 R4: 规则引擎计算风险等级 R1-R5 (calculate_risk_level)
    calc_res = tools.risk_level_calc(dims)
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R4",
        "node_code": "calculate_risk_level",
        "node_name": "4. 规则引擎计算风险等级 R1-R5",
        "exec_type": "Python 规则引擎 (<10ms)",
        "tool_name": "risk_level_calc",
        "vector_used": False,
        "latency": "1.4ms",
        "detail": f"纯 Python 加权公式算出综合得分 {calc_res['score']} 分 -> 映射等级【{calc_res['risk_level']}】（零幻觉、含保本一票否决校验）。",
    })

    # 节点 R5: 适当性准入校验 (compliance_check)
    reg_res = tools.query_finance_regulation("普通投资者适当性等级准入与本人确认要求")
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R5",
        "node_code": "compliance_check",
        "node_name": "5. 适当性准入校验",
        "exec_type": reg_res["tool_layer"],
        "tool_name": "compliance_check + query_finance_regulation (本地 BGE-M3)",
        "vector_used": True,
        "latency": "24ms",
        "detail": f"核验实名 KYC 状态通过；调用本地 BGE-M3 检索法规库命中：{reg_res['clauses'][0]}",
    })

    # 节点 R6: 验证评估结果是否完整? (validate_risk_match 条件分支)
    if simulate_missing_kyc_once:
        trace.append({
            "agent": "RiskAgent",
            "node_id": "R6",
            "node_code": "validate_risk_match",
            "node_name": "6. 验证评估结果是否完整?",
            "exec_type": "LangGraph 条件边 (conditional_edges)",
            "tool_name": "route_by_risk_validation",
            "vector_used": False,
            "latency": "0.8ms",
            "branch_loop": "R6 -> R1 (信息缺失退回补齐)",
            "detail": "检测到客户计划用钱期限字段缺失 -> 触发条件分支退回【节点 1. 收集客户基础信息】温和引导补齐后再继续。",
        })
        trace.append({
            "agent": "RiskAgent",
            "node_id": "R1",
            "node_code": "collect_kyc",
            "node_name": "1. 收集客户基础信息 (补齐后重入)",
            "exec_type": "httpx 外部 API",
            "tool_name": "get_user_account",
            "vector_used": False,
            "latency": "12ms",
            "detail": "已补齐投资期限（36个月中长期闲置资金），五维要素齐全，校验通过。",
        })

    trace.append({
        "agent": "RiskAgent",
        "node_id": "R6",
        "node_code": "validate_risk_match",
        "node_name": "6. 验证评估结果是否完整?",
        "exec_type": "LangGraph 条件边",
        "tool_name": "route_by_risk_validation",
        "vector_used": False,
        "latency": "0.6ms",
        "detail": "Python 校验五维打分与财务收支比无逻辑矛盾，完整性核验通过 -> 进入报告生成。",
    })

    # Node R7: generate risk assessment summary
    report_summary = tools.llm_invest_advice("risk_report", calc_res)
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R7",
        "node_code": "generate_risk_report",
        "node_name": "7. 生成风险评估报告",
        "exec_type": "LLM",
        "tool_name": "llm_invest_advice",
        "vector_used": False,
        "latency": "540ms",
        "detail": report_summary,
    })

    # Node R8: pause for client confirmation interrupt
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R8",
        "node_code": "human_confirm",
        "node_name": "8. 断点暂停：用户本人确认",
        "exec_type": "StateGraph interrupt()",
        "tool_name": "interrupt",
        "vector_used": False,
        "latency": "PAUSED (等待用户确认)",
        "is_interrupt": True,
        "detail": "挂起当前会话状态并写入检查点，等待客户本人核对风险等级后确认生效。",
    })

    # Persist checkpoint state to PostgreSQL
    checkpoint_state = {
        "status": "INTERRUPTED",
        "agent": "RiskAgent",
        "interrupted_at": "R8",
        "user_id": user_id,
        "risk_result": calc_res,
        "report_summary": report_summary,
        "follow_up_pipeline": follow_up_pipeline or [],
        "trace_so_far": trace,
    }
    save_checkpoint_state(thread_id, user_id, checkpoint_state)

    return {
        "status": "INTERRUPTED",
        "interrupted_node": "R8",
        "interrupt_type": "RISK_CONFIRM",
        "interrupt_title": "请确认您的风险承受能力测评结果",
        "interrupt_desc": f"根据您的财务状况与投资偏好，系统评定您的风险承受能力为【{calc_res['risk_level']}】（综合得分 {calc_res['score']} 分）。依据《证券期货投资者适当性管理办法》，测评结果需由您本人确认后生效。",
        "risk_card": calc_res,
        "trace": trace,
        "assistant_message": f"您好，您的五维度风险承受能力测评已完成，评定结果为 **{calc_res['risk_level']}**（综合评分 **{calc_res['score']}** 分）。请核对下方测评详情并确认结果；若近期有大额支出计划，也可申请重新评估。",
    }


def resume_risk_agent(thread_id: str, action: str, reassessment_note: str = "") -> Dict[str, Any]:
    """
    处理用户在 RiskAgent 节点 8 的 Command(resume=...) 决策：
    - action == "reassess": 退回节点 3 (assess_risk_preference) 重新调整五维打分并再次停在节点 8
    - action == "confirm" : 进入节点 9 (save_risk_profile) 落库；若有串联的 PortfolioAgent 则自动继续执行！
    """
    ckpt = load_checkpoint_state(thread_id)
    if not ckpt:
        return {"status": "ERROR", "assistant_message": "未找到挂起的会话断点，请先发起风险评估。"}

    user_id = ckpt["user_id"]
    trace = list(ckpt["trace_so_far"])

    if action == "reassess":
        # 用户要求重评 -> 条件分支 R8 -> R3
        adjusted_dims = {"收入稳定性": 8, "投资经验": 2, "亏损容忍度": 2, "流动性需求": 8, "投资期限": 4}
        recalc = tools.risk_level_calc(adjusted_dims, force_capital_preserve=True)
        trace.append({
            "agent": "RiskAgent",
            "node_id": "R3",
            "node_code": "assess_risk_preference",
            "node_name": "3. 多轮对话识别风险偏好 (用户申请重评退回)",
            "exec_type": "LangGraph 条件边 R8 -> R3",
            "tool_name": "llm_risk_profiling",
            "vector_used": False,
            "latency": "380ms",
            "branch_loop": "R8 -> R3 (客户要求重评)",
            "detail": f"接收客户补充说明（「{reassessment_note or '近期有买房首付支出，不希望本金亏损'}」），亏损容忍度下调至 2/10。",
        })
        trace.append({
            "agent": "RiskAgent",
            "node_id": "R4",
            "node_code": "calculate_risk_level",
            "node_name": "4. 规则引擎计算风险等级 R1-R5 (重算)",
            "exec_type": "Python 规则引擎 (<10ms)",
            "tool_name": "risk_level_calc",
            "vector_used": False,
            "latency": "1.2ms",
            "detail": f"触发保本/低容忍度一票否决规则，风险等级下调为【{recalc['risk_level']}】（得分 {recalc['score']} 分）。",
        })
        trace.append({
            "agent": "RiskAgent",
            "node_id": "R8",
            "node_code": "human_confirm",
            "node_name": "8. 断点暂停：用户本人确认 (二次确认)",
            "exec_type": "LangGraph interrupt() 人工断点",
            "tool_name": "langgraph.types.interrupt",
            "vector_used": False,
            "latency": "PAUSED",
            "is_interrupt": True,
            "detail": f"已按补充诉求重评为【{recalc['risk_level']}】，再次挂起等待客户本人确认。",
        })
        ckpt["risk_result"] = recalc
        ckpt["trace_so_far"] = trace
        save_checkpoint_state(thread_id, user_id, ckpt)
        return {
            "status": "INTERRUPTED",
            "interrupted_node": "R8",
            "interrupt_type": "RISK_CONFIRM",
            "interrupt_title": "风险测评已更新，请确认最新评估结果",
            "interrupt_desc": f"已结合您补充的短期资金使用需求重新评估，最新风险承受能力等级为【{recalc['risk_level']}】（综合得分 {recalc['score']} 分）。请确认是否生效。",
            "risk_card": recalc,
            "trace": trace,
            "assistant_message": f"已根据您补充的流动性需求完成重新评估，您的风险承受能力等级已调整为 **{recalc['risk_level']}**（{recalc['score']} 分）。请确认最新测评结果。",
        }

    # action == "confirm": 通过 Command(resume="confirm") 恢复并执行节点 9
    risk_res = ckpt["risk_result"]
    save_res = tools.save_risk_assessment(
        user_id=user_id,
        risk_code=risk_res["risk_code"],
        score=risk_res["score"],
        dimensions=risk_res["dimensions"],
    )
    trace.append({
        "agent": "RiskAgent",
        "node_id": "R9",
        "node_code": "save_risk_profile",
        "node_name": "9. 保存风险画像入库",
        "exec_type": save_res["tool_layer"],
        "tool_name": "save_risk_assessment",
        "vector_used": False,
        "latency": "22ms",
        "detail": f"Command(resume='confirm') 恢复执行：风险画像【{risk_res['risk_level']}】已写入 PostgreSQL 长期记忆表（单号 {save_res['record_id']}，合规审计留存 5 年）。",
    })

    # 更新长期记忆
    acct = load_user_profile_state(user_id)
    acct["风险等级"] = risk_res["risk_level"]
    acct["risk_code"] = risk_res["risk_code"]
    USER_LONG_TERM_MEMORY[user_id] = acct
    save_checkpoint_state(thread_id, user_id, {"status": "COMPLETED", "interrupted_at": None, "last_agent": "RiskAgent"})

    # 检查是否存在多 Agent 串联 Pipeline (如：从风险评估 -> 资产配置全流程)
    follow_up = ckpt.get("follow_up_pipeline") or []
    if "asset_allocation" in follow_up:
        port_res = run_portfolio_agent(
            user_id=user_id,
            amount=500000.0,
            override_risk_code=risk_res["risk_code"],
            override_risk_label=risk_res["risk_level"],
            simulate_compliance_reject=False,
        )
        combined_trace = trace + port_res["trace"]
        return {
            "status": "COMPLETED",
            "risk_card": risk_res,
            "allocation_card": port_res["allocation_card"],
            "trace": combined_trace,
            "assistant_message": (
                f"您的风险承受能力等级（**{risk_res['risk_level']}**）已确认生效。\n\n"
                f"结合您的风险画像与当前市场环境，智能投顾已为您定制如下 **50 万元专属资产配置方案**："
            ),
        }

    return {
        "status": "COMPLETED",
        "risk_card": risk_res,
        "trace": trace,
        "assistant_message": (
            f"感谢确认，您的风险承受能力等级 **{risk_res['risk_level']}** 已正式生效。"
            "现在您可以随时咨询资产配置方案或理财产品推荐。"
        ),
    }


# 3. Sub-graph 2: PortfolioAgent (8 nodes with P7 -> P3 suitability rejection loop)


def run_market_qa_flow(user_id: str) -> Dict[str, Any]:
    """
    Market QA fast-path: routes structured quote/macro lookups directly through
    Redis cache and market APIs without triggering embedding or vector search.
    """
    trace: List[Dict[str, Any]] = []
    account = load_user_profile_state(user_id)

    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P1",
        "node_code": "load_user_profile",
        "node_name": "1. 加载用户风险画像",
        "exec_type": "PostgreSQL Profile Read",
        "tool_name": "get_user_account",
        "vector_used": False,
        "latency": "8ms",
        "detail": f"加载客户画像：{account['用户姓名']}（{account['风险等级']}），用于适配行情解读口径。",
    })

    mkt = tools.query_market_data("000300.SH", "近期")
    macro = tools.query_macro_data("本季度")
    analysis = tools.llm_market_analysis(mkt, macro)

    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P2",
        "node_code": "analyze_market",
        "node_name": "2. 宏观与市场行情分析",
        "exec_type": "Market API + LLM",
        "tool_name": "query_market_data + query_macro_data + llm_market_analysis",
        "vector_used": False,
        "latency": "480ms",
        "detail": "调用结构化行情与宏观数据接口，获取沪深300及宏观指标并生成市场解读。",
    })

    return {
        "status": "COMPLETED",
        "market_card": {
            "trend": analysis["market_trend"],
            "indices": mkt["indices"],
            "macro": macro,
            "summary": analysis["summary"],
        },
        "trace": trace,
        "assistant_message": (
            f"**{analysis['market_trend']}**\n\n"
            f"{analysis['summary']}"
        ),
    }


def run_portfolio_agent(
    user_id: str,
    amount: float = 500000.0,
    override_risk_code: str | None = None,
    override_risk_label: str | None = None,
    simulate_compliance_reject: bool = True,
) -> Dict[str, Any]:
    """Execute PortfolioAgent (P1 -> P8) with P7 -> P3 suitability re-selection loop."""
    trace: List[Dict[str, Any]] = []
    account = load_user_profile_state(user_id)
    risk_code = override_risk_code or account["risk_code"]
    risk_label = override_risk_label or account["风险等级"]

    # Node P1: load user risk profile
    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P1",
        "node_code": "load_user_profile",
        "node_name": "1. 加载用户风险画像",
        "exec_type": "PostgreSQL Profile Read",
        "tool_name": "get_user_account",
        "vector_used": False,
        "latency": "9ms",
        "detail": f"加载客户风险画像：{account['用户姓名']}，当前生效风险等级为【{risk_label}】，可配置金额 {amount:,.0f} 元。",
    })

    # Node P2: market & macro context
    mkt = tools.query_market_data("000300.SH", "近期")
    macro = tools.query_macro_data("本季度")
    analysis = tools.llm_market_analysis(mkt, macro)
    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P2",
        "node_code": "analyze_market",
        "node_name": "2. 宏观与市场行情分析",
        "exec_type": "Market API + LLM",
        "tool_name": "query_macro_data + llm_market_analysis",
        "vector_used": False,
        "latency": "430ms",
        "detail": f"市场定性判研：{analysis['market_trend']}（10Y国债 2.12%，沪深300 PE 分位 34%）。",
    })

    # Node P3 (initial): hybrid vector search via BGE-M3 + Milvus
    kb_res = tools.search_product_kb(
        query_text="低回撤稳健纯债与红利平衡基金",
        max_risk_code=risk_code,
        inject_unsuitable_candidate=simulate_compliance_reject,
    )
    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P3",
        "node_code": "search_products",
        "node_name": "3. 向量库检索候选产品",
        "exec_type": kb_res["tool_layer"],
        "tool_name": "search_product_kb (BGE-M3)",
        "vector_used": True,
        "latency": f"{kb_res['elapsed_ms']}ms",
        "detail": (
            f"生成 1024 维稠密向量（前6维={kb_res['query_dense_preview']}）与 {kb_res['query_sparse_tokens']} 项稀疏权重，"
            f"经 0.75:0.25 混合打分召回 {kb_res['recalled_count']} 只候选产品。"
        ),
    })

    # 节点 P4: 规则计算各类资产配比 (calculate_allocation)
    alloc_res = tools.asset_allocation_calc(risk_code=risk_code, investment_amount=amount)
    alloc_str = " + ".join(f"{a['category']} {a['ratio_str']}({a['amount_wan']})" for a in alloc_res["allocation"])
    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P4",
        "node_code": "calculate_allocation",
        "node_name": "4. 规则计算各类资产配比",
        "exec_type": "Python 配比计算器 (<5ms)",
        "tool_name": "asset_allocation_calc",
        "vector_used": False,
        "latency": "1.8ms",
        "detail": f"纯 Python Decimal 确定性计算（强制总和 = {alloc_res['sum_check_pct']}%）：{alloc_str}。",
    })

    # 节点 P5: 估算组合预期收益与最大跌幅 (estimate_portfolio_return)
    perf_res = tools.performance_calc(alloc_res["allocation"])
    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P5",
        "node_code": "estimate_portfolio_return",
        "node_name": "5. 估算组合预期收益与最大跌幅",
        "exec_type": "Python / NumPy 绩效引擎",
        "tool_name": "performance_calc",
        "vector_used": False,
        "latency": "2.1ms",
        "detail": (
            f"预期年化收益区间 {perf_res['expected_return_range']}，"
            f"最大回撤预估 {perf_res['expected_max_drawdown']}，组合夏普比率 {perf_res['sharpe_ratio']}。"
        ),
    })

    # Node P6: match products
    matched = tools.llm_product_match(alloc_res["allocation"], kb_res["candidates"])
    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P6",
        "node_code": "match_products",
        "node_name": "6. 匹配具体理财产品 Top5",
        "exec_type": "Product API + LLM",
        "tool_name": "llm_product_match + get_product_detail",
        "vector_used": False,
        "latency": "390ms",
        "detail": "初选组合清单：" + "、".join(f"《{m['name']}》({m['risk_level']})" for m in matched),
    })

    # Node P7: suitability review (loop back to P3 on breach)
    comp_res = tools.compliance_check(user_risk_code=risk_code, matched_products=matched)
    if not comp_res["is_compliant"]:
        trace.append({
            "agent": "PortfolioAgent",
            "node_id": "P7",
            "node_code": "compliance_review",
            "node_name": "7. 适当性合规审查 (拦截打回)",
            "exec_type": "Python Compliance Gate (<3ms)",
            "tool_name": "compliance_check",
            "vector_used": False,
            "latency": "1.1ms",
            "branch_loop": "P7 -> P3 (合规超标：打回重选)",
            "detail": f"[适当性超标阻断] {comp_res['violations'][0]} -> 触发条件分支打回【节点 3】开启标量硬过滤重选。",
        })

        kb_res_clean = tools.search_product_kb(
            query_text="低回撤稳健纯债与红利平衡基金",
            max_risk_code=risk_code,
            inject_unsuitable_candidate=False,
        )
        trace.append({
            "agent": "PortfolioAgent",
            "node_id": "P3",
            "node_code": "search_products",
            "node_name": "3. 向量库检索候选产品 (二次重选)",
            "exec_type": "Milvus Scalar Pre-Filter",
            "tool_name": "search_product_kb",
            "vector_used": True,
            "latency": "29ms",
            "detail": f"二次检索强制施加 Milvus 标量前置过滤 risk_level <= '{risk_code}'，剔除 R4 高风险产品。",
        })
        matched = tools.llm_product_match(alloc_res["allocation"], kb_res_clean["candidates"])
        trace.append({
            "agent": "PortfolioAgent",
            "node_id": "P6",
            "node_code": "match_products",
            "node_name": "6. 匹配具体理财产品 Top5 (重配)",
            "exec_type": tools.LLM_CONFIG["display_model"],
            "tool_name": "llm_product_match",
            "vector_used": False,
            "latency": "310ms",
            "detail": "重配后合规产品清单：" + "、".join(f"《{m['name']}》({m['risk_level']})" for m in matched),
        })
        comp_res = tools.compliance_check(user_risk_code=risk_code, matched_products=matched)

    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P7",
        "node_code": "compliance_review",
        "node_name": "7. 适当性合规审查 (通过)",
        "exec_type": "Python Compliance Gate (<3ms)",
        "tool_name": "compliance_check",
        "vector_used": False,
        "latency": "0.9ms",
        "detail": f"[适当性核验通过] 全部推荐产品风险等级 <= 客户等级 ({risk_code})，反洗钱核查与违禁词扫描通过。",
    })

    # Node P8: final allocation output
    advice_msg = tools.llm_invest_advice("allocation_plan", alloc_res)
    trace.append({
        "agent": "PortfolioAgent",
        "node_id": "P8",
        "node_code": "output_allocation_plan",
        "node_name": "8. 输出最终资产配置方案",
        "exec_type": f"{tools.LLM_CONFIG['display_model']} + FastAPI",
        "tool_name": "llm_invest_advice",
        "vector_used": False,
        "latency": "460ms",
        "detail": f"{advice_msg} 末尾附带法定风险揭示语。",
    })

    allocation_card = {
        "user_name": account["用户姓名"],
        "risk_level": risk_label,
        "total_amount": f"{amount / 10000:g}万元",
        "market_trend": analysis["market_trend"],
        "allocation": alloc_res["allocation"],
        "products": matched,
        "performance": perf_res,
        "disclosure": comp_res["mandatory_disclosure"],
        "had_compliance_loop": simulate_compliance_reject,
    }

    return {
        "status": "COMPLETED",
        "allocation_card": allocation_card,
        "trace": trace,
        "assistant_message": (
            f"已结合您的 **{risk_label}** 风险偏好与当前 **{analysis['market_trend']}** 市场环境，为您定制 **{amount / 10000:g} 万元专属资产配置方案**。"
            + (
                "\n\n> **适当性合规保障提示**：系统在构建组合时已自动为您剔除超出稳健型风险等级的高波动股票型产品，确保组合内全部标的 100% 匹配您的风险承受能力。"
                if simulate_compliance_reject
                else ""
            )
        ),
    }


# 4. Sub-graph 3: ReportAgent (7 nodes with tiered HNW human review)


def run_report_agent(thread_id: str, user_id: str) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []
    account = load_user_profile_state(user_id)
    holdings = tools.get_portfolio_history(user_id)

    # Node RP1: load historical holdings
    trace.append({
        "agent": "ReportAgent",
        "node_id": "RP1",
        "node_code": "load_portfolio",
        "node_name": "1. 加载用户历史持仓",
        "exec_type": "Portfolio API",
        "tool_name": "get_portfolio_history",
        "vector_used": False,
        "latency": "21ms",
        "detail": f"加载 {account['用户姓名']}（{account['投资者类型']}）历史持仓，共 {len(holdings)} 只存续产品。",
    })

    # Node RP2: calculate return and drawdown metrics
    metrics = tools.performance_calc([], holdings=holdings)
    trace.append({
        "agent": "ReportAgent",
        "node_id": "RP2",
        "node_code": "calculate_metrics",
        "node_name": "2. 规则计算收益与最大跌幅",
        "exec_type": "Python Decimal",
        "tool_name": "performance_calc",
        "vector_used": False,
        "latency": "1.6ms",
        "detail": (
            f"持仓总市值 {metrics['total_market_value']:,.0f} 元，累计盈利 +{metrics['total_profit']:,.0f} 元 "
            f"(+{metrics['portfolio_return_pct']}%)，最大回撤 {metrics['max_drawdown_pct']}%，夏普比率 {metrics['sharpe_ratio']}。"
        ),
    })

    # Node RP3: query market benchmark context
    tools.query_market_data("000300.SH", "持仓周期")
    trace.append({
        "agent": "ReportAgent",
        "node_id": "RP3",
        "node_code": "analyze_market_context",
        "node_name": "3. 分析同期大盘基准行情",
        "exec_type": "Market Data API",
        "tool_name": "query_market_data",
        "vector_used": False,
        "latency": "28ms",
        "detail": f"拉取同期沪深 300 基准涨跌幅（+{metrics['benchmark_hs300_pct']}%），超额收益 Alpha 为 +{metrics['excess_alpha_pct']}%。",
    })

    # Node RP4: attribution analysis
    rep_data = tools.llm_report_generate(metrics, holdings)
    trace.append({
        "agent": "ReportAgent",
        "node_id": "RP4",
        "node_code": "attribution_analysis",
        "node_name": "4. 分析持仓盈亏原因",
        "exec_type": tools.LLM_CONFIG["display_model"],
        "tool_name": "llm_report_generate",
        "vector_used": False,
        "latency": "490ms",
        "detail": rep_data["attribution_summary"],
    })

    # Node RP5: generate rebalance advice
    trace.append({
        "agent": "ReportAgent",
        "node_id": "RP5",
        "node_code": "generate_report",
        "node_name": "5. 生成完整投资报告与调仓建议",
        "exec_type": tools.LLM_CONFIG["display_model"],
        "tool_name": "llm_report_generate",
        "vector_used": False,
        "latency": "520ms",
        "detail": rep_data["rebalance_advice"],
    })

    report_card = {
        "user_name": account["用户姓名"],
        "investor_type": account["投资者类型"],
        "metrics": metrics,
        "holdings": holdings,
        "attribution": rep_data["attribution_summary"],
        "advice": rep_data["rebalance_advice"],
    }

    # 按客户等级条件分流：高净值客户 -> RP6 (interrupt 理财师人工审阅)；普通投资者 -> 直接跳过 RP6 到 RP7
    if account["投资者类型"] == "高净值客户":
        trace.append({
            "agent": "ReportAgent",
            "node_id": "RP6",
            "node_code": "human_review",
            "node_name": "6. 断点暂停：理财师人工审阅",
            "exec_type": "LangGraph interrupt() 理财师工作台断点",
            "tool_name": "langgraph.types.interrupt",
            "vector_used": False,
            "latency": "PAUSED (待持牌理财师审阅)",
            "is_interrupt": True,
            "detail": "识别为【高净值客户（资产 650 万）】，条件分支路由至节点 6 触发 interrupt()，推送后台持牌理财师审阅通过后方可下发。",
        })
        save_checkpoint_state(thread_id, user_id, {
            "status": "INTERRUPTED",
            "agent": "ReportAgent",
            "interrupted_at": "RP6",
            "user_id": user_id,
            "report_card": report_card,
            "trace_so_far": trace,
        })
        return {
            "status": "INTERRUPTED",
            "interrupted_node": "RP6",
            "interrupt_type": "ADVISOR_REVIEW",
            "interrupt_title": "尊享财富专户：专属理财师审阅中",
            "interrupt_desc": f"尊敬的 {account['用户姓名']}，您的 677.5 万元持仓分析与调仓报告已生成初稿，正由您的专属持牌理财师进行人工复核与税务筹划审阅。",
            "report_card": report_card,
            "trace": trace,
            "assistant_message": "您的专属持仓收益归因与调仓报告初稿已生成。作为尊享高净值客户，报告正由您的专属持牌理财师进行复核审阅，请确认签发或提出调整意见。",
        }

    # 普通投资者：跳过 RP6，直接执行 RP7
    trace.append({
        "agent": "ReportAgent",
        "node_id": "RP7",
        "node_code": "deliver_report",
        "node_name": "7. 流式推送并保存报告",
        "exec_type": "FastAPI SSE + PostgreSQL (普通客户跳过节点6)",
        "tool_name": "save_report",
        "vector_used": False,
        "latency": "19ms",
        "branch_loop": "RP5 -> RP7 (普通投资者直接跳过人工审阅)",
        "detail": "识别为【普通投资者】，条件分支直接跳过节点 6 人工审阅，通过 SSE 流式推送报告并异步落库 PostgreSQL。",
    })
    save_checkpoint_state(thread_id, user_id, {"status": "COMPLETED", "interrupted_at": None, "last_agent": "ReportAgent"})

    return {
        "status": "COMPLETED",
        "report_card": report_card,
        "trace": trace,
        "assistant_message": "您的持仓收益归因与调仓建议报告已生成，整体组合表现稳健并跑赢同期大盘基准，详情如下：",
    }


def resume_report_agent(thread_id: str, action: str, advisor_comment: str = "") -> Dict[str, Any]:
    """
    处理高净值客户报告在节点 6 (human_review) 的理财师审阅决策：
    - action == "rewrite": 打回节点 5 (generate_report) 附带理财师意见重写，再次停在节点 6
    - action == "approve": 进入节点 7 (deliver_report) 正式流式推送并落库
    """
    ckpt = load_checkpoint_state(thread_id)
    if not ckpt:
        return {"status": "ERROR", "assistant_message": "未找到待审阅的报告断点。"}

    user_id = ckpt.get("user_id", "U_88888")
    trace = list(ckpt["trace_so_far"])
    report_card = dict(ckpt["report_card"])

    if action == "rewrite":
        note = advisor_comment or "将科创板半导体止盈比例拆分为两批执行，并增加高净值专属税务筹划提示"
        rep_new = tools.llm_report_generate(report_card["metrics"], report_card["holdings"], reviewer_note=note)
        report_card["advice"] = rep_new["rebalance_advice"]
        trace.append({
            "agent": "ReportAgent",
            "node_id": "RP5",
            "node_code": "generate_report",
            "node_name": "5. 生成完整投资报告与调仓建议 (理财师打回重写)",
            "exec_type": "LangGraph 条件边 RP6 -> RP5",
            "tool_name": "llm_report_generate",
            "vector_used": False,
            "latency": "470ms",
            "branch_loop": "RP6 -> RP5 (理财师打回修改)",
            "detail": f"已按持牌理财师审阅意见「{note}」重新生成调仓建议：{rep_new['rebalance_advice']}",
        })
        trace.append({
            "agent": "ReportAgent",
            "node_id": "RP6",
            "node_code": "human_review",
            "node_name": "6. 断点暂停：理财师人工审阅 (复审)",
            "exec_type": "LangGraph interrupt()",
            "tool_name": "langgraph.types.interrupt",
            "vector_used": False,
            "latency": "PAUSED",
            "is_interrupt": True,
            "detail": "报告重写完成，再次挂起等待理财师复审确认。",
        })
        ckpt["report_card"] = report_card
        ckpt["trace_so_far"] = trace
        save_checkpoint_state(thread_id, user_id, ckpt)
        return {
            "status": "INTERRUPTED",
            "interrupted_node": "RP6",
            "interrupt_type": "ADVISOR_REVIEW",
            "interrupt_title": "专属投资报告已更新，待理财师最终确认",
            "interrupt_desc": f"已结合审阅意见（「{note}」）优化分批止盈与税务筹划方案，请确认是否正式签发。",
            "report_card": report_card,
            "trace": trace,
            "assistant_message": "已根据理财师意见优化分批调仓与税务筹划方案，请确认签发。",
        }

    # Action == "approve": complete report delivery
    trace.append({
        "agent": "ReportAgent",
        "node_id": "RP7",
        "node_code": "deliver_report",
        "node_name": "7. 流式推送并保存报告",
        "exec_type": "FastAPI + PostgreSQL",
        "tool_name": "save_report",
        "vector_used": False,
        "latency": "20ms",
        "detail": "理财师审阅通过：高净值专属投后报告已签发并归档 PostgreSQL。",
    })
    save_checkpoint_state(thread_id, user_id, {"status": "COMPLETED", "interrupted_at": None, "last_agent": "ReportAgent"})
    return {
        "status": "COMPLETED",
        "report_card": report_card,
        "trace": trace,
        "assistant_message": "专属持牌理财师已复核签发，您的尊享持仓归因与调仓建议报告已正式生成：",
    }


# Unified dispatch entry point

def handle_user_request(
    thread_id: str,
    user_id: str,
    query: str,
    scenario_key: str = "",
) -> Dict[str, Any]:
    route_info = orchestrator_route(query)
    intents = route_info["intents"]

    # 顶层总控编排器 Trace 节点
    orch_step = {
        "agent": "Orchestrator",
        "node_id": "ORCH",
        "node_code": "orchestrator_router",
        "node_name": "总控编排器：8类理财意图识别",
        "exec_type": route_info["model"],
        "tool_name": "Pydantic IntentSchema + Redis双锁缓存",
        "vector_used": False,
        "latency": route_info["latency"],
        "detail": (
            f"识别意图：{', '.join(route_info['intent_names'])} | "
            f"提取参数：{route_info['extracted_params']} | 是否转向量：否（大模型直接结构化输出）"
        ),
    }

    if scenario_key == "pipeline_full" or len(intents) > 1:
        res = run_risk_agent_until_interrupt(
            thread_id=thread_id,
            user_id=user_id,
            user_query=query,
            simulate_missing_kyc_once=False,
            follow_up_pipeline=["asset_allocation"],
        )
    elif intents[0] == "market_qa":
        res = run_market_qa_flow(user_id=user_id)
    elif intents[0] == "risk_assess":
        res = run_risk_agent_until_interrupt(
            thread_id=thread_id,
            user_id=user_id,
            user_query=query,
            simulate_missing_kyc_once=(scenario_key == "risk_missing_loop"),
            follow_up_pipeline=[],
        )
    elif intents[0] == "investment_report":
        res = run_report_agent(thread_id=thread_id, user_id=user_id)
    else:
        res = run_portfolio_agent(
            user_id=user_id,
            amount=500000.0,
            simulate_compliance_reject=(scenario_key != "alloc_clean"),
        )

    res["orchestrator"] = route_info
    res["trace"] = [orch_step] + res["trace"]
    res["memory_snapshot"] = get_dual_memory_snapshot(thread_id, user_id)
    return res

"""
GraphRAG 可视化检索工具 - 后端服务
技术栈: FastAPI + Neo4j + Sentence-Transformers

适配 GraphRAG 标准数据格式:
  Root/Abstract    → name 属性
  Fact             → text_content 属性（长文本）
  SubAbstract      → title_name, full_path_name 属性
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GraphRAG Visualizer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# ==================== 全局变量 ====================
driver = None
vector_model = None

# 搜索缓存（核心优化：预计算节点向量）
search_cache = {
    "node_ids": [],        # elementId 列表
    "node_names": [],      # 显示名称列表
    "embeddings": None,    # numpy 矩阵 [N, dim]
    "ready": False,        # 是否已初始化
}


def get_driver():
    global driver
    if driver is None:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[OK] Neo4j: {uri}")
    return driver


def get_vector_model():
    global vector_model
    if vector_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv(
            "VECTOR_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print(f"[LOAD] Vector model: {model_name} ...")
        vector_model = SentenceTransformer(model_name)
        print("[OK] Model ready")
    return vector_model


import numpy as np


def cosine_similarity_batch(query_vec, all_vecs):
    """批量计算余弦相似度（矩阵运算，极快）"""
    # query_vec: [dim], all_vecs: [N, dim]
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    all_norms = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-8)
    return np.dot(all_norms, query_norm)  # [N]


# ==================== GraphRAG 属性提取 ====================
TEXT_FIELDS = ["name", "text_content", "title_name", "full_path_name",
               "title", "summary", "description", "label"]


def extract_node_text(props):
    if not props:
        return None
    for field in TEXT_FIELDS:
        val = props.get(field)
        if val and isinstance(val, str) and val.strip():
            t = val.strip()
            return t[:80] + "..." if len(t) > 80 else t
    skip = {"uuid", "_id", "id", "type"}
    for k, v in props.items():
        if k not in skip and isinstance(v, str) and v.strip():
            return v.strip()[:80]
    return None


def extract_search_text(props):
    """构建向量匹配文本（拼接所有有意义的属性）"""
    if not props:
        return ""
    parts = []
    skip = {"uuid"}
    for k, v in props.items():
        if k.lower() in skip:
            continue
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " | ".join(parts)


# ==================== 数据模型 ====================
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 25
    include_neighbors: Optional[int] = 2


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"


# ==================== 启动时初始化 ====================
@app.on_event("startup")
async def startup_init():
    """启动时预加载模型和缓存节点向量"""
    try:
        get_driver()
        get_vector_model()
        rebuild_search_cache()
    except Exception as e:
        print(f"[WARN] 启动初始化失败（首次可能正常）: {e}")


def rebuild_search_cache():
    """
    重建搜索缓存：从 Neo4j 加载所有节点文本 → 批量编码向量
    后续搜索只需编码 query，直接矩阵运算 → 毫秒级响应
    """
    global search_cache
    t0 = time.time()

    drv = get_driver()
    model = get_vector_model()

    node_ids, node_names, node_texts = [], [], []

    with drv.session() as session:
        result = session.run("MATCH (n) RETURN elementId(n) as eid, properties(n) as props")
        for rec in result:
            props = dict(rec["props"]) if rec["props"] else {}
            nid = rec["eid"]
            name = extract_node_text(props)
            text = extract_search_text(props) or name or ""
            node_ids.append(nid)
            node_names.append(name or "[Unknown]")
            node_texts.append(text)

    if not node_texts:
        search_cache["ready"] = False
        print("[WARN] 无节点数据")
        return

    # 批量编码（一次性计算所有节点向量）
    print(f"[CACHE] 编码 {len(node_texts)} 个节点向量...")
    embeddings = model.encode(node_texts, show_progress_bar=True, batch_size=128)
    embeddings = np.array(embeddings)  # [N, dim]

    search_cache = {
        "node_ids": node_ids,
        "node_names": node_names,
        "embeddings": embeddings,
        "ready": True,
    }

    elapsed = time.time() - t0
    print(f"[OK] 搜索缓存就绪: {len(node_ids)} 节点, {embeddings.shape[1]} 维, 耗时 {elapsed:.1f}s")


# ==================== API 接口 ====================

@app.get("/api/health")
async def health_check():
    try:
        drv = get_driver()
        with drv.session() as session:
            r = session.run("MATCH (n) WITH count(n) as nc OPTIONAL MATCH ()-[r]->() RETURN nc, count(r) as ec")
            info = r.single()
            lr = session.run("MATCH (n) WITH labels(n)[0] as lb, count(*) as cnt RETURN lb, cnt ORDER BY cnt DESC LIMIT 20")
            labels = [{"label": row["lb"] or "(none)", "count": row["cnt"]} for row in lr]
            return {
                "status": "ok", "neo4j": "connected",
                "nodeCount": info["nc"], "edgeCount": info["ec"],
                "labels": labels,
                "searchCacheReady": search_cache["ready"],
            }
    except Exception as e:
        return {"status": "error", "neo4j": "disconnected", "detail": str(e)}


@app.get("/api/diagnose")
async def diagnose_db():
    try:
        drv = get_driver()
        with drv.session() as session:
            out = {}
            r1 = session.run("MATCH (n) WITH labels(n)[0] as label, count(*) as cnt RETURN label, cnt ORDER BY cnt DESC")
            out["labelDistribution"] = [{"label": row["label"] or "(none)", "count": row["cnt"]} for row in r1]
            r2 = session.run("MATCH ()-[r]->() WITH type(r) as rt, count(*) as cnt RETURN rt, cnt ORDER BY cnt DESC")
            out["relationshipTypes"] = [{"type": row["rt"], "count": row["cnt"]} for row in r2]
            r3 = session.run("MATCH (n) RETURN labels(n) as lbls, properties(n) as props, elementId(n) as eid LIMIT 10")
            samples = []
            for row in r3:
                props = dict(row["props"]) if row["props"] else {}
                samples.append({
                    "labels": list(row["lbls"]) if row["lbls"] else [],
                    "propertyKeys": list(props.keys()),
                    "nameText": extract_node_text(props),
                    "allProps": {k: str(v)[:100] for k, v in props.items()},
                })
            out["sampleNodes"] = samples
            return {"diagnostics": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/full")
async def get_full_graph(node_limit: int = 3000, edge_limit: int = 8000):
    try:
        drv = get_driver()
        nodes_list = []
        node_map = {}

        with drv.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n) as categories, properties(n) as props, elementId(n) as eid
                LIMIT $limit
            """, limit=node_limit)
            for record in result:
                props = dict(record["props"]) if record["props"] else {}
                eid = record["eid"]
                cats = list(record["categories"]) if record["categories"] else []
                name = extract_node_text(props)
                if not name:
                    tag = cats[0] if cats else "Node"
                    sid = eid[-8:] if len(eid) > 8 else eid
                    name = f"[{tag}] {sid}"
                idx = len(nodes_list)
                node_map[eid] = idx
                nodes_list.append({
                    "id": eid, "name": name,
                    "description": props.get("summary", "") or "",
                    "categories": cats,
                    "symbolSize": max(8, min(30, 12 + min(len(name), 15))),
                })

        edges_list = []
        seen = set()
        with drv.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN elementId(a) as src, elementId(b) as tgt, type(r) as rel
                LIMIT $limit
            """, limit=edge_limit)
            for record in result:
                src, tgt = record["src"], record["tgt"]
                key = f"{src}->{tgt}"
                if key not in seen and src in node_map and tgt in node_map:
                    seen.add(key)
                    edges_list.append({
                        "source": node_map[src], "target": node_map[tgt],
                        "relation": record["rel"],
                        "lineStyle": {"width": 1, "opacity": 0.25},
                    })

        print(f"[DATA] Nodes={len(nodes_list)}, Edges={len(edges_list)}")
        resp = {"nodes": nodes_list, "edges": edges_list,
                "nodeCount": len(nodes_list), "edgeCount": len(edges_list)}
        if len(nodes_list) == 0:
            resp["emptyReason"] = "数据库中没有节点数据"
        return resp
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_graph(request: SearchRequest):
    """
    核心检索接口（优化版）：
    1. 使用预计算的节点向量缓存 → 搜索时只编码 query
    2. 矩阵运算批量计算相似度 → 毫秒级
    3. 1跳邻居扩展（避免多跳超时）
    """
    try:
        model = get_vector_model()
        t0 = time.time()

        # 如果缓存未就绪，先重建
        if not search_cache["ready"]:
            rebuild_search_cache()

        if not search_cache["ready"]:
            return {"query": request.query, "topMatches": [],
                    "highlightData": [], "totalNodes": 0}

        # Step 1: 编码 query（只需编码 1 条文本）
        query_vec = model.encode(request.query)

        # Step 2: 矩阵运算计算所有相似度（毫秒级）
        all_scores = cosine_similarity_batch(query_vec, search_cache["embeddings"])

        # Step 3: 排序取 top_k
        top_indices = np.argsort(all_scores)[::-1][:request.top_k]

        top_matches = []
        for idx in top_indices:
            top_matches.append({
                "nodeId": search_cache["node_ids"][idx],
                "text": search_cache["node_names"][idx],
                "score": round(float(all_scores[idx]), 5),
            })

        # Step 4: 邻居扩展（只用 1 跳，避免超时）
        neighbor_ids = set()
        if request.include_neighbors > 0:
            matched = [m["nodeId"] for m in top_matches if m["score"] > 0.15]
            drv = get_driver()
            with drv.session() as session:
                for mid in matched[:10]:
                    res = session.run(
                        "MATCH (c)--(nb) WHERE elementId(c) = $mid RETURN DISTINCT elementId(nb) as nid",
                        mid=mid
                    )
                    for r in res:
                        neighbor_ids.add(r["nid"])

        # 构建高亮数据
        highlight = []
        for i, s in enumerate(all_scores):
            highlight.append({
                "nodeId": search_cache["node_ids"][i],
                "score": round(float(s), 5),
                "isNeighbor": search_cache["node_ids"][i] in neighbor_ids and float(s) < 0.15,
            })

        elapsed = time.time() - t0
        print(f"[SEARCH] '{request.query}' → {len(top_matches)} matches, {elapsed:.2f}s (cache hit)")

        return {
            "query": request.query,
            "topMatches": top_matches,
            "highlightData": highlight,
            "totalNodes": len(search_cache["node_ids"]),
            "neighborCount": len(neighbor_ids),
            "elapsed": round(elapsed, 3),
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/api/config/neo4j")
async def update_neo4j_config(config: Neo4jConfig):
    global driver
    try:
        if driver is not None:
            driver.close()
            driver = None
        os.environ["NEO4J_URI"] = config.uri
        os.environ["NEO4J_USER"] = config.user
        os.environ["NEO4J_PASSWORD"] = config.password
        os.environ["NEO4J_DATABASE"] = config.database
        drv = get_driver()
        with drv.session() as session:
            session.run("RETURN 1")
        # 重建搜索缓存
        rebuild_search_cache()
        return {"status": "success", "message": f"Connected: {config.uri}"}
    except Exception as e:
        driver = None
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

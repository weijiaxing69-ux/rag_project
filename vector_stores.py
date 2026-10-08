"""
向量检索服务（基于自建的 SimpleVectorStore，不再依赖 chromadb）。

改造点（原版只有 as_retriever(k=2)，没有任何过滤能力）：
1. 按法规过滤（metadata filter）——这是「每部法规都拿到检索配额」的关键；
2. MMR 去冗余，避免同一部法规的相邻条文刷满召回位；
3. 按法规分组的证据包（collect_evidence），而不是拍平成一个列表；
4. 关键词兜底：法条号（「第三十八条」）和专业术语这类精确串，字面匹配补一路。
"""

import re

from langchain_core.runnables import RunnableLambda

import config_data as config
from simple_store import SimpleVectorStore

# 问题里点名的条号，例如「森林法第三十八条怎么规定的」
_ARTICLE_IN_QUERY = re.compile(r"第[一二三四五六七八九十百零〇两]+条")
# 用于拆出候选关键词
_SPLIT = re.compile(r"[，,。；;、\s（）()《》\"'’“”？！!?:：/]+")


class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = SimpleVectorStore(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    # ------------------------------------------------------------
    # 过滤条件
    # ------------------------------------------------------------
    @staticmethod
    def _law_filter(law_names=None):
        """把法规名列表转成过滤条件；无过滤时返回 None。"""
        names = [n for n in (law_names or []) if n]
        if not names:
            return None
        if len(names) == 1:
            return {"law_name": names[0]}
        return {"law_name": {"$in": names}}

    # ------------------------------------------------------------
    # 语义检索
    # ------------------------------------------------------------
    def get_retriever(self, law_names=None, k=None):
        """返回一个可直接串进 LCEL 的检索器（RunnableLambda）。"""
        k = k or config.retrieve_k
        where = self._law_filter(law_names)
        use_mmr = config.use_mmr

        def _search(query):
            try:
                if use_mmr:
                    return self.vector_store.max_marginal_relevance_search(
                        query,
                        k=k,
                        fetch_k=max(k * 4, k),
                        lambda_mult=config.mmr_lambda,
                        filter=where,
                    )
                return self.vector_store.similarity_search(query, k=k, filter=where)
            except Exception:
                return []

        return RunnableLambda(_search)

    def search(self, query: str, law_names=None, k=None):
        """直接返回 Document 列表。"""
        return self.get_retriever(law_names=law_names, k=k).invoke(query)

    # ------------------------------------------------------------
    # 关键词兜底
    # ------------------------------------------------------------
    @staticmethod
    def _query_terms(query: str):
        terms = []
        for m in _ARTICLE_IN_QUERY.finditer(query or ""):
            terms.append(m.group(0))
        for piece in _SPLIT.split(query or ""):
            piece = piece.strip()
            # 太短的词做包含匹配噪音太大，从 3 字起
            if len(piece) >= 3 and piece not in terms:
                terms.append(piece)
        return terms[:6]

    def _literal_get(self, terms, where, limit):
        hits = []
        seen = set()
        for term in terms:
            if not term:
                continue
            got = self.vector_store.get(
                where=where,
                where_document={"$contains": term},
                include=["documents", "metadatas"],
            )
            for text, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
                key = (meta.get("law_name"), meta.get("article_no"))
                if key in seen:
                    continue
                seen.add(key)
                hits.append(self._to_item(text, meta, f"字面:{term}"))
                if len(hits) >= limit:
                    return hits
        return hits

    def keyword_search(self, query: str, law_names=None, k=None):
        """字面匹配兜底检索。"""
        if not config.enable_keyword_fallback:
            return []
        k = k or config.keyword_k
        return self._literal_get(self._query_terms(query), self._law_filter(law_names), k)

    # ------------------------------------------------------------
    # 对比模式的证据收集
    # ------------------------------------------------------------
    @staticmethod
    def _to_item(text: str, meta: dict, match: str):
        meta = meta or {}
        return {
            "law_name": meta.get("law_name", ""),
            "law_level": meta.get("law_level", ""),
            "law_level_rank": meta.get("law_level_rank", 9),
            "article_no": meta.get("article_no", ""),
            "chapter": meta.get("chapter", ""),
            "effective_date": meta.get("effective_date", ""),
            "text": text or "",
            "match": match,
        }

    def collect_evidence(self, query: str, law_names, per_law_k=None):
        """按法规分组收集证据：每部法规单独检索并给配额。"""
        per_law_k = per_law_k or config.per_law_k
        evidence = {}
        for law_name in law_names:
            items = []
            seen = set()
            for doc in self.search(query, law_names=[law_name], k=per_law_k):
                meta = getattr(doc, "metadata", {}) or {}
                key = (meta.get("law_name"), meta.get("article_no"))
                if key in seen:
                    continue
                seen.add(key)
                items.append(self._to_item(getattr(doc, "page_content", ""), meta, "语义"))

            for hit in self.keyword_search(query, law_names=[law_name], k=config.keyword_k):
                key = (hit["law_name"], hit["article_no"])
                if key in seen:
                    continue
                seen.add(key)
                items.append(hit)

            evidence[law_name] = items
        return evidence

    # ------------------------------------------------------------
    # 精确取条
    # ------------------------------------------------------------
    def get_articles(self, law_name: str, article_nos=None):
        """按「法规 + 条号」精确取回原文；article_nos 为空时返回该法规全部切片。"""
        wanted = {a for a in (article_nos or []) if a}
        where = self._law_filter([law_name])
        got = self.vector_store.get(where=where, include=["documents", "metadatas"])
        results = []
        for text, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
            if not wanted or (meta or {}).get("article_no") in wanted:
                results.append(self._to_item(text, meta, "精确"))
        return results

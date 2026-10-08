"""
轻量自建向量库：numpy 存向量 + json 存文本/元数据。

为什么不用 Chroma：chromadb 1.5.9 在这台机器上无法创建新的 sqlite 数据库文件
（报错 unable to open database file），而本项目的向量规模很小——约 1400 条 ×
1024 维，总共几 MB——用 numpy 完全够用，还彻底去掉了这个黑盒依赖。

对外提供与 chromadb 子集兼容的方法面（add_texts / delete / get /
similarity_search），让 knowledge_base 和 vector_stores 的改动最小。
本模块不依赖 chromadb / langchain_chroma，只依赖 numpy 与 langchain 的 Document。
"""

import json
import uuid
from pathlib import Path

import numpy as np
from langchain_core.documents import Document


def _matches(metadata: dict, where) -> bool:
    """判断一条 metadata 是否满足 where 条件（支持 == 与 $in）。"""
    if not where:
        return True
    for key, cond in where.items():
        value = metadata.get(key)
        if isinstance(cond, dict):
            if "$in" in cond:
                if value not in cond["$in"]:
                    return False
            else:
                return False  # 不支持的算子，保守判为不匹配
        elif value != cond:
            return False
    return True


class SimpleVectorStore:
    def __init__(self, collection_name, embedding_function, persist_directory):
        self.name = collection_name
        self.embedding = embedding_function
        self.dir = Path(persist_directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.dir / f"{collection_name}.vectors.npy"
        self.records_path = self.dir / f"{collection_name}.records.json"
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.records = []  # [{"id", "text", "metadata"}]
        self._load()

    # ------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------
    def _load(self):
        records = []
        if self.records_path.exists():
            try:
                records = json.loads(self.records_path.read_text(encoding="utf-8"))
            except Exception:
                records = []
        vectors = None
        if self.vectors_path.exists():
            try:
                vectors = np.load(self.vectors_path)
            except Exception:
                vectors = None
        if vectors is None or len(vectors) != len(records):
            # 两个文件对不上（比如只删了其中一个）时，宁可当空库，也别错位
            self.vectors = np.zeros((0, 0), dtype=np.float32)
            self.records = []
        else:
            self.vectors = vectors.astype(np.float32)
            self.records = records

    def _save(self):
        # 注意：np.save 会在文件名不是 .npy 结尾时自动追加 .npy，
        # 所以临时文件名必须以 .npy 结尾，否则 replace 时找不到文件。
        tmp_vec = self.vectors_path.with_suffix(".tmp.npy")
        np.save(tmp_vec, self.vectors)
        tmp_vec.replace(self.vectors_path)
        tmp_rec = self.records_path.with_suffix(".json.tmp")
        tmp_rec.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
        tmp_rec.replace(self.records_path)

    def drop_files(self):
        """删除持久化文件（用于清理探针等临时集合）。"""
        for path in (self.vectors_path, self.records_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------
    def add_texts(self, texts, metadatas=None):
        texts = list(texts)
        metadatas = list(metadatas) if metadatas else [{} for _ in texts]
        if len(texts) != len(metadatas):
            raise ValueError("texts 与 metadatas 数量不一致")
        if not texts:
            return []

        matrix = np.asarray(self.embedding.embed_documents(texts), dtype=np.float32)
        if self.vectors.shape[1] == 0:
            self.vectors = matrix
        else:
            if matrix.shape[1] != self.vectors.shape[1]:
                raise ValueError(
                    f"向量维度不一致：库里是 {self.vectors.shape[1]} 维，"
                    f"新写入是 {matrix.shape[1]} 维"
                )
            self.vectors = np.concatenate([self.vectors, matrix], axis=0)

        ids = []
        for text, meta in zip(texts, metadatas):
            rec_id = uuid.uuid4().hex
            ids.append(rec_id)
            self.records.append({"id": rec_id, "text": text, "metadata": dict(meta)})
        self._save()
        return ids

    def delete(self, where=None):
        """删除满足 where 的记录；where 为 None 时清空整个集合。"""
        if where is None:
            self.records = []
            self.vectors = np.zeros((0, 0), dtype=np.float32)
        else:
            keep = [i for i, rec in enumerate(self.records) if not _matches(rec["metadata"], where)]
            if keep:
                self.records = [self.records[i] for i in keep]
                self.vectors = self.vectors[keep]
            else:
                self.records = []
                self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._save()

    # ------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------
    def get(self, where=None, where_document=None, include=None, limit=None):
        include = include or ["documents", "metadatas"]
        ids, documents, metadatas = [], [], []
        for rec in self.records:
            if where is not None and not _matches(rec["metadata"], where):
                continue
            if where_document and not self._doc_contains(rec["text"], where_document):
                continue
            ids.append(rec["id"])
            documents.append(rec["text"])
            metadatas.append(rec["metadata"])
            if limit and len(ids) >= limit:
                break
        result = {"ids": ids}
        if "documents" in include:
            result["documents"] = documents
        if "metadatas" in include:
            result["metadatas"] = metadatas
        return result

    @staticmethod
    def _doc_contains(text, where_document):
        if isinstance(where_document, dict):
            term = where_document.get("$contains")
            if term:
                return term in text
        return True

    def _scores(self, query):
        if not self.records or self.vectors.shape[1] == 0:
            return None
        q = np.asarray(self.embedding.embed_query(query), dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        v = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
        return v @ q

    def similarity_search(self, query, k=4, filter=None):
        scores = self._scores(query)
        if scores is None:
            return []
        results = []
        for i in np.argsort(-scores):
            rec = self.records[i]
            if filter is not None and not _matches(rec["metadata"], filter):
                continue
            results.append(Document(page_content=rec["text"], metadata=dict(rec["metadata"])))
            if len(results) >= k:
                break
        return results

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20, lambda_mult=0.5, filter=None):
        """贪心 MMR：在相关性（lambda）与多样性（1-lambda）之间折中。"""
        scores = self._scores(query)
        if scores is None:
            return []
        order = np.argsort(-scores)[: max(fetch_k, k)]
        candidates = []
        for i in order:
            rec = self.records[i]
            if filter is not None and not _matches(rec["metadata"], filter):
                continue
            candidates.append((i, rec, float(scores[i])))
        if not candidates:
            return []

        v = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
        selected_records = []
        selected_vecs = []
        while candidates and len(selected_records) < k:
            best_idx, best_score = None, -1e18
            for i, rec, rel in candidates:
                diversity_penalty = 0.0
                if selected_vecs:
                    diversity_penalty = max(float(v[i] @ sv) for sv in selected_vecs)
                score = lambda_mult * rel - (1 - lambda_mult) * diversity_penalty
                if score > best_score:
                    best_score, best_idx = score, i
            if best_idx is None:
                break
            selected_records.append(self.records[best_idx])
            selected_vecs.append(v[best_idx])
            candidates = [c for c in candidates if c[0] != best_idx]
        return [
            Document(page_content=rec["text"], metadata=dict(rec["metadata"]))
            for rec in selected_records
        ]

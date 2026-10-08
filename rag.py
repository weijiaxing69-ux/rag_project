"""
RAG 服务：单法问答 + 跨法规差异/冲突/漏洞对比分析。

原版是一条「检索 → LLM 压缩 → 自由文本回答」的问答链。
新版按「建档 → 分组检索 → 对齐判定 → 自检取证」四段式组织：

- stream_answer / answer   单法问答，保留原来的流式体验
- compare                  跨法规对比分析，返回结构化结果供界面渲染对照表

对抗幻觉的三条硬约束（这是「发现漏洞」这类任务能不能信的关键）：
1. 只允许引用检索结果里真实存在的条款，判定完用确定性代码回查一遍；
2. 每条结论必须挂引用，挂不上就标注「依据不足」并压低置信度；
3. 可选再走一次 LLM 自检，审查结论是否真被它自己的引用支持。
"""

import json
import re
from operator import itemgetter

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory

import config_data as config
from file_history_store import get_history
from knowledge_base import KnowledgeBaseService
from vector_stores import VectorStoreService

# 判定类型：五种问题类型 + 两种「没问题」的结论
DIFFERENCE_TYPES = list(config.LOOPHOLE_TYPES) + ["一致", "无法判定"]

MODE_LABEL = {
    "qa": "单法问答",
    "compare": "跨法规对比",
    "loophole": "漏洞排查",
}

# 规则化的模式识别，仅用于给界面挑默认值，不参与最终判定
_LOOPHOLE_HINTS = ("漏洞", "空白", "缺失", "衔接", "抵触", "违反上位法", "罚则缺失", "不一致")
_COMPARE_HINTS = ("区别", "差异", "不同", "对比", "比较", "冲突", "矛盾", "差别", "异同")

_EMPTY_EVIDENCE = "（未检索到相关资料）"


def _extract_json(text):
    """从模型输出里抠出 JSON。

    容忍 ```json 围栏、以及前后夹带的解释性文字——模型偶尔会不听话，
    这种时候应该尽量救回来，而不是直接判定失败。
    """
    if not text:
        return None
    raw = str(text).strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = raw.find(open_ch), raw.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                continue
    return None


def _as_text(value, default=""):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def print_prompt(prompt):
    """调试用：打印实际送进模型的提示词（由 config.debug_print_prompt 控制）。"""
    if not getattr(config, "debug_print_prompt", False):
        return prompt
    print("*" * 20)
    print(prompt.to_string())
    print("*" * 20)
    return prompt


class RagService(object):
    def __init__(self):
        self.embedding = DashScopeEmbeddings(model=config.embedding_model_name)
        self.vector_service = VectorStoreService(embedding=self.embedding)
        self.kb = KnowledgeBaseService()

        # 注意：ChatTongyi 没有 temperature 字段（只有 top_p），直接传
        # temperature= 会被 pydantic 静默忽略。要降温必须走 model_kwargs，
        # 它会被合并进请求参数（见 langchain_community/chat_models/tongyi.py
        # 的 _default_params）。原版没设过温度，法律比对场景下这是要补的。
        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            model_kwargs={"temperature": config.chat_temperature},
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是林业与自然资源领域的法规助手。请**严格依据参考资料**回答问题，"
             "不要用你自己的知识补充资料中没有的内容。"
             "每一条结论后面用【法规名·第X条】标注依据；"
             "如果参考资料无法回答该问题，直接说明「参考资料中未涉及」，不要猜测。\n\n"
             "参考资料：\n{context}"),
            ("system", "以下是此前的对话记录："),
            MessagesPlaceholder("history"),
            ("user", "请回答用户提问：{input}"),
        ])
        self.chain = self.__get_chain()

    # ============================================================
    # 单法问答（保留原有流式能力）
    # ============================================================
    def _compress_docs(self, inputs: dict) -> str:
        """检索后用 LLM 压缩文档，去除无关信息，但保留法规名与条号。

        原版压缩时把法规名和条号丢掉了，导致后续无法标注出处。这里保留。
        """
        docs = inputs["docs"]
        question = inputs["question"]
        if not docs:
            return _EMPTY_EVIDENCE
        raw = "\n".join(
            f"[{i + 1}] 《{d.metadata.get('law_name', '')}》"
            f"{d.metadata.get('article_no', '')}：{d.page_content}"
            for i, d in enumerate(docs)
        )
        prompt_text = (
            f"用户问题：{question}\n\n参考材料：\n{raw}\n\n"
            f"请根据用户问题，从参考材料中提取最相关的关键信息，去除无关内容，"
            f"保留关键细节。**保留每条信息对应的法规名称和条号**，分条输出。"
        )
        compressed = self.chat_model.invoke(prompt_text)
        return _as_text(getattr(compressed, "content", compressed))

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()
        chain = (
            {
                "input": itemgetter("input"),
                "context": {
                    "docs": itemgetter("input") | retriever,
                    "question": itemgetter("input"),
                } | RunnableLambda(self._compress_docs),
                "history": itemgetter("history"),
            }
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )
        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def stream_answer(self, question: str, session_id: str = None):
        """流式回答，供 Streamlit 逐字渲染。"""
        cfg = {"configurable": {"session_id": session_id or config.default_session_id}}
        return self.chain.stream({"input": question}, cfg)

    def answer(self, question: str, session_id: str = None) -> str:
        cfg = {"configurable": {"session_id": session_id or config.default_session_id}}
        return self.chain.invoke({"input": question}, cfg)

    # ============================================================
    # 对比分析
    # ============================================================
    @staticmethod
    def detect_mode(question: str) -> str:
        """规则化识别问题意图，用于给界面挑默认模式。"""
        text = question or ""
        if any(hint in text for hint in _LOOPHOLE_HINTS):
            return "loophole"
        if any(hint in text for hint in _COMPARE_HINTS):
            return "compare"
        return "qa"

    def list_laws(self):
        """库里已有的法规清单（供界面多选）。"""
        return self.kb.list_laws()

    def _ask(self, system: str, user: str) -> str:
        response = self.chat_model.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return _as_text(getattr(response, "content", response))

    def _plan_dimensions(self, question, law_names, mode):
        """第一步：把问题拆成可逐项对照的比较维度。"""
        system = "你是法律法规比对分析专家，擅长把一个问题拆成可以逐项对照的比较维度。"
        user = (
            "需要比较的法规：\n"
            + "\n".join(f"· {name}" for name in law_names)
            + f"\n\n用户的问题：{question}\n分析模式：{MODE_LABEL.get(mode, mode)}\n\n"
            "请拆出 3-6 个「比较维度」，用于逐项对照这些法规的规定。\n"
            "要求：维度要具体到可比对的事项（例如「临时占用林地的审批权限」"
            "「擅自改变林地用途的罚款幅度」），不要用「总则」「其他规定」"
            "这类无法逐项对照的泛化维度。\n\n"
            '只输出 JSON，不要任何解释：{"dimensions": ["维度1", "维度2"]}'
        )
        data = _extract_json(self._ask(system, user))
        dimensions = []
        if isinstance(data, dict):
            dimensions = [_as_text(d) for d in _as_list(data.get("dimensions")) if _as_text(d)]
        return dimensions[:6] or list(config.DEFAULT_DIMENSIONS)

    @staticmethod
    def _evidence_index(evidence):
        """(法规名, 条号) → 证据条目，用于回查引用是否真实存在。"""
        index = {}
        for law_name, items in (evidence or {}).items():
            for item in items:
                article_no = item.get("article_no")
                if article_no:
                    index.setdefault((law_name, article_no), item)
        return index

    @staticmethod
    def _format_evidence(evidence, law_names):
        """把分组证据拼成给模型看的材料，按法规平均分配上下文预算。

        原实现用一个不断递减的全局预算，前面的法规超长时会把后面的法规
        整个丢掉，对比就变成单边的了。这里改成按法规平均配额，保证每部
        法规都占一份公平的份额。
        """
        count = max(1, len(law_names))
        per_law_budget = max(1500, config.max_context_chars // count)
        blocks = []
        for law_name in law_names:
            items = evidence.get(law_name) or []
            level = items[0].get("law_level", "") if items else ""
            lines = [f"◆《{law_name}》（{level}）"]
            if not items:
                lines.append("  （该法规未检索到相关条款）")
            used = len(lines[0]) + 1
            for item in items:
                head = f"  {item.get('article_no') or '（引言）'}"
                if item.get("chapter"):
                    head += f"〔{item['chapter']}〕"
                body = re.sub(r"\s+", " ", item.get("text") or "").strip()[:config.evidence_item_chars]
                line = f"{head}：{body}"
                if used + len(line) > per_law_budget:
                    lines.append("  ……（该法规证据过长，已截断）")
                    break
                lines.append(line)
                used += len(line) + 1
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _judge_dimension(self, question, dimension, evidence_text, mode):
        """第二步：针对单个维度做判定，返回结构化结论。"""
        system = (
            "你是法律法规比对分析专家。你只能依据给定的条款材料作答，"
            "不得引用材料之外的任何条款，也不得凭常识补充法条。"
        )
        user = (
            f"【比较维度】{dimension}\n"
            f"【用户问题】{question}\n"
            f"【分析模式】{MODE_LABEL.get(mode, mode)}\n\n"
            f"【可引用的条款材料】\n{evidence_text}\n\n"
            "【判定类型】只能取以下之一：\n"
            "- 冲突：同一事项两部法规要求不一致（下位法与上位法不一致时请在 conclusion 点明）\n"
            "- 空白：某部法规规制了该事项，另一部没有对应条款\n"
            "- 衔接断裂：有行为规范无罚则、有罚则无程序、有授权无边界\n"
            "- 时效范围错位：生效/失效日期、适用对象、地域范围不一致\n"
            "- 用语不一致：同一概念在不同法规中定义不同\n"
            "- 一致：各法规规定实质相同\n"
            "- 无法判定：材料不足\n\n"
            "【硬性要求】\n"
            "1. positions 里每部法规的立场都要给出条款依据，articles 只能填上面材料中真实出现的条号；\n"
            "2. citations[].quote 必须是材料里的原文片段，不得改写；\n"
            "3. 材料不足就填「无法判定」，并在 missing_evidence 说明还缺什么；\n"
            "4. 严禁用你自己的法律常识补充材料里没有的条款。\n\n"
            "只输出 JSON：\n"
            '{"dimension": "...", '
            '"positions": [{"law": "...", "stance": "...", "articles": ["第X条"]}], '
            '"difference_type": "...", "severity": "高|中|低", "conclusion": "...", '
            '"citations": [{"law": "...", "article": "第X条", "quote": "..."}], '
            '"missing_evidence": "...", "confidence": "高|中|低"}'
        )
        raw_text = self._ask(system, user)
        data = _extract_json(raw_text)

        finding = {
            "dimension": dimension,
            "positions": [],
            "difference_type": "无法判定",
            "severity": "低",
            "conclusion": "",
            "citations": [],
            "missing_evidence": "",
            "confidence": "低",
            "raw": "",
        }
        if not isinstance(data, dict):
            # 结构化解析失败时把原文留一段，方便排查提示词问题
            finding["conclusion"] = "模型未返回可解析的结构化结果，本维度未能判定。"
            finding["raw"] = raw_text[:800]
            return finding

        finding["dimension"] = _as_text(data.get("dimension"), dimension) or dimension
        for pos in _as_list(data.get("positions")):
            if not isinstance(pos, dict):
                continue
            finding["positions"].append({
                "law": _as_text(pos.get("law")),
                "stance": _as_text(pos.get("stance")),
                "articles": [_as_text(a) for a in _as_list(pos.get("articles")) if _as_text(a)],
            })

        dtype = _as_text(data.get("difference_type"))
        finding["difference_type"] = dtype if dtype in DIFFERENCE_TYPES else "无法判定"
        severity = _as_text(data.get("severity"))
        finding["severity"] = severity if severity in ("高", "中", "低") else "低"
        finding["conclusion"] = _as_text(data.get("conclusion"))
        for cite in _as_list(data.get("citations")):
            if not isinstance(cite, dict):
                continue
            finding["citations"].append({
                "law": _as_text(cite.get("law")),
                "article": _as_text(cite.get("article")),
                "quote": _as_text(cite.get("quote")),
            })
        finding["missing_evidence"] = _as_text(data.get("missing_evidence"))
        confidence = _as_text(data.get("confidence"))
        finding["confidence"] = confidence if confidence in ("高", "中", "低") else "低"
        return finding

    def _verify_citations(self, findings, index, exists_check=None):
        """确定性回查：核对引用是否真实存在，并标注依据状态。

        这一步不花 token、结果可复现，是防幻觉最硬的一道闸：
        模型完全可能编出「《森林法》第一百二十条」这种不存在的条号。

        exists_check 是可选回调，签名 (法规名, 条号) -> bool。只按「本次检索到
        的材料」核对，会把「引用了库里真实存在、但这次没检索到的条款」误判成
        编造；传入库回查后这类引用会保留并标注来源，避免漏掉真实发现。
        """
        for finding in findings:
            kept = []
            for cite in finding.get("citations") or []:
                law = _as_text(cite.get("law"))
                article = _as_text(cite.get("article"))
                if (law, article) in index:
                    cite["source"] = "本次材料"
                    kept.append(cite)
                elif exists_check is not None and article and exists_check(law, article):
                    cite["source"] = "库中其他条款"
                    kept.append(cite)
            finding["citations"] = kept
            finding["citation_ok"] = bool(kept)

            for pos in finding.get("positions") or []:
                law = _as_text(pos.get("law"))
                verified = []
                from_kb_only = False
                for article in (pos.get("articles") or []):
                    article_text = _as_text(article)
                    if not article_text:
                        continue
                    if (law, article_text) in index:
                        verified.append(article_text)
                    elif exists_check is not None and exists_check(law, article_text):
                        # 与 citations 保持一致：库里真实存在但未参与本次检索的
                        # 条款也要保留，只是标记来源，而不是当成编造删掉
                        verified.append(article_text)
                        from_kb_only = True
                pos["articles"] = verified
                pos["grounded"] = bool(verified)
                pos["from_kb_only"] = from_kb_only

            if not kept:
                finding["citation_note"] = "未找到可核对的引用，该结论依据不足"
                if finding.get("confidence") == "高":
                    finding["confidence"] = "低"
            elif any(cite.get("source") == "库中其他条款" for cite in kept):
                finding["citation_note"] = "部分引用来自库中未参与本次检索的条款，请人工复核。"
                if finding.get("confidence") == "高":
                    finding["confidence"] = "中"
            else:
                finding["citation_note"] = ""
        return findings

    def _self_check(self, findings, index):
        """可选第三步：再让模型当审查者，复核结论是否真被自己的引用支持。"""
        allowed = "\n".join(f"《{law}》{article}" for law, article in sorted(index.keys()))
        payload = json.dumps(
            [
                {
                    "dimension": f.get("dimension"),
                    "difference_type": f.get("difference_type"),
                    "conclusion": f.get("conclusion"),
                    "citations": f.get("citations"),
                }
                for f in findings
            ],
            ensure_ascii=False,
        )
        system = "你是严格的审查者，负责核对法律比对结论是否有条款支撑。"
        user = (
            f"【允许引用的条款清单】\n{allowed}\n\n"
            f"【待核对的结论】\n{payload}\n\n"
            "请逐条核对：conclusion 是否真的被它自己列出的 citations 支持？\n"
            "verdict 只能取：通过 / 降级（结论偏强或部分无依据）/ 剔除（无依据或与材料矛盾）。\n\n"
            '只输出 JSON：{"results": [{"dimension": "...", "verdict": "...", "reason": "..."}]}'
        )
        data = _extract_json(self._ask(system, user))
        if not isinstance(data, dict):
            for finding in findings:
                finding["review"] = "自检未返回可解析结果，已跳过"
            return findings

        verdicts = {}
        for row in _as_list(data.get("results")):
            if isinstance(row, dict):
                verdicts[_as_text(row.get("dimension"))] = (
                    _as_text(row.get("verdict")),
                    _as_text(row.get("reason")),
                )

        kept = []
        for finding in findings:
            verdict, reason = verdicts.get(finding.get("dimension"), ("通过", ""))
            finding["review"] = reason
            if verdict == "剔除":
                finding["rejected"] = True
                finding["review"] = reason or "自检判定为无依据，已剔除"
                continue
            if verdict == "降级":
                finding["confidence"] = "低"
            kept.append(finding)
        return kept

    def compare(self, question: str, law_names=None, mode: str = "compare", on_step=None):
        """跨法规对比分析主流程。

        on_step 是进度回调，界面用它显示当前进行到哪一步。
        """
        def step(message):
            if on_step:
                on_step(message)

        known = [row["law_name"] for row in self.kb.list_laws()]
        if not known:
            raise ValueError("法规库是空的：请先在「知识库更新」页上传法规文本。")

        selected = [name for name in (law_names or []) if name in known]
        if not selected:
            selected = known[:config.max_laws]
        selected = selected[:config.max_laws]
        if len(selected) < 2:
            raise ValueError(f"对比分析至少需要 2 部法规，当前只选定了 {len(selected)} 部。")
        step(f"已选定 {len(selected)} 部法规：{'、'.join(selected)}")

        dimensions = self._plan_dimensions(question, selected, mode)
        step(f"拆解出 {len(dimensions)} 个比较维度")

        evidence = self.vector_service.collect_evidence(question, selected)
        counts = "，".join(f"{name} {len(items)} 条" for name, items in evidence.items())
        step(f"检索到相关条款：{counts}")

        index = self._evidence_index(evidence)
        if not index:
            raise ValueError(
                "没有检索到任何条款。请确认这几部法规已经入库，或换一种更具体的问法。"
            )
        evidence_text = self._format_evidence(evidence, selected)

        findings = []
        for position, dimension in enumerate(dimensions, 1):
            step(f"判定中 {position}/{len(dimensions)}：{dimension}")
            findings.append(self._judge_dimension(question, dimension, evidence_text, mode))

        # 引用回查：先按本次材料核对，材料里没有的再回库确认是否真实存在
        law_articles = {}

        def exists_in_kb(law, article):
            if law not in law_articles:
                law_articles[law] = {
                    item.get("article_no")
                    for item in self.vector_service.get_articles(law, None)
                }
            return article in law_articles[law]

        findings = self._verify_citations(findings, index, exists_check=exists_in_kb)
        step("引用回查完成")

        if config.enable_self_check:
            step("自检中：复核结论是否被引用支持")
            findings = self._self_check(findings, index)
            step("自检完成")

        return {
            "question": question,
            "mode": mode,
            "laws": selected,
            "dimensions": dimensions,
            "findings": findings,
            "evidence": evidence,
            "evidence_text": evidence_text,
        }


if __name__ == "__main__":
    service = RagService()
    rows = service.list_laws()
    if not rows:
        print("法规库为空，请先运行 ingest_corpus.py 或使用上传页。")
    else:
        for row in rows:
            print(f"{row['law_level']} | {row['law_name']} | {row['article_count']} 条")

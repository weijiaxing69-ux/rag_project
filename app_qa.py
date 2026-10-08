"""
法规智能助手（Streamlit 前端）。

三种模式：
- 单法问答：沿用原来的流式问答，但要求标注条款出处
- 跨法规对比：跑完整比对流水线，输出差异对照表
- 漏洞排查：同上，提示词强调冲突/空白/衔接断裂等可核查的问题类型

与「对比」有关的渲染都不走流式：对照表本身没法逐字输出，
用结构化结果直接渲染表格 + 可展开的依据，信息密度更高。
"""

import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import config_data as config  # noqa: E402
from file_history_store import FileChatMessageHistory  # noqa: E402
from rag import RagService  # noqa: E402

st.set_page_config(page_title="林业法规差异与漏洞分析", page_icon="⚖️", layout="wide")

MODE_OPTIONS = {"qa": "单法问答", "compare": "跨法规对比", "loophole": "漏洞排查"}
# 这些类型属于「确实发现了问题」，默认展开。直接复用 config 里的分类定义，
# 避免两处各维护一份清单而走样。
PROBLEM_TYPES = tuple(config.LOOPHOLE_TYPES)


@st.cache_resource(show_spinner=False)
def get_rag_service():
    """RagService 会建立 Chroma 连接，整个进程复用一份即可，不必每次重跑都重建。"""
    return RagService()


def _cell(value) -> str:
    """表格单元格里不能出现竖线和换行，否则会把 Markdown 表格撑坏。"""
    return str(value if value is not None else "").replace("|", "｜").replace("\n", " ").strip()


def render_report(result: dict) -> None:
    """渲染比对结果：概览 → 对照表 → 逐项结论 → 原始条款。"""
    findings = result.get("findings") or []
    laws = result.get("laws") or []

    counts = {}
    for finding in findings:
        counts[finding["difference_type"]] = counts.get(finding["difference_type"], 0) + 1
    if counts:
        st.markdown("**结论概览：** " + " ｜ ".join(f"{k} {v} 项" for k, v in counts.items()))

    st.subheader("① 差异对照表")
    header = ["比较维度"] + laws + ["差异类型", "严重度", "置信度"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for finding in findings:
        positions = {p.get("law"): p for p in (finding.get("positions") or [])}
        cells = [finding["dimension"]]
        for law in laws:
            position = positions.get(law)
            if not position:
                cells.append("—")
                continue
            text = position.get("stance") or "—"
            if position.get("articles"):
                text += "（" + "、".join(position["articles"]) + "）"
            elif position.get("grounded") is False:
                text += "（无条款依据）"
            cells.append(text)
        cells += [finding["difference_type"], finding["severity"], finding["confidence"]]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    st.markdown("\n".join(lines))

    st.subheader("② 逐项结论与依据")
    if not findings:
        st.info("没有产出任何结论。可以换一种更具体的问法再试。")
    for finding in findings:
        is_problem = finding["difference_type"] in PROBLEM_TYPES
        title = (
            f"{finding['difference_type']} · {finding['dimension']}"
            f"（严重度 {finding['severity']} / 置信度 {finding['confidence']}）"
        )
        with st.expander(title, expanded=is_problem):
            st.markdown(finding.get("conclusion") or "（无结论）")

            citations = finding.get("citations") or []
            if citations:
                st.markdown("**引用条款**")
                for cite in citations:
                    quote = cite.get("quote") or ""
                    st.markdown(f"- 《{cite['law']}》{cite['article']}：{quote}")
            if finding.get("citation_note"):
                st.warning(finding["citation_note"])
            if finding.get("missing_evidence"):
                st.info(f"还缺少的证据：{finding['missing_evidence']}")
            if finding.get("review"):
                st.caption(f"自检意见：{finding['review']}")
            if finding.get("raw"):
                st.caption("模型返回了无法解析的内容，原文如下：")
                st.code(finding["raw"], language="text")

    st.subheader("③ 检索到的原始条款")
    evidence = result.get("evidence") or {}
    for law, items in evidence.items():
        with st.expander(f"《{law}》（{len(items)} 条）"):
            for item in items:
                heading = item.get("article_no") or "（引言）"
                chapter = item.get("chapter") or ""
                st.markdown(f"**{heading}**　{chapter}")
                st.text(item.get("text") or "")

    st.divider()
    st.caption(
        "⚠️ 以上结果由本地法规库检索与比对生成，可能存在遗漏或误判，不构成法律意见；"
        "重要结论请核对法规原文。"
    )


# ============================================================
# 页面
# ============================================================
st.title("林业法规差异与漏洞分析")
st.caption("检索真实条款 → 逐维度比对 → 引用回查 → 自检，结论均标注条款出处")
st.divider()

rag = get_rag_service()

try:
    laws = rag.list_laws()
except Exception as exc:  # 库连接失败时给出可读提示，而不是整页报错
    st.error(f"读取法规库失败：{exc}")
    st.stop()

if not laws:
    st.warning(
        "法规库是空的。请先到「知识库更新」页上传法规文本，"
        "或在项目目录运行 `python ingest_corpus.py` 导入 DataBase 里的三份语料。"
    )
    st.stop()

law_names = [row["law_name"] for row in laws]

# 会话 id 按「分析任务」生成，而不是像原版那样全局写死 user_001。
# 必须先于侧边栏初始化：侧边栏的「新建会话」按钮要用到它。
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"task-{uuid.uuid4().hex[:8]}"
if "message" not in st.session_state:
    st.session_state["message"] = [{
        "role": "assistant",
        "content": "你好。可以问我某部法规的规定，也可以让我比对多部法规之间的差异与漏洞。",
    }]

with st.sidebar:
    st.header("分析设置")
    mode = st.radio(
        "分析模式",
        list(MODE_OPTIONS),
        format_func=lambda key: MODE_OPTIONS[key],
        index=1,
        help="「对比」和「漏洞排查」都会跑完整的比对流水线；区别在提示词强调的重点。",
    )

    selected_laws = st.multiselect(
        "参与分析的法规",
        law_names,
        default=law_names[: min(2, len(law_names))],
        help=f"最多同时比对 4 部；对比/排查模式至少选 2 部。",
    )

    st.caption(f"库中已有 {len(laws)} 部法规 / {sum(row['chunks'] for row in laws)} 个切片")

    if st.button("新建会话（清空对话历史）", use_container_width=True):
        FileChatMessageHistory(st.session_state.get("session_id"), "./chat_history").clear()
        st.session_state["session_id"] = f"task-{uuid.uuid4().hex[:8]}"
        st.session_state["message"] = [{"role": "assistant", "content": "已开启新的分析会话。"}]
        st.session_state.pop("last_result", None)

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 上一轮比对结果在重跑后仍然显示
if st.session_state.get("last_result"):
    render_report(st.session_state["last_result"])

prompt = st.chat_input("例如：临时占用林地的审批权限，《森林法》和《森林法实施条例》有什么不同？")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    if mode == "qa":
        collected = []
        with st.chat_message("assistant"):
            with st.spinner("检索并作答中…"):
                try:
                    stream = rag.stream_answer(prompt, st.session_state["session_id"])

                    def capture(generator, cache):
                        for chunk in generator:
                            cache.append(chunk)
                            yield chunk

                    st.write_stream(capture(stream, collected))
                except Exception as exc:
                    st.error(f"回答失败：{exc}")
        answer = "".join(collected) or "（未获得回答）"
        st.session_state["message"].append({"role": "assistant", "content": answer})
        st.session_state.pop("last_result", None)
    else:
        if len(selected_laws) < 2:
            st.error("对比分析至少需要在左侧选择 2 部法规。")
        else:
            try:
                with st.status(f"正在以「{MODE_OPTIONS[mode]}」模式分析…", expanded=True) as status:
                    def on_step(message):
                        status.write(message)

                    result = rag.compare(prompt, selected_laws, mode=mode, on_step=on_step)
                    status.update(label="分析完成", state="complete", expanded=False)
                st.session_state["last_result"] = result
                summary = "已完成比对，结果见下方对照表。"
                st.session_state["message"].append({"role": "assistant", "content": summary})
                render_report(result)
            except Exception as exc:
                st.error(f"比对失败：{exc}")
                st.session_state["message"].append({
                    "role": "assistant",
                    "content": f"比对失败：{exc}",
                })

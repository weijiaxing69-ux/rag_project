"""
法规知识库更新服务（Streamlit 前端）。

改造点：
1. 支持一次上传多个文件（原版 accept_multiple_files=False，只能一个一个传）；
2. 可以手选效力层级，覆盖从文件名推断的结果——效力层级是后面判断
   「下位法是否违反上位法」的依据，不能靠猜；
3. 入库后回显每部法规解析出多少条，而不是笼统一句「成功」；
4. 列出已入库法规，可按法规删除。

页面顺序刻意把「已入库法规」放在上传区之后，这样刚传完就能立刻看到结果，
不需要额外刷新一次页面。
"""

import streamlit as st  
from dotenv import load_dotenv

load_dotenv()

from knowledge_base import KnowledgeBaseService  # noqa: E402

st.set_page_config(page_title="法规知识库更新", page_icon="📚", layout="wide")

LEVEL_OPTIONS = [
    "自动识别（按文件名）",
    "法律",
    "行政法规",
    "部门规章",
    "地方性法规",
    "地方政府规章",
    "司法解释",
    "其他",
]


@st.cache_resource(show_spinner=False)
def get_service():
    """Chroma 连接整个进程复用一份。"""
    return KnowledgeBaseService()


st.title("法规知识库更新")
st.caption("上传的法规会按「第X条」切成条文入库，并带上名称、效力层级、文号等元数据")
st.divider()

service = get_service()

# ============================================================
# 上传
# ============================================================
level_choice = st.selectbox(
    "效力层级",
    LEVEL_OPTIONS,
    index=0,
    help="选「自动识别」时按文件名推断，例如 01_国家法律.txt → 法律、03_部门规章.txt → 部门规章。",
)

uploader_files = st.file_uploader(
    "请上传法规 txt 文件（可多选）",
    type=["txt"],
    accept_multiple_files=True,
)

if uploader_files:
    if st.button(f"载入这 {len(uploader_files)} 个文件", type="primary"):
        explicit_level = "" if level_choice.startswith("自动") else level_choice
        for uploader_file in uploader_files:
            file_name = uploader_file.name
            try:
                text = uploader_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.subheader(file_name)
                st.error("不是 UTF-8 编码的文本，请另存为 UTF-8 后重试。")
                continue

            st.subheader(file_name)
            st.caption(
                f"格式 {uploader_file.type} ｜ 大小 {uploader_file.size / 1024:.1f} KB ｜ {len(text)} 字"
            )
            with st.spinner("解析并载入中…"):
                try:
                    result = service.upload_by_str(text, file_name, explicit_level)
                    st.text(result)
                except Exception as exc:
                    st.error(f"载入失败：{exc}")

        st.success("处理完成，已入库法规见下方列表。")

# ============================================================
# 库概览与清单（放在上传之后，保证看到的是最新状态）
# ============================================================
st.divider()
try:
    laws = service.list_laws()
except Exception as exc:
    st.error(f"读取知识库失败：{exc}")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("已入库法规", len(laws))
col2.metric("切片总数", sum(row["chunks"] for row in laws))

st.subheader("已入库法规")
if not laws:
    st.info(
        "还没有任何法规入库。可以上传上面的 txt，"
        "或在项目目录运行 `python ingest_corpus.py` 导入 DataBase 里的三份语料。"
    )
else:
    for row in laws:
        left, middle, right = st.columns([6, 2, 1])
        left.markdown(
            f"**{row['law_name']}**　`{row['law_level']}`　"
            f"{row['article_count']} 条 / {row['chunks']} 切片"
        )
        middle.caption((row.get("effective_date") or "")[:28])
        if right.button("删除", key=f"delete-{row['law_name']}"):
            service.delete_law(row["law_name"])
            st.rerun()

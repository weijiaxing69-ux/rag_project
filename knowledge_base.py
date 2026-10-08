"""
知识库服务：法规文档的结构化解析 + 法条级入库 + 法规维度的库管理。

与原版的区别（原版只有 md5 去重 + 1000 字硬切 + source 一个元数据字段）：

1. 切分单位从「固定 1000 字」改为「第X条」。原版按字数硬切会把条文拦腰截断，
   而法条比对的最小单位就是条/款/项，切碎了既对不齐也引不准。
2. 每条切片带上法规元数据（名称/效力层级/文号/施行日期/条款号/章节）。
   这是后续「按法规过滤检索」和判断「下位法是否违反上位法」的前提。
3. 入库指纹带上结构版本，解析逻辑升级后老记录不会导致静默跳过。

本模块的解析部分只依赖标准库，可离线测试（不需要联网、不需要 API Key）。
"""

import hashlib
import os
import re
from datetime import datetime

import config_data as config6 
from langchain_community.embeddings import DashScopeEmbeddings
from simple_store import SimpleVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# 中文数字
# ============================================================
_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(text: str):
    """把「七十五」「一百零三」「十一」这类中文数字转成 int，失败返回 None。"""
    if not text:
        return None
    total = 0
    number = 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if number == 0:
                number = 1  # 「十」单独出现时代表 10
            total += number * unit
            number = 0
        else:
            return None
    return total + number


# ============================================================
# 法规文本解析
# ============================================================
# 语料格式（DataBase/法规库整理版/*.txt）：
#   ================      ← 头部块分隔线
#   【法规名称】中华人民共和国森林法
#   【制定机关】...
#   【通过/修订】...
#   【施行日期】...
#   【文号】...
#   【效力状态】现行有效
#   【官方来源】...
#   ----------------      ← 头部与正文分隔线
#   第一条 ... 第二条 ... ← 正文；条文之间可能是换行、空格，甚至全挤在一行
#
# 正因为分隔符不统一，条文切分绝不按换行做，而是按「第X条」标记的位置做。
_HEADER_SPLIT = re.compile(r"\n={20,}\n")
_HEADER_SEP = re.compile(r"-{20,}")
_FIELD = re.compile(r"^【(.+?)】\s*(.*)$", re.M)
_ARTICLE = re.compile(r"第([一二三四五六七八九十百零〇两]+)条")
_CHAPTER = re.compile(r"第([一二三四五六七八九十百零〇]+)([章节])")
# 章节标题后面紧跟着下一条/下一章时，要在那里把标题切断
_TITLE_CUT = re.compile(r"第[一二三四五六七八九十百零〇]+[条章节]")

# 效力层级及其权重，用于「下位法是否违反上位法」的判断
LEVEL_RANK = {
    "法律": 1,
    "行政法规": 2,
    "部门规章": 3,
    "地方性法规": 4,
    "地方政府规章": 5,
    "司法解释": 6,
    "其他": 9,
}

# 从文件名前缀推断效力层级（01_国家法律 / 02_行政法规 / 03_部门规章）
_FILE_LEVEL = [
    ("国家法律", "法律"),
    ("行政法规", "行政法规"),
    ("部门规章", "部门规章"),
    ("地方性法规", "地方性法规"),
    ("地方政府规章", "地方政府规章"),
    ("司法解释", "司法解释"),
]


def infer_level_from_filename(filename: str) -> str:
    """从文件名推断效力层级；推断不出来就返回空串，由调用方补。"""
    base = os.path.basename(filename or "")
    for keyword, level in _FILE_LEVEL:
        if keyword in base:
            return level
    return ""


def _chapter_title(body: str, start: int, end: int) -> str:
    """取出章节标题：从标记之后截到句末/换行/下一条标记为止。

    语料里既可能是「第二章 草原权属」独占一行，也可能是
    「第二章 草原权属第九条 ...」这样与下一条紧贴，所以三种边界都要切。
    """
    tail = body[end:end + 40]
    tail = re.split(r"[\n。；]", tail, maxsplit=1)[0]
    tail = _TITLE_CUT.split(tail, maxsplit=1)[0]
    return re.sub(r"\s+", "", tail)  # 「罚 则」这类排版空格统一去掉


def extract_articles(body: str):
    """把一部法规的正文切成条文列表。

    关键点：用「序号递增校验」剔除交叉引用。
    条文正文里大量出现「依照本法第七十四条的规定」这类引用，如果见
    「第X条」就切，会把一条正文切碎成好几块。真正的条号必然是第一条、
    第二条……依次递增的，所以只接受「等于期望序号」的标记。

    返回 (preamble, articles)。
    """
    candidates = []
    for m in _ARTICLE.finditer(body):
        num = cn_to_int(m.group(1))
        if num is not None:
            candidates.append((m.start(), m.end(), num, m.group(0)))

    accepted = []
    expected = 1
    for marker in candidates:
        if marker[2] == expected:
            accepted.append(marker)
            expected += 1

    if not accepted:
        return body.strip(), []

    # 章节标记：要求位于行首，或紧跟在句末标点/空白之后，避免把
    # 「依照本条例第三章的规定」当成章节标题。
    chapters = []
    for m in _CHAPTER.finditer(body):
        before = body[m.start() - 1] if m.start() > 0 else "\n"
        if before not in "\n。；" and not before.isspace():
            continue
        title = _chapter_title(body, m.start(), m.end())
        if not title:
            continue
        chapters.append((m.start(), f"{m.group(0)} {title}"))

    # 条文与章节都是边界：条文正文在遇到下一个边界（下一条或下一章）时结束，
    # 这样章标题不会被塞进上一条的正文里。
    article_starts = [a[0] for a in accepted]
    boundaries = sorted(article_starts + [c[0] for c in chapters])

    articles = []
    for start, _end, num, label in accepted:
        later = [b for b in boundaries if b > start]
        stop = later[0] if later else len(body)
        chapter = ""
        for chapter_start, chapter_title in chapters:
            if chapter_start < start:
                chapter = chapter_title
            else:
                break
        articles.append({
            "article_no": label,
            "article_num": num,
            "chapter": chapter,
            "text": body[start:stop].strip(),
        })

    preamble = body[:boundaries[0]].strip()
    return preamble, articles


def parse_law_document(text: str, source_name: str = "", default_level: str = ""):
    """把一份「多法规合集」文本解析成法规列表。

    返回 list[dict]，每项是一部法规；解析不出任何法规时返回空列表，
    调用方应退回普通文档的入库路径。
    """
    laws = []
    fallback_level = default_level or infer_level_from_filename(source_name)

    for section in _HEADER_SPLIT.split(text):
        fields = dict(_FIELD.findall(section))
        law_name = (fields.get("法规名称") or "").strip()
        if not law_name:
            continue

        parts = _HEADER_SEP.split(section, maxsplit=1)
        body = parts[1] if len(parts) > 1 else ""
        preamble, articles = extract_articles(body)
        if not articles:
            continue

        level = fallback_level or "其他"
        laws.append({
            "law_name": law_name,
            "doc_id": get_string_md5(law_name)[:12],
            "law_level": level,
            "law_level_rank": LEVEL_RANK.get(level, 9),
            "issuing_body": (fields.get("制定机关") or "").strip(),
            "revision": (fields.get("通过/修订") or "").strip(),
            "effective_date": (fields.get("施行日期") or "").strip(),
            "doc_number": (fields.get("文号") or "").strip(),
            "status": (fields.get("效力状态") or "").strip(),
            "official_source": (fields.get("官方来源") or "").strip(),
            "source_file": source_name,
            "preamble": preamble,
            "articles": articles,
        })
    return laws


# ============================================================
# 入库指纹
# ============================================================
def get_string_md5(input_str: str, encoding: str = "utf-8") -> str:
    """计算字符串的 md5（16 进制）。"""
    return hashlib.md5(input_str.encode(encoding=encoding)).hexdigest()


def content_fingerprint(data: str, filename: str) -> str:
    """入库指纹 = 结构版本 + 文件名 + 内容。

    把结构版本和文件名一起算进去，是为了修掉原版的一个坑：原版只用内容
    md5，导致「解析逻辑升级了但文件内容没变」时被判为「已入库」而静默跳过，
    库里留着的其实还是旧结构的切片。
    """
    payload = f"v{config.ingest_schema_version}|{filename}|{data}"
    return get_string_md5(payload)


def check_md5(md5_str: str) -> bool:
    """检查该指纹是否已处理过。"""
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()
        return False
    with open(config.md5_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == md5_str:
                return True
    return False


def save_md5(md5_str: str) -> None:
    """记录已处理的指纹。"""
    with open(config.md5_path, "a", encoding="utf-8") as handle:
        handle.write(md5_str + "\n")


# ============================================================
# 知识库服务
# ============================================================
class KnowledgeBaseService(object):
    def __init__(self):
        self.chroma = SimpleVectorStore(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )
        # 仅用于「单条超长」的二次切分，以及非法规文本的兜底入库
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.sub_split_threshold,
            chunk_overlap=config.sub_split_overlap,
            separators=config.sub_separators,
            length_function=len,
        )

    # ---------- 元数据 ----------
    @staticmethod
    def _article_metadata(law: dict, article: dict, filename: str, now: str) -> dict:
        """构造单条切片的元数据。

        Chroma 只接受 str/int/float/bool，所以缺失字段一律写空串而不是 None。
        """
        return {
            "law_name": law["law_name"],
            "doc_id": law["doc_id"],
            "law_level": law["law_level"],
            "law_level_rank": law["law_level_rank"],
            "issuing_body": law["issuing_body"],
            "doc_number": law["doc_number"],
            "effective_date": law["effective_date"],
            "law_revision": law["revision"],
            "law_status": law["status"],
            "official_source": law["official_source"],
            "article_no": article.get("article_no", ""),
            "article_num": article.get("article_num", 0),
            "chapter": article.get("chapter", ""),
            "chunk_kind": article.get("chunk_kind", "article"),
            "source": filename,
            "ingest_version": config.ingest_schema_version,
            "create_time": now,
            "operator": "wjx",
        }

    def _chunks_for_law(self, law: dict):
        """把一部法规展开成 (文本, 元数据) 列表，超长条文做二次切分。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chunks = []

        # 正文前的引言（标题/目录等），有内容才收
        if law.get("preamble") and len(law["preamble"]) > 20:
            chunks.append((
                f"【{law['law_name']}】{law['preamble']}",
                self._article_metadata(
                    law,
                    {"article_no": "", "article_num": 0, "chunk_kind": "preamble"},
                    law["source_file"], now,
                ),
            ))

        for article in law["articles"]:
            text = article["text"]
            prefix = f"【{law['law_name']}·{article['article_no']}】"
            if article.get("chapter"):
                prefix += f"（{article['chapter']}）"
            if len(text) > config.sub_split_threshold:
                # 超长条文（含大量「（一）（二）」分项）按项/句二次切分，
                # 每片仍带上「法规名·条号」前缀，保证引用可回溯。
                for piece in self.spliter.split_text(text):
                    if piece.strip():
                        chunks.append((
                            f"{prefix}{piece.strip()}",
                            self._article_metadata(law, article, law["source_file"], now),
                        ))
            else:
                chunks.append((
                    f"{prefix}{text}",
                    self._article_metadata(law, article, law["source_file"], now),
                ))
        return chunks

    # ---------- 写入 ----------
    def _add_chunks(self, chunks):
        """分批写入切片，返回写入条数。

        Chroma 在元数据字段较多时，一次性提交过多记录会触发
        "Failed to apply logs to the metadata segment"；分批提交能规避，
        代价只是多几次调用。
        """
        size = max(1, config.ingest_batch_size)
        written = 0
        for start in range(0, len(chunks), size):
            batch = chunks[start:start + size]
            self.chroma.add_texts(
                texts=[item[0] for item in batch],
                metadatas=[item[1] for item in batch],
            )
            written += len(batch)
        return written

    def self_check(self, size: int = None) -> str:
        """预检：往目标库写一小批真实结构的记录再删掉，确认库可写。

        Chroma 的写入故障通常要等到真正入库时才暴露，一跑就是几分钟。
        这个预检让问题在几秒内暴露，并给出可执行的下一步。
        """
        size = size or max(2, min(config.ingest_batch_size, 6))
        probe_name = "preflight_check"
        placeholder_law = {
            "law_name": "预检占位法规",
            "doc_id": "preflight",
            "law_level": "其他",
            "law_level_rank": 9,
            "issuing_body": "",
            "doc_number": "",
            "effective_date": "",
            "revision": "",
            "status": "",
            "official_source": "",
        }
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadatas = [
            self._article_metadata(
                placeholder_law,
                {"article_no": f"第{i}条", "article_num": i, "chapter": "", "chunk_kind": "article"},
                "preflight",
                now,
            )
            for i in range(size)
        ]
        try:
            probe = SimpleVectorStore(
                collection_name=probe_name,
                embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
                persist_directory=config.persist_directory,
            )
            probe.add_texts(
                texts=[f"预检占位记录 {i}" for i in range(size)],
                metadatas=metadatas,
            )
            probe.delete()
            probe.drop_files()
        except Exception as exc:
            return (
                f"[预检失败] 目标库无法写入（{type(exc).__name__}: {exc}）。\n"
                f"  目录：{config.persist_directory}"
            )
        return f"[预检通过] {config.persist_directory} 可正常写入（{size} 条测试记录已清理）"

    # ---------- 入库 ----------
    def upload_law_text(self, text: str, filename: str, level: str = "", on_progress=None) -> str:
        """解析并入库一份法规合集，返回人类可读的结果说明。

        on_progress 是可选回调，用于逐部法规汇报进度：一次导入要跑几百次
        embedding，中间若没有任何输出，很容易被误以为卡死而手动中断——而在
        写库过程中强制中断会让 Chroma 的写前日志无法回放，直接损坏该集合。
        """
        laws = parse_law_document(text, source_name=filename, default_level=level)
        if not laws:
            # 不是法规格式的普通文档，退回原版的字数切分路径
            return self._upload_plain_text(text, filename)

        def report(message):
            if on_progress:
                on_progress(message)

        lines = [f"[成功] 已载入 {len(laws)} 部法规："]
        total_chunks = 0
        for index, law in enumerate(laws, 1):
            report(f"  [{index}/{len(laws)}] 《{law['law_name']}》{len(law['articles'])} 条，向量化中…")
            chunks = self._chunks_for_law(law)
            if not chunks:
                continue
            # 同名法规先清旧切片，保证重复入库幂等、不会新旧并存
            try:
                self.chroma.delete(where={"law_name": law["law_name"]})
            except Exception:
                pass
            total_chunks += self._add_chunks(chunks)
            lines.append(
                f"  · {law['law_name']}（{law['law_level']}）"
                f"{len(law['articles'])} 条 → {len(chunks)} 个切片"
            )
        lines.append(f"共 {total_chunks} 个切片，效力层级：{laws[0]['law_level']}")
        return "\n".join(lines)

    def _upload_plain_text(self, data: str, filename: str) -> str:
        """非法规格式文本的兜底入库（沿用原版逻辑）。"""
        if len(data) > config.sub_split_threshold:
            pieces = self.spliter.split_text(data)
        else:
            pieces = [data]
        metadata = {
            "law_name": os.path.splitext(os.path.basename(filename or ""))[0] or "未命名文档",
            "source": filename,
            "chunk_kind": "plain",
            "ingest_version": config.ingest_schema_version,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "wjx",
        }
        self._add_chunks([(piece, dict(metadata)) for piece in pieces])
        return f"[成功] 未识别出法规结构，按普通文档切分为 {len(pieces)} 个切片入库"

    def upload_by_str(self, data: str, filename: str, level: str = "", on_progress=None) -> str:
        """对外的入库入口：先查指纹，再解析入库。

        level 用于覆盖从文件名推断出的效力层级（界面上手选时会传）。
        """
        fingerprint = content_fingerprint(data, filename)
        if check_md5(fingerprint):
            return "[跳过] 内容已经存在数据库中"
        result = self.upload_law_text(data, filename, level=level, on_progress=on_progress)
        save_md5(fingerprint)
        return result

    # ---------- 库管理 ----------
    def list_laws(self):
        """列出库里已有的法规及其条款数，供界面做多选。"""
        got = self.chroma.get(include=["metadatas"])
        metadatas = got.get("metadatas") or []
        aggregate = {}
        for meta in metadatas:
            meta = meta or {}
            name = meta.get("law_name") or "未命名文档"
            record = aggregate.setdefault(name, {
                "law_name": name,
                "law_level": meta.get("law_level", ""),
                "law_level_rank": meta.get("law_level_rank", 9),
                "doc_number": meta.get("doc_number", ""),
                "effective_date": meta.get("effective_date", ""),
                "source": meta.get("source", ""),
                "chunks": 0,
                "_articles": set(),
            })
            record["chunks"] += 1
            article_no = meta.get("article_no")
            if article_no:
                record["_articles"].add(article_no)
        rows = []
        for record in aggregate.values():
            record["article_count"] = len(record.pop("_articles"))
            rows.append(record)
        rows.sort(key=lambda r: (r["law_level_rank"], r["law_name"]))
        return rows

    def delete_law(self, law_name: str) -> str:
        """按法规名删除其全部切片。"""
        self.chroma.delete(where={"law_name": law_name})
        return f"[成功] 已删除《{law_name}》"

    def stats(self) -> dict:
        """库概览：法规数 / 切片数。"""
        laws = self.list_laws()
        return {
            "law_count": len(laws),
            "chunk_count": sum(law["chunks"] for law in laws),
        }

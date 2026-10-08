"""
一次性导入脚本：把 DataBase/法规库整理版/ 下的法规语料导入法条级集合。

用法：
    python ingest_corpus.py

说明：
- 语料是三份按效力层级分好的合集，文件名里的 01_国家法律 / 02_行政法规 /
  03_部门规章 会被自动识别成效力层级。
- 旧的 "rag" 集合（1000 字硬切、只有 source 元数据）本脚本不碰，
  新数据进入 config.collection_name（默认 "rag_law"），两者互不影响，
  旧集合可以随时回退。
- 入库指纹带结构版本，所以本脚本可以重复运行：第一次真入库，之后会被跳过。
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import config_data as config  # noqa: E402
from knowledge_base import KnowledgeBaseService  # noqa: E402

CORPUS_DIR = Path("DataBase") / "法规库整理版"


def progress(message: str) -> None:
    """逐部法规打印进度。

    一次导入要跑几百次 embedding，全程静默几分钟会被误以为卡死；
    而此时按 Ctrl+C 打断写库，会让 Chroma 的写前日志无法回放、
    直接损坏集合（报错：Failed to apply logs to the metadata segment）。
    """
    print(message, flush=True)


def main() -> int:
    if not CORPUS_DIR.is_dir():
        print(f"[错误] 找不到语料目录：{CORPUS_DIR}")
        return 1

    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        print(f"[错误] {CORPUS_DIR} 下没有 txt 文件")
        return 1

    print(f"语料目录：{CORPUS_DIR}")
    print(f"目标集合：{config.collection_name}")
    print(f"持久化目录：{config.persist_directory}")
    print(f"入库结构版本：v{config.ingest_schema_version}")
    print(f"单批写入条数：{config.ingest_batch_size}")
    print("-" * 60)

    service = KnowledgeBaseService()

    # 预检：先花几秒确认目标库可写，避免跑了几分钟才在写库阶段失败。
    # 正式入库前只消耗一个 embedding 请求用于预检。
    print("预检目标向量库是否可写…", flush=True)
    check = service.self_check()
    print(check, flush=True)
    if check.startswith("[预检失败]"):
        print("\n预检未通过，已中止（没有为正式入库消耗 embedding 配额）。")
        return 1
    print("-" * 60)

    try:
        for path in files:
            text = path.read_text(encoding="utf-8")
            print(f"\n>>> {path.name}（{len(text)} 字）", flush=True)
            try:
                print(service.upload_by_str(text, path.name, on_progress=progress), flush=True)
            except Exception as exc:
                message = str(exc)
                print(f"[失败] {path.name}：{message}")
                if "向量维度不一致" in message:
                    # 持久化目录里残留了旧模型（维度不同）写的向量文件
                    print("\n  ↑ 向量维度与当前 embedding 模型不一致。")
                    print("    多半是持久化目录里残留了旧模型写的向量文件。")
                    print(f"    删掉 {config.persist_directory} 里的 {config.collection_name}.vectors.npy")
                    print(f"    和 {config.collection_name}.records.json 两个文件后重跑即可。")
                return 1
    except KeyboardInterrupt:
        print("\n\n[已中断] 已取消导入。")
        print("  现在的存储层是「先写临时文件、再原子替换」，中断不会损坏已入库的数据；")
        print("  已导入成功的文件会被入库指纹跳过，重跑不会重复消耗配额。")
        return 130

    print("-" * 60)
    stats = service.stats()
    print(f"完成：{stats['law_count']} 部法规，{stats['chunk_count']} 个切片")
    print("\n已入库法规清单：")
    for row in service.list_laws():
        print(f"  [{row['law_level']}] {row['law_name']}（{row['article_count']} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

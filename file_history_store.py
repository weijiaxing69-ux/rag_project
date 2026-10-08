"""
基于文件的会话历史存储。

原版的问题：session_id 由 config 写死成 "user_001"，所有人共用一份历史，
且上一轮的结论会被当成事实喂进下一轮——在对话式问答里这只是不方便，
但在「法规比对」场景里，上一轮对 A 法的结论会被当成既定事实去比对 B 法，
是会直接污染结论的。

改造点：
1. 会话 id 由调用方传入（界面按「分析任务」生成），这里负责把它规范化成
   安全的文件名，避免路径分隔符等字符把文件写到 chat_history 之外；
2. 历史文件损坏时不再抛异常，退回空历史。
"""

import json
import os
import re
from typing import List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

#所有的内容都放在 “当前文件夹下的chat_history文件夹中” 
SESSION_ROOT = "./chat_history" 
# 允许中文、字母、数字、下划线、点、短横线，其余一律替换掉
_UNSAFE = re.compile(r"[^0-9A-Za-z._\u4e00-\u9fff-]+")


def normalize_session_id(session_id: str) -> str:
    """把会话 id 规范成安全的文件名。"""
    raw = (session_id or "").strip() or "default"
    safe = _UNSAFE.sub("_", raw).strip("._")
    return (safe or "default")[:80]


def get_history(session_id):
    return FileChatMessageHistory(session_id, SESSION_ROOT)


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = normalize_session_id(session_id)
        self.storage_path = storage_path
        # 沿用原版的无扩展名命名，老的历史文件仍然能被读到
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)
        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False)

    @property
    def messages(self) -> List[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            # 文件被写坏（例如上次写到一半被中断）时退回空历史，而不是让页面崩掉
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)

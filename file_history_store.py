import os
from typing import Sequence, List
import json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path=os.path.join(self.storage_path,self.session_id)
        #确保文件夹存在
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self,messages:Sequence[BaseMessage])->None:
        all_messages=list(self.messages) #已经存在的消息列表
        all_messages.extend(messages)    #新的和已有的融合成一个list

        #为了方便写入 转换成为字典的形式
        new_messages=[message_to_dict(message) for message in all_messages]

        #写入文件
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)

    @property #变成成员属性 更加方便
    def messages(self)->List[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data=json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self)->None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)


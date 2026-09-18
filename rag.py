from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_community.chat_models.tongyi import ChatTongyi
from file_history_store import get_history
class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system","以我给你的资料为主,用简洁的语言回答问题,参考资料{context}"),
                ("system","并且我提供用户的对话历史记录如下"),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问{input}")
            ]
        )
        self.chat_model=ChatTongyi(model=config.chat_model_name)
        self.chain=None


    def __get_chain(self):
        retriever=self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if docs is not docs:
                return "无相关资料"
            formatted_str=""
            for doc in docs:
                formatted_str+=doc.text+"\n"
            return formatted_str
        chain=(
            {
                "input":RunnablePassthrough(),
                "context":retriever |format_document
            } | self.prompt_template |self.chat_model|StrOutputParser()
        )
        conversion_chain=RunnableWithMessageHistory(
            chain=chain,
            get_history=get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversion_chain
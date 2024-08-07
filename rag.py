from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_openai import ChatOpenAI
from nemo_embed import NemoEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
import secrets
from os.path import os
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def make_token():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16)  

class ChatCSV:
    vector_store = None
    retriever = None
    history_aware_retriever = None
    memory = None
    model = None
    chain = None
    db = None
    sessionid = make_token()
    llm = os.getenv("LLM")
    api_key = os.getenv("API_KEY")
    NIMhost = os.getenv("NIMHOST")
    token = os.getenv("MAXTOKEN")
    temp = os.getenv("TEMPERATURE")
    top_p = os.getenv("TOP_P")
    collection_name = os.getenv("COLLECTION_NAME")
    CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION")
    store = {}

    def __init__(self):
        """
        Initializes the question-answering system with default configurations.

        This constructor sets up the following components:
        - A ChatOpenAI model for generating responses ('neural-chat').
        - A RecursiveCharacterTextSplitter for splitting text into chunks.
        - A PromptTemplate for constructing prompts with placeholders for question and context.
        """

        self.model = ChatNVIDIA(model=self.llm,
            temperature=self.temp,
            top_p=self.top_p,
            max_tokens=self.token,
            base_url=f"http://{self.NIMhost}/v1")

    def load_model(self, model_llm: str):
        self.model = None
        self.__init__()

    def ingest(self, ingest_path: str, index: bool, type: str):
        '''
        Ingests data from a web url or pdf file containing and set up the
        components for further analysis.
        '''      
        embeddings=NemoEmbeddings(
            server_url=f"htpp://localhost:9080/v1/embeddings",
            model_name="NV-Embed-QA",
        )
        if index:
            print("loading indexes")
            vector_store = PGVector(
                collection_name=self.collection_name,
                connection=self.CONNECTION_STRING,
                embeddings=embeddings,
                use_jsonb=True,
            )
            self.retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": float(self.temp),
                },
            )
        else:
            if type == "web":
                loader = WebBaseLoader(ingest_path)
            elif type == "pdf":
                loader = PyPDFLoader(
                    file_path=ingest_path,
                )
            elif type == "csv":
                loader = CSVLoader(file_path=ingest_path)
            # loads the data
            data = loader.load()
            # splits the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
            all_splits = text_splitter.split_documents(data)
            self.db = PGVector.from_documents(
                embedding=embeddings,
                documents=all_splits,
                collection_name=self.collection_name,
                connection=self.CONNECTION_STRING,
                use_jsonb=True,
                pre_delete_collection=False,
            )
            # sets up the retriever
            self.retriever = self.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": float(self.temp),
                },
            )

    def ask(self, query: str, kb, prompt):
        """
        Asks a question using the configured processing chain.

        Parameters:
        - query (str): The question to be asked.

        Returns:
        - str: The result of processing the question through the configured chain.
        If the processing chain is not set up (empty), a message is returned
        prompting to add a CSV document first.
        """

        qa_system_prompt = prompt

        if kb:
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                ]
            )
            # print(contextualize_q_prompt)
            self.history_aware_retriever = create_history_aware_retriever(
                self.model, self.retriever, contextualize_q_prompt
            )
  
            qa_system_prompt = qa_system_prompt + """\\nUse the following pieces of retrieved context to answer the question.\\n\\n{context}"""
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(self.model, self.prompt)
            runnable = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)
        else:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                ]
            )
            runnable: Runnable = self.prompt | self.model | StrOutputParser()

        ### Statefully manage chat history ###

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        # if not kb:
        #     runnable: Runnable = self.prompt | self.model

        self.chain = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer" if kb else None,
        )

        response = None
        phrase = "Conversation roles must alternate user/assistant/user/assistant/..."
        response = self.chain.invoke({"input": query},
                                         config={
                                        "configurable": {"session_id": self.sessionid}
                                        },
                                    )
        print(response)
        mod_resp = response.replace(phrase, "")

        if kb:
            return response["answer"]
        else:
            return mod_resp

    def check_kb(self, kb: bool):
        self.clear()
        if kb:
            self.ingest(self, index=True, type="")

    def clear(self):
        """
        Clears the components in the question-answering system.

        This method resets the vector store, retriever, and processing chain to None,
        effectively clearing the existing configuration.
        """
        print("clearing the components in the question-answering system.")
        self.chain = None
        self.vector_store = None
        self.retriever = None
        # self.history_aware_retriever = None
        self.memory = None
        self.db = None
        self.store = {}
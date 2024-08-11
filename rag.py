from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector, DistanceStrategy
from langchain.retrievers.multi_query import MultiQueryRetriever
from nemo_embed import NemoEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import BaseOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from typing import List
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
import secrets
import logging
from os.path import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
text_splitter = None
TEXT_SPLITTER_EMBEDDING_MODEL = "intfloat/e5-large-v2"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def make_token():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16)  
# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines
    
class ChatCSV:
    vector_store = None
    embeddings = None
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

        self.embeddings=NemoEmbeddings(
            server_url=f"http://localhost:9080/v1/embeddings",
            model_name="NV-Embed-QA",
        )

    def get_vector_index(self, data: str = ""):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)
        return PGVector.from_documents(
            embedding=self.embeddings,
            documents=all_splits,
            collection_name=self.collection_name,
            connection=self.CONNECTION_STRING,
            use_jsonb=True,
            distance_strategy=DistanceStrategy.COSINE,
            pre_delete_collection=False,
        )
    
    def get_vector_retriever(self, num_nodes: int = 4) -> PGVector:
        """Create the document retriever."""
        index = PGVector(
                collection_name=self.collection_name,
                connection=self.CONNECTION_STRING,
                embeddings=self.embeddings,
                use_jsonb=True,
            )
        return index.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": num_nodes,
                    "score_threshold": 0.5,
                },
            )

    def get_multiquery_retriever(self, query: str = ""):
            output_parser = LineListOutputParser()
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate five 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines.
                Original question: {question}""",
            )
            llm_chain = QUERY_PROMPT | self.model | output_parser
            retriever_from_llm = MultiQueryRetriever(
                retriever=self.get_vector_retriever(), llm_chain=llm_chain, parser_keys="lines"
            )
            logging.basicConfig()
            logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
            docs = retriever_from_llm.invoke(query)
            return docs

    def get_response_with_history(self, query: str, prompt: str):
        ### Contextualize question ###
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )
        runnable: Runnable = qa_prompt | self.model
        ### Statefully manage chat history ###
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            return self.store[session_id]


        self.chain = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        response = self.chain.invoke({"input": query},
                                            config={
                                            "configurable": {"session_id": self.sessionid}
                                            },
                                        )
        return response.content

    def get_response_kb(self, query: str):
        # multi question option
        multi = self.get_multiquery_retriever(query)
        print(multi)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_prompt=QA_PROMPT
        doc_chain = load_qa_chain(self.model, chain_type="stuff", prompt=QA_PROMPT)
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=self.get_vector_retriever(),
            chain_type="stuff",
            memory=memory,
            combine_docs_chain_kwargs={'prompt': qa_prompt},
        )
        response = qa({"question": query})
        print(response)
        return response["answer"]

    def load_model(self, model_llm: str):
        self.model = None
        self.__init__()

    def ingest(self, ingest_path: str, index: bool, type: str):
        '''
        Ingests data from a web url or pdf file containing and set up the
        components for further analysis.
        '''      
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
        embedding_model_name = TEXT_SPLITTER_EMBEDDING_MODEL

        self.vector_store = self.get_vector_index(data)
        # sets up the retriever
        self.retriever = self.vector_store.as_retriever(
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
        """
        if kb:
            response = self.get_response_kb(query)
        else:
            response = self.get_response_with_history(query, prompt)
        phrase = "Conversation roles must alternate user/assistant/user/assistant/..."
        mod_resp = response.replace(phrase, "")
        return mod_resp
        
    def check_kb(self, kb: bool):
        if kb:
            self.retriever = self.get_vector_retriever()
            print(self.retriever)
        else:
            self.clear()

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
        self.memory = None
        self.db = None
        self.store = {}
from langchain_openai import ChatOpenAI
from templates import BASIC_TEMPLATE, SYSTEM_TEMPLATE
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS, EMBEDDED_MODEL
import os


class TextEmbeddings():
    def __init__(self, query):
        self.llm =  llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    model=MODEL_NAME,   
                    temperature=TEMPERATURE,   
                    max_tokens=MAX_TOKENS
                )
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDED_MODEL)
        self.query = query
        self.index_name = None

    def pdf_loader(self, filename:str):
        try:
            loader = PyPDFLoader(filename)
            documents = loader.load()

            return documents
        except Exception as e:
            print("Error :  ",e)
    
    def text_splitter(self, document):
        try:
            pass
        except Exception as e:
            pass
    
    def vector_store(self):
        try:
            # Store embeddings in FAISS
            vector_store = FAISS.from_documents(self.all_splits, self.embeddings)
            vector_store.save_local("faiss_index")
        except Exception as e:
            pass

    def retrieve_vector(self):
        try:
            # Load embeddings in FAISS
            vector_store = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            
            # k is the number of chunks to retrieve
            retriever = vector_store.as_retriever(k=2)
        except Exception as e:
            print("Error    :   ", e)

    def embedding_document(self, path: str):
        try:
            if os.path.exists(path):
                print("embedding document")
                # Step 1: Load the document
                document = self.pdf_loader(path)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                self.all_splits = text_splitter.split_documents(documents=document)

                if not os.path.exists("faiss_index"):
                    self.vector_store()

                retriever = self.retrieve_vector(self.index_name)
                docs = retriever.invoke(self.query)
                
                question_answering_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            SYSTEM_TEMPLATE,
                        ),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )

                document_chain = create_stuff_documents_chain(self.llm, question_answering_prompt)
                result = document_chain.invoke({
                    "context": docs,
                    "messages": [
                        HumanMessage(content=self.query)
                    ]
                })
                print("\nresult: ", result)
                return result
            else:
                print(f"\nThe file {path} does not exist.")
        except Exception as e:
            print("Error :  ",e)


    

if __name__=="__main__":
    # runnable_attempt("bear")
    query = input("Type in your query: \n")
    obj = TextEmbeddings()
    while query != "exit":
        obj(query)
        query = input("Type in your query: \n")
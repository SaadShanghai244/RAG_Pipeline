from langchain_openai import ChatOpenAI
from .templates import SYSTEM_TEMPLATE
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from .config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS, EMBEDDED_MODEL, DIR_PATH, FILE_PATH, URL
from langchain_community.document_loaders import PyPDFDirectoryLoader
from .custom_loader import CustomDocumentLoader
from .scrape_url import Scrapper
from .utils import log_error
import os
import asyncio
import re


class TextEmbeddings:
    """
    A class to handle document loading, splitting, embedding, and retrieval for answering questions
    asynchronously based on a PDF document using FAISS.
    """

    def __init__(self, query, file_path=None):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDED_MODEL)
        self.query = query
        self.index_name = "faiss_index"  # Default FAISS index name
        self.all_splits = None

    async def pdf_loader_single_file(self):
        """
        Asynchronously loads a PDF document and returns its content as a list of `Document` objects.

        Args:
            filename (str): The path to the PDF file.

        Returns:
            list: A list of `Document` objects containing the text content of the PDF.
        """
        try:
            loader = PyPDFLoader(self.file_path)
            documents = await asyncio.to_thread(loader.load)  # Offload blocking IO to thread
            return documents
        except Exception as e:
            print("Error in pdf_loader: ", e)
    
    async def pdf_loader_dir(self):
        """
        Asynchronously loads a PDF document and returns its content as a list of `Document` objects.

        Args:
            filename (str): The path to the PDF file.

        Returns:
            list: A list of `Document` objects containing the text content of the PDF.
        """
        try:
            loader = PyPDFDirectoryLoader(DIR_PATH)
            documents = await asyncio.to_thread(loader.load)  # Offload blocking IO to thread
            return documents
        except Exception as e:
            print("Error in pdf loader dir      : ", e)

    async def txt_file_loader(self):
        """
        Asynchronously loads a text file and returns its content as a list of `Document` objects.
        """
        try:
            loader = CustomDocumentLoader(self.file_path)
            doc = loader.load()
            return doc
        except Exception as e:
            print("Error in txt_file_loader: ", e)

    async def storing_vector(self):
        """
        Asynchronously stores document splits (chunks) as embeddings in a FAISS vector store.

        Returns:
            bool: True if the vector store was successfully saved, False otherwise.
        """
        try:
            if not self.all_splits:
                raise ValueError("Document splits are not available for vector storage.")

            # Store the document splits into the FAISS vector store
            vector_store = FAISS.from_documents(documents=self.all_splits, embedding=self.embeddings)
            await asyncio.to_thread(vector_store.save_local, self.index_name)  # Offload to avoid blocking

            # Confirm if the index was created successfully
            if os.path.exists(f"{self.index_name}/index.faiss"):
                return True
            else:
                log_error("Vector store index was not saved correctly.")
                return False
        except Exception as e:
            log_error(f"Error in vector_store: {str(e)}")
            return False

    async def retrieve_vector(self):
        """
        Asynchronously loads a FAISS vector store from disk and sets up a retriever for querying the stored documents.

        Returns:
            retriever: A retriever object that allows querying of the FAISS vector store.
        """
        try:
            vector_store = await asyncio.to_thread(FAISS.load_local, self.index_name, self.embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever(k=4)
            return retriever
        except Exception as e:
            log_error(f"Error in retrieve_vector: {str(e)}")
            return None

    async def getting_file_extension(self):
        # Regex to capture the file extension without the dot
        regex = r".*\/[^\/]+\.([\w]+)$"
        
        match = re.search(regex, self.file_path)
        
        if match:
            file_extension = match.group(1)  # Capture the file extension without the dot
            return file_extension
        else:
            return "No file extension found."
        
    async def select_method_to_apply(self):
        """
        Asynchronously selects the appropriate method to load and process a document based on the file type.
        
        If `self.file_path` is `None`, it processes a directory of files.
        If `self.file_path` is not `None`, it checks the file extension (e.g., pdf, txt) and calls the appropriate
        loader function for the file type.
        
        Returns:
            document (object): The loaded document content (based on file type), or None if an error occurs or
                               if an invalid file path is provided.
        
        Raises:
            Exception: Any exceptions encountered during file loading will be caught and logged with a timestamp.
        """
        try:
            if self.file_path is None:
                print("No specific file path provided, loading documents from the directory...")
                
                # Step 1: Load the document asynchronously from a directory
                document = await self.pdf_loader_dir()
                return document
            
            else:
                # Get the file extension
                file_type = await self.getting_file_extension()
                print(f"File type determined: {file_type}")

                # Dictionary mapping file types to their respective loader functions
                file_loaders = {
                    'pdf': self.pdf_loader_single_file,
                    'txt': self.txt_file_loader
                }

                # Check if the file type is supported
                if file_type in file_loaders:
                    document = await file_loaders[file_type]()
                    print(f"Document loaded successfully from file type: {file_type}")
                    return document
                else:
                    log_error(f"Unsupported file type: {file_type}")
                    return None

        except Exception as e:
            log_error(f"Error in select_method_to_apply: {str(e)}")
            return None

    async def split_document(self, document):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
            self.all_splits = text_splitter.split_documents(documents=document)
            return True
        except Exception as e:
            log_error(f"Error in split_document: {str(e)}")
            return None
        
    async def embedding_document(self):
        """
        Asynchronously embeds a document and performs a retrieval-based question answering process.

        Args:
            path (str): The file path to the PDF document.
        """
        try:
            if os.path.exists(self.file_path):
                document = await self.select_method_to_apply()
                if document:
                    # Step 2: Split the document into chunks
                    res = await self.split_document(document)
                    if res is None:
                        return None

                    # Step 3: Create the FAISS vector store if it doesn't exist
                    if not os.path.exists(self.index_name):
                        await self.storing_vector()

                    # Step 4: Retrieve documents using vector store
                    retriever = await self.retrieve_vector()
                    # docs = await asyncio.to_thread(retriever.invoke, self.query)
                    docs = retriever.invoke(self.query)

                    # Step 5: Create the chain and answer the question
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

                    print("\nResult: ", result)
                    return result
            else:
                print(f"\nThe file {self.file_path} does not exist.")
        except Exception as e:
            log_error(f"Error in embedding_document: {str(e)}")
            return None
    
    async def main(self, user_input,query, path=FILE_PATH, url=URL):
        try:
            """This function is the main function of TextEmbedding class and it will check the user requirements that does it want to scrape or want to embed the document"""
            self.query = query
            if user_input == "doc":
                if path:
                    self.file_path = path
                    await self.embedding_document()
            elif user_input == "scrape":
                scrapper = Scrapper(url=url, text_embeddings=self)
                await scrapper.scrape()
            else:
                print("\nInvalid user input. Please choose either 'doc' or'scrape'.")
        except Exception as e:
            log_error(f"Error in Main Function of TextEmbedding class: {str(e)}")
            return None



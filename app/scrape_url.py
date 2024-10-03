import asyncio
from .templates import SYSTEM_TEMPLATE
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from .utils import log_error
from langchain_core.messages import HumanMessage, SystemMessage
import os

class Scrapper:
    def __init__(self, url: str, text_embeddings):
        """
        Initialize the Scrapper class with a URL and text embedding object.
        
        Parameters:
        url (str): The URL of the web page to scrape.
        text_embeddings: The embedding model and associated properties for processing text.
        """
        self.url = url
        self.text_embeddings = text_embeddings
        print("Entered in __init__ scrape mode")
    
    async def load_url_data(self):
        """
        Asynchronously load data from the provided URL.

        Returns:
        data: The scraped data from the URL.
        None: Returns None if an error occurs during scraping.
        """
        try:
            loader = WebBaseLoader(self.url)
            data = loader.load()
            print("\nScraped data successfully.")
            return data
        except Exception as e:
            log_error(f"Error in load_url_data inside Scrapper class: {str(e)}")
            return None

    async def generate_response(self, vector_store):
        """
        Asynchronously generate a response by performing similarity search and using the LLM to generate an answer.

        Parameters:
        vector_store: The FAISS vector store object loaded from disk.

        Returns:
        str: The generated response from the LLM based on the retrieved documents.
        None: Returns None if an error occurs during response generation.
        """
        try:
            docs = vector_store.similarity_search(self.text_embeddings.query)
            print("\nRetrieved relevant documents.", docs)

            context = "\n\n".join([doc.page_content for doc in docs])

            system_message = SystemMessage(content=SYSTEM_TEMPLATE)
            human_message = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{self.text_embeddings.query}\n\n")

            messages = [system_message, human_message]
            response = await asyncio.to_thread(self.text_embeddings.llm.invoke, messages)
            return response.content
        except Exception as e:
            log_error(f"Error in generate_response inside Scrapper class: {str(e)}")
            return None
        
    async def scrape(self):
        """
        The main scraping function that orchestrates the entire scraping and processing pipeline.

        This function:
        1. Loads webpage content.
        2. Splits the loaded document into manageable chunks.
        3. Creates a FAISS vector store if it doesn't exist.
        4. Loads the FAISS vector store from disk.
        5. Performs similarity search using the user's query.
        6. Generates a response based on the retrieved documents.

        Returns:
        str: The final answer generated based on the scraping and LLM processing.
        None: Returns None if an error occurs during the process.
        """
        try:
            print("Entered in scrape mode")
            data = await self.load_url_data()
            res = await self.text_embeddings.split_document(data)
            if res is None:
                print("Error in splitting the document.")
                return None

            print("\nCreated and saved vector store to disk.")

            if not os.path.exists(self.text_embeddings.index_name):
                await self.text_embeddings.storing_vector()

            vector_store = FAISS.load_local(self.text_embeddings.index_name, self.text_embeddings.embeddings, allow_dangerous_deserialization=True)
            print("\nLoaded vector store from disk.")

            answer = await self.generate_response(vector_store)

            print("\nAnswer:", answer)

            return answer
        except Exception as e:
            log_error(f"Error in latest_scrape_data: {str(e)}")
            return None

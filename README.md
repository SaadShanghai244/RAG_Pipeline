# Document Embedding and Retrieval System with Langchain and FAISS

This project is designed to load, process, embed, and retrieve document content using Langchain and FAISS. It supports asynchronous document processing for PDFs and text files. The core functionality revolves around embedding document chunks for semantic search and using vector stores for question-answering based on document content.

## Features

Asynchronous Document Loading: Handles both single PDF files and directories containing multiple PDF files.
Document Embedding: Embeds documents using Langchain's OpenAI Embeddings and stores them in FAISS.
File Type Handling: Supports multiple file types (currently .pdf and .txt) and automatically applies the correct method for loading based on the file extension.
Document Retrieval: Retrieves document chunks based on the provided query using FAISS and Langchain.
Question-Answering: Uses Langchain's model chaining to answer questions based on document content.

## Installation

Clone the repository:
git clone https://github.com/SaadShanghai244/RAG_Pipeline.git
cd RAG_Pipeline

### Install the required dependencies:

pip install -r requirements.txt
Set up environment variables by creating a .env file in the root directory or export them in your terminal.

## Run the application:

python retrieval_object.py

## Asynchronous Program Flow

Main Class: TextEmbeddings
Handles document embedding, loading, and retrieval.
Supports both single files and directories.

### Key Methods:

pdf_loader_single_file(): Loads a single PDF file.
pdf_loader_dir(): Loads all PDF files in a specified directory.
txt_file_loader(): Loads a single text file.
select_method_to_apply(): Determines which method to apply based on file type.
embedding_document(): Performs embedding and retrieval.

## Running the Application

When you run the application, you will be prompted to enter a query. The query is then processed using the loaded document content:
The application will:

Determine the file type (PDF or text).
Load the content.
Embed the document.
Store and retrieve chunks using FAISS.
Answer the query based on the document content.

## Error Handling

The system logs errors using the utils.py module, which defines the following logging functions:

log_error(message): Logs error messages to the console with a timestamp.
log_info(message): Logs informational messages (if needed).
Customization
You can easily extend the project to support other document types by:

## Modifying the file_loaders dictionary in select_method_to_apply() to include new loaders.

Implementing new document loaders for other file types (e.g., .docx, .csv).

## Contributing

Feel free to submit pull requests or open issues to add more functionality or improve the code.

## License

This project is licensed under the MIT License.

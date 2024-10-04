# Document Embedding and Retrieval System with Langchain and FAISS

This project is designed to load, process, embed, and retrieve document content using Langchain and FAISS. It supports asynchronous document processing for PDFs and text files. The core functionality revolves around embedding document chunks for semantic search and using vector stores for question-answering based on document content.

## Features

- **Asynchronous Document Loading**: Handles both single PDF files and directories containing multiple PDF files.
- **Document Embedding**: Embeds documents using Langchain's OpenAI Embeddings and stores them in FAISS.
- **File Type Handling**: Supports multiple file types (currently `.pdf` and `.txt`) and automatically applies the correct method for loading based on the file extension.
- **Document Retrieval**: Retrieves document chunks based on the provided query using FAISS and Langchain.
- **Question-Answering**: Uses Langchain's model chaining to answer questions based on document content.

## Installation

### Clone the repository:

git clone https://github.com/SaadShanghai244/RAG_Pipeline.git
cd RAG_Pipeline

## Install the required dependencies:

pip install -r requirements.txt

## Set up environment variables:

Create a .env file in the root directory with the necessary environment variables (e.g., API keys) or export them in your terminal session.

export OPENAI_API_KEY=your_openai_key
export MODEL_NAME=gpt-4
export TEMPERATURE=0.5
export MAX_TOKENS=1000
export EMBEDDED_MODEL

## How to Run the Project

After setting up the environment, you can run the project using:
python main.py

## Program Flow

Main Class: TextEmbeddings
Handles document embedding, loading, and retrieval.
Supports both single files and directories.

## Key Methods:

pdf_loader_single_file(): Loads a single PDF file.
pdf_loader_dir(): Loads all PDF files in a specified directory.
txt_file_loader(): Loads a single text file.
select_method_to_apply(): Determines which method to apply based on the file type.
embedding_document(): Performs embedding and retrieval.

## How It Works

When enviornment and config files configured properly.
Then user needs to tell that knwoledge base is doument or an website url. If it is website url and url is placed inside the config file, then select 'scrape'.
Otherwise type doc for document
Then it asks about the user query.
When you fulfill the required things then it automatically detects the file format like its pdf or txt
The system determines the file type (PDF or text) and loads the document content.
The document is split into manageable chunks.
FAISS is used to store and retrieve document chunks.
The system processes the query using Langchain's embeddings and retrieval system and generates an answer based on the document content.

## Error Handling

The system logs errors using the utils.py module, which defines the following logging functions:

log_error(message): Logs error messages to the console with a timestamp.
log_info(message): Logs informational messages (if needed).

## Customization

You can easily extend the project to support other document types by:

Modifying the file_loaders dictionary in the select_method_to_apply() function to include new loaders.
Implementing new document loaders for other file types (e.g., .docx, .csv).
Contributing
Feel free to submit pull requests or open issues to add more functionality or improve the code.

## License

This project is licensed under the MIT License.

### Key Additions:

- **Installation Instructions**: Step-by-step instructions on how to clone the repository, install dependencies, and set up environment variables.
- **Running the Project**: Detailed instructions on how to run the project using the `python main.py` command.
- **Program Flow and Features**: Describes how the system works, the key methods, and error handling.
- **Customization**: Instructions on how to extend the project to support other file types.

This should give anyone a clear understanding of how to install, run, and contribute to your project.

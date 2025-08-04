
# RAG-Based Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG)** system to answer questions based on a provided PDF document. It combines document retrieval with a language model to generate precise answers, ensuring responses are grounded in the context of the document.

## Features
- **Document Processing**: Extracts text from PDF files and splits it into manageable chunks.
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face to generate embeddings for text chunks.
- **Vector Storage**: Stores embeddings in a FAISS vector database for efficient similarity searches.
- **Question Answering**: Retrieves relevant document chunks and uses the Cohere language model to generate answers.
- **Handling Unknown Queries**: Returns "answer not available in context" if the question cannot be answered from the document.

## Technologies Used
- **LangChain**: For building the RAG pipeline, including text splitting, embeddings, and retrieval.
- **Hugging Face Embeddings**: To convert text into numerical representations.
- **FAISS**: For efficient similarity search and retrieval of document chunks.
- **Cohere**: As the language model to generate answers from retrieved context.
- **PyPDF2**: For extracting text from PDF files.

## How It Works
1. **Document Loading**: The system reads a PDF file and extracts its text.
2. **Text Splitting**: The text is split into smaller chunks for processing.
3. **Embeddings Generation**: Each chunk is converted into an embedding using Hugging Face's model.
4. **Vector Storage**: Embeddings are stored in a FAISS vector database.
5. **Question Answering**: When a question is asked, the system retrieves the most relevant chunks and uses Cohere to generate an answer based on the context.

## Example Use Case
```python
ans = generate_answer("Methods to handle missing values?")
print(ans)
```
**Output**:  
*The provided context mentions two methods to handle missing values, COALESCE() and IFNULL(). COALESCE() returns the first non-NULL value from a list of arguments, while IFNULL() replaces NULL with a specified value...*

## Setup
1. Install the required libraries:
   ```bash
   pip install PyPDF2 langchain langchain_community langchain_google_genai langchain_text_splitters sentence-transformers faiss-cpu cohere
   ```
2. Add your Cohere API key in the code.
3. Place your PDF file in the specified path and run the notebook.

## Ideal For
- Extracting insights from technical documents, manuals, or reports.
- Building a context-aware Q&A system for educational or professional use.


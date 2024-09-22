# AI-Powered RAG 
This project is an interactive Streamlit app that allows users to upload a PDF document and ask questions about the contents of the PDF using AI models. The app leverages the power of Retrieval-Augmented Generation (RAG) with LLMs to provide accurate and context-based answers from the uploaded document.

![Screenshot (181)](https://github.com/user-attachments/assets/8f734395-c6dd-4811-aaed-67e243ce1a0c)

## Features
- PDF Ingestion: Upload any PDF file to the application.

- Chunking and Embedding: The uploaded PDF is split into chunks, and embeddings are created for each chunk to store in a vector store.

- Question Answering: Ask questions based on the contents of the uploaded PDF, and the model will retrieve relevant information to generate an answer.

- LLM Powered: Uses the `ChatOllama` language model for generating responses and Hugging Face embeddings to embed the PDF content.

- Vector Store: Efficiently stores and retrieves document embeddings using Chroma as the vector store.

- RAG Architecture: Combines document retrieval and language models to answer questions based on the document content.

##  How It Works
- Upload a PDF: The user uploads a PDF, which is processed by the app.
- PDF Processing: The document is loaded and split into manageable text chunks using `RecursiveCharacterTextSplitter`.
- Vectorization: These chunks are then embedded using Hugging Face Embeddings and stored in a Chroma vector database.
- Question Answering: When a user inputs a question, the app uses the vector store to retrieve relevant document chunks. These chunks are passed to the `ChatOllama` model to generate a response based on the context.

 ## Project Structure
- `app.py`: The main Streamlit app file containing all the functionality for PDF ingestion, embedding, vector store creation, and question answering.
- `chroma_db/`: The directory where the vector store embeddings are stored.

##  Installation
To run the app locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/bhudil/RAG_QnA.git
cd your-repo
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up access to Hugging Face:

You'll need a Hugging Face account with access tokens for using models and embeddings 

from- [HuggingFaceHub](https://huggingface.co/)

4. Ollama Installation:
   Pre-install a model like mistral or Llama3.1 on your local machine
   
  from- [OLLAMA](https://github.com/ollama/ollama)

6. Run the Streamlit app:
```
streamlit run app.py
```

## Dependencies
The following dependencies are required to run the app:

- streamlit: Frontend framework for creating interactive web applications.
- langchain_community: A set of tools for integrating language models into your workflow.
- Chroma: Vector store for efficiently retrieving document embeddings.
- PyPDFLoader: PDF document loader for processing PDFs.
- HuggingFaceEmbeddings: Embedding tool for turning document text into numerical embeddings using Hugging Face models.
- ChatOllama: A language model for generating responses.
- huggingface_hub: Provides access to Hugging Face API for model and embedding handling.
- Install all these dependencies using the requirements.txt file.

##  Usage
Upload PDF: Choose a PDF file using the upload button.

Process PDF: Click on the "Process PDF" button to split the PDF into chunks and store the embeddings in a Chroma vector store.

Ask Questions: After processing, enter your question in the input field, and click "Get Answer". The app will retrieve relevant document chunks and generate an answer using the language model.

## Example

![Screenshot (180)](https://github.com/user-attachments/assets/382f90b2-6669-40cd-956e-3669919da3c9)

## Limitations
The app currently processes only PDF files.

Large PDFs might take a while to process due to chunking and embedding steps.

The model response is limited to the content of the uploaded PDF, and irrelevant or unrelated questions may result in an "No Context available" response.


## Future Enhancements
Support for additional document formats (e.g., .docx, .txt).

Improved retrieval mechanism to handle larger documents and improve answer accuracy.

Addition of more advanced language models for better question-answering capabilities.


## Contributing
Contributions are welcome! Please submit a pull request with any improvements or suggestions for future updates.

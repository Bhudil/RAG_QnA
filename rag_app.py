import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from huggingface_hub import login
from io import BytesIO
import tempfile

# Login to Hugging Face
access_token_read = "YOUR_HF_API_KEY"
access_token_write = "YOUR_HF_API_KEY(SAME)"
login(token=access_token_read)

st.set_page_config(page_title="Chatbot App", page_icon=":robot_face:")

# Streamlit app layout
st.title("AI-Powered RAG 🤖")
st.write("Upload a PDF and ask questions based on the document leveraging the power of AI! ")
st.markdown("---")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file to upload", type="pdf")

# Ingestion function
def ingest(file):
    with st.spinner(" Ingesting and processing the PDF..."):
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        # Load the PDF from the temporary file path
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()
        
        # Split the pages by character
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(pages)
        st.success(f" Split {len(pages)} documents into {len(chunks)} chunks!")
        
        embedding = HuggingFaceEmbeddings()
        
        # Create and persist the vector store
        Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./chroma_db")
        st.success("✅ PDF processing complete and embeddings stored!")

# Chain setup function
def rag_chain():
    model = ChatOllama(model="llama3.1")
    
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a creative assistant. Answer the question in detail based only on the following context in a neat format of bullet points. 
        If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
    )
    
    embedding = HuggingFaceEmbeddings()
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding)

    # Create the retrieval chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5
        }
    )
    
    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    return chain

# Question answering function
def ask(query: str):
    chain = rag_chain()
    result = chain.invoke({"input": query})
    
    return result["answer"], result["context"]

# PDF ingestion button
if uploaded_file is not None:
    if st.button("Process PDF"):
        ingest(uploaded_file)

# Query input
query = st.text_input("💬 Ask a question :")

# Process the question and display results
if st.button(" Get Answer"):
    if uploaded_file is None:
        st.error("Please upload and process a PDF first!")
    elif not query:
        st.warning("Please enter a question!")
    else:
        with st.spinner(" Thinking..."):
            answer, context = ask(query)
            st.write("### 💡 Answer:")
            st.write(answer)


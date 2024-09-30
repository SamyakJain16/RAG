import streamlit as st
import pypdf
from pypdf import PdfReader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter your OpenAI API key: ")


# Streamlit GUI
def main():
    st.title("PDF Conversational Chatbot")

    # File uploader for PDF documents
    uploaded_pdfs = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_pdfs:
        # Process uploaded PDFs
        pdf_text = get_pdf_text(uploaded_pdfs)
        text_chunks = get_text_chunks(pdf_text)

        # Generate vectorstore
        vectorstore = get_vectorstore(text_chunks)

        # Initialize conversation chain
        conversation_chain = get_conversation_chain(vectorstore)

        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []  # Initialize chat history

        # User input for conversation
        user_input = st.text_input("Your Question:")
        if user_input:
            # Generate and display response
            response = conversation_chain.run(user_input)
            # Update chat history
            st.session_state["chat_history"].append(("User", user_input))
            st.session_state["chat_history"].append(("Bot", response))

        # Display chat history
        if st.session_state["chat_history"]:
            st.write("### Chat History:")
            for i, (speaker, message) in enumerate(st.session_state["chat_history"]):
                st.write(f"**{speaker}:** {message}")

# Extracts and concatenates text from a list of PDF documents


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure non-empty text
                text += page_text
    return text

# Splits a given text into smaller chunks based on specified conditions


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Generates embeddings for given text chunks and creates a vector store using FAISS


def get_vectorstore(text_chunks):
    # Using OpenAIEmbeddings instead of SentenceTransformerEmbeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initializes a conversation chain with a given vector store


def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        # Static temperature value set to 0 (deterministic responses)
        llm=ChatOpenAI(),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h: h,
        memory=memory
    )
    return conversation_chain


if __name__ == '__main__':
    main()

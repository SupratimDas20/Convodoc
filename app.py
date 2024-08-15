import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set.")
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if not text_chunks:
        raise ValueError(
            "Text chunks are empty. Please provide valid text for embedding."
        )
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except IndexError as e:
        st.error(f"Error during vectorstore creation: {e}")
        st.error(f"Text chunks: {text_chunks}")
        raise ValueError(
            "Embedding generation failed, likely due to empty or invalid input text."
        )
    return vectorstore


def get_conversation(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation(
        {"question": user_question, "chat_history": st.session_state.chat_history}
    )
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with files", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize as an empty list

    st.header("Chat with files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            handle_userinput(user_question)
        else:
            st.error("Please process your documents first.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # get file text
                    raw_text = get_pdf_text(pdf_docs)

                    if raw_text.strip():
                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        try:
                            vectorstore = get_vectorstore(text_chunks)
                            st.success("Vectorstore created successfully!")

                            # create conversation chain
                            st.session_state.conversation = get_conversation(
                                vectorstore
                            )
                        except ValueError as e:
                            st.error(f"Processing failed: {e}")
                    else:
                        st.error(
                            "No text could be extracted from the PDF files. Please check the files and try again."
                        )
            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()

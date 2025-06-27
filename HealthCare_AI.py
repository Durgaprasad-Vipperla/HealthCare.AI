import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from CanvasLLMWrapper import CanvasLLM
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

# To log conversation history to a CSV file for RAGAS
def log_chat_to_excel(user_question, context, response):
    file_path = "chat_history.xlsx"

    new_data = pd.DataFrame([{
        "User Question" : user_question,
        "Context": context,
        "LLM Response": response
    }])

    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, mode='a', if_sheet_exists="overlay", engine="openpyxl") as writer:
            new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        new_data.to_excel(file_path, index=False, engine="openpyxl")

# To load and extract input documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# To split the docs into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks
 
# Configuring the embedding model
def get_embedding_model():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

# Configuring the LLM model
def get_llm():
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # Google_Model = ChatGoogleGenerativeAI(
    #     model="gemini-pro",
    #     temperature=0.1
    # )

    Canvas_Model = CanvasLLM()
    return Canvas_Model

# Initiation of the FAISS Vector Database
def get_vector_store(text_chunks, embedding_model):
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")

# Configuring the retriever to retriever the appropriate chunks from the Vector DB
def get_retriever(embedding_model):
    faiss_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = faiss_db.as_retriever(search_type = "similarity_score_threshold", search_kwargs = {"score_threshold": 0.2, "k": 5})
    return retriever

# Helper funcion to format the retrieved chunks
def format_docs(docs):
    return "\n|||\n".join(doc.page_content for doc in docs)

# Defining our memory aware chain to get the final answer
def get_memory_llm_chain(llm, retriever, memory):
    prompt_template = """
    You are an AI-powered medical assistant. You will aid users with their queries based on the provided context and previous chat history. Only provide precise answers to the user's questions without adding any irrelevant information.
    The context, chat_history and user query are given within the angled brackets notations below.
    Reorganize your answer in a way that is easy to read and visually appealing to the user.
    
    <Previous Conversation>
    {chat_history}
    </Previous Conversation>

    <Question> {question} </Question>

    <context>
    {context}
    </context>

    Note:
    1. If you can't find the relevant answer within the provided context, politely respond to the user that you are not able to find the answer in the given context.
    2. If the user asks any questions that are outside the provided context, respond politely by saying that the query is outside the scope of the given context.
    3. If the user requests any medical advice that requires professional judgment, advise them to consult a healthcare professional.
    4. Ensure all responses are clear, concise, and tailored to the user's question.
    """

    prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=prompt_template)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        combine_docs_chain_kwargs = {"prompt" : prompt}, 
        return_source_documents = True
        # , verbose = True
    )

    return conversation_chain

def main():
    llm = get_llm()
    embedding_model = get_embedding_model()

    st.set_page_config(page_title="HealthCare.AI", layout="wide")

    # CSS styling for the UI
    st.markdown(
        """
        <style>
        .chat-input {
            position: fixed;
            top: 120px;
            left: 0;
            right: 0;
            background: white;
            z-index: 1000;
            padding: 10px 20px;
        }

        .main-title {
            position: fixed;
            top: 7%;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            background-color: white;
            width: 100%;
            padding: 10px 0;
            font-size: 2em;
            font-weight: bold;
            z-index: 1000;
        }

        .sidebar-content {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            margin-top: 10vh;
        }

        .sidebar-menu {
            display: flex;
            justify-content: center;
            width: 100%;
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 80px 20px 20px;
            max-height: 70vh;
            overflow-y: auto;
            margin-top: -30px;
        }

        .message-row {
            display: flex;
            width: 100%;
        }

        .message-bubble {
            display: inline-block;
            padding: 10px 15px;
            border-radius: 18px;
            margin: 5px;
            word-wrap: break-word;
            max-width: 60%;
        }

        .user-row {
            justify-content: flex-end;
        }
        .user-message {
            background-color: #0084ff;
            color: white;
            text-align: right;
        }

        .bot-row {
            justify-content: flex-start;
        }
        .bot-message {
            background-color: #e5e5ea;
            color: black;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='main-title'>⚕️ HealthCare.AI</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sidebar-menu'>📂 File Uploader</h2>", unsafe_allow_html=True)

        pdf_docs = st.file_uploader(" ", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, embedding_model)
                st.success("processing complete!")
        
        st.markdown("</div>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
    user_question = st.text_input("How may I assist you?")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_question:
        retriever = get_retriever(embedding_model)
        mem_chain = get_memory_llm_chain(llm, retriever, st.session_state.memory)
        mem_response = mem_chain.invoke({"question":user_question})
        # context = [i.page_content for i in mem_response['source_documents']]
        context = format_docs(mem_response['source_documents'])
        mem_response = mem_response['answer']

        # Store user query and response in session state
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", mem_response))

        # Storing data for RAGAS
        log_chat_to_excel(user_question, context, mem_response)

        # Display chat history
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for role, message in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"<div class='message-row user-row'><div class='message-bubble user-message'>{message}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='message-row bot-row'><div class='message-bubble bot-message'>{message}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
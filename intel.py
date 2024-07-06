import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()


##Load the GROQ API Key
groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embedding = OllamaEmbeddings()
    st.session_state.loader = CSVLoader("IndianFoodDatasetCSV.csv")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents[:25], st.session_state.embedding)


st.title("Cheffy - Your AI Kitchen Assistant")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama3-8b-8192")


prompt = ChatPromptTemplate.from_template("""
You are a Kitchen Chatbot. Give answers to the user only based on the given context. You have to provide the user with proper guidance and suggestions on how to make a dish with given ingredients, and for how many people. Use your capability to intelligently answer user's queries. Please refrain yourself from answering any other questions that are not related to cooking, kitchen, food, ingredients etc. 
                                          <context>{context}</context>
                                          Question : {input}

""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Ask your queries to Cheffy")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input" : prompt})
    print("Response time : ",time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("______________________________________")







    



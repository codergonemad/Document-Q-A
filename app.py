import streamlit as st
from dotenv import load_dotenv
# import dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.llms import OpenAI
import os

def get_text_chunks(documents,chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
        # separators=['.','\n\n']
    )
    return text_splitter.split_documents(documents)

# vector=None
def get_vectorstore(text_chunks):
    embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
    )
    vectordb = Chroma.from_documents(
    documents=text_chunks, embedding=embeddings
    )
    # vector=vectordb
    return vectordb


def get_conversation_chain(vectorstore):
    llm = AzureOpenAI(
    deployment_name="gpt-35-turbo",
    model_name="gpt-35-turbo",
    temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
        # chain_type="map_reduce"
    )
    # conversation_chain=RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank",retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    # print(response)
    st.session_state.chat_history = response['chat_history']

    i=len(st.session_state.chat_history)-1
    while i>=0:
        st.write(user_template.replace(
                "{{MSG}}", st.session_state.chat_history[i-1].content), unsafe_allow_html=True)
        similar_docs=st.session_state.vector.similarity_search(st.session_state.chat_history[i-1].content)
        temp="Reference Docs:\n\n"
        print()
        myset=set()
        for doc in similar_docs:
            curr="pageno: "+str(doc.metadata['page'])+"     source: "+doc.metadata['source']
            if curr not in myset:
                temp+="pageno: "+str(doc.metadata['page'])+"    source: "+doc.metadata['source']+"\n\n"
            myset.add(curr)
        st.write(bot_template.replace(
                "{{MSG}}", st.session_state.chat_history[i].content.replace(">","&gt;").replace("<","&lt;")+"\n\n"+temp), unsafe_allow_html=True)
        print(st.session_state.chat_history[i].content)
        i-=2

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Document Question Answering",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        loader = PyPDFDirectoryLoader("data")
        documents = loader.load_and_split()
        if st.button("Process"):
            with st.spinner("Processing"):              
                text_chunks = get_text_chunks(documents)
                
                vectorstore = get_vectorstore(text_chunks)
                # vector=vectorstore
               
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.session_state.vector=vectorstore
                # print(st.session_state.conversation)

if __name__ == '__main__':
    main()
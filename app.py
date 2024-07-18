from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import joblib
import pymilvus
import json
from pymilvus import MilvusClient
import streamlit as st
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



milvus_client = MilvusClient(host='172.19.14.186',port=19530)
collection_name = "triology1"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")

def give_context(question,collection_name,milvus_client):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            GoogleGenerativeAIEmbeddings(model="models/text-embedding-004").embed_query(question)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=5,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
    return context



# Pull the prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# Post-processing function
def format_docs(context):
    return context

rag_chain = (
    {"context": RunnablePassthrough() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit interface
st.title("Amish Triphathi Triology books (RAPTOR RAG)")
# User input
question = st.text_input("Enter your question:")
# Button to invoke the chain
if st.button("Get Answer"):
    if question:
        # Invoke the chain with the question
        answer = rag_chain.invoke({"context": give_context(question,collection_name=collection_name,milvus_client=milvus_client
                                                           ), "question": question})
        
        # Display the answer
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")

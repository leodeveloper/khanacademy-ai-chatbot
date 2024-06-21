
import sqlite3

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import json
from datetime import datetime
from langchain_chroma import Chroma
#from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import chromadb
import chromadb.config

st.set_page_config(page_title="BBC youtube channel Generative ai chatbot", page_icon="")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=1,model="gemma-7b-it")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_llm_response(llm_response):
    st.write(llm_response['result'])
    st.write('\n\nSources:')
    for source in llm_response["source_documents"]:
        st.write(source.metadata['source'])

template = """You are a help full BBC helpfull assistant. please include the youtube url, publish date and source if available. 
Don't try to make up an answer. Also do not add any information which is not available in the context.

{context}

Question: {question}

Helpful Answer:
"""

def loadModel(question):
    try:
        question = f"{question}"
        st.write(f"You asked: {question}")
        modelname=os.getenv("modelname")
        collectionname=os.getenv("collectionname")

        embedding_function = HuggingFaceEmbeddings(model_name=modelname)
        db=Chroma(collection_name=collectionname,embedding_function=embedding_function,persist_directory="embeding/chromadb")
        retriever = db.as_retriever(search_type="mmr",search_kwargs={'k':1})
        #search_kwargs={'k':1}
        #"What donalod trump said about the FBI raid? and what happened in pakistan floods?"
        #response2=retriever.invoke(question)
        #st.write(response2[0].page_content)



        llama_prompt = PromptTemplate(template=template, input_variables=["text"])

        chain_type_kwargs = {"prompt": llama_prompt}

        # create the chain to answer questions
        #qa_chain = RetrievalQA.from_chain_type(llm=llm,
        #                                chain_type="stuff",
        #                                retriever=retriever,
        #                                return_source_documents=True)

        #chain = prompt | llm
        #response=qa_chain.invoke(question)
        #process_llm_response(response)
        st.write("--------------------------------------------")
        custom_rag_prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
            )
        
        response=rag_chain.invoke(f"{question} please must include the youtube url and publish date in the answer")
        st.write(response)
        #return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Failed to load the model or process the request. Please check the logs for more details.")


# Load the JSON data
with open('combined_extracted_data.json', 'r') as file:
    data = json.load(file)

# Convert publish_date to datetime and sort by publish_date descending
for item in data:
    item['publish_date'] = datetime.strptime(item['publish_date'], "%Y-%m-%d %H:%M:%S")
data = sorted(data, key=lambda x: x['publish_date'], reverse=True)

# Add custom CSS to limit the title to two lines
st.markdown("""
    <style>
    .truncate-title {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """, unsafe_allow_html=True)

# Define the page size
PAGE_SIZE = 9

# Streamlit app
st.title("BBC Youtube Channel")

# Big question bar
st.write("Generative AI BBC youtube channel chatbot.")
st.write("Last update on 18 June 2018, only 4588 video's are available for question and answers")
question = st.text_input("Ask any question regarding the bbc news youtube channel video's.")

# Submit button
if st.button("Submit"):
    with st.spinner("Waiting......"):
        #st.write(f"You asked: {question}")
        loadModel(question)
    



# Display images in a grid
st.subheader("BBC youtube video's")
page_number = st.number_input('Page Number', min_value=1, max_value=(len(data) // PAGE_SIZE) + 1, value=1)
start_index = (page_number - 1) * PAGE_SIZE
end_index = start_index + PAGE_SIZE

for i, item in enumerate(data[start_index:end_index]):
    if i % 3 == 0:
        cols = st.columns(3)
    cols[i % 3].image(item['thumbnail_url'])
    cols[i % 3].markdown(f"<div class='truncate-title'>{item['title']}</div>", unsafe_allow_html=True)
    cols[i % 3].write(item['publish_date'].strftime("%d %b %Y"))
    

# If there are more than PAGE_SIZE items, add pagination
if len(data) > PAGE_SIZE:
    st.write("Pagination:")
    total_pages = (len(data) // PAGE_SIZE) + 1
    st.write(f"Page {page_number} of {total_pages}")

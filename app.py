import os
import streamlit as st
import json
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone
from utils_dycrypt import decrypt_string
import pprint



st.set_page_config(page_title="khanacademy youtube channel Generative ai chatbot", page_icon="")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()
os.environ['PINECONE_API_KEY']=os.getenv("PINECONE_API_KEY1")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0.5,model="llama3-70b-8192")
key=os.getenv("encryptkey")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_llm_response(llm_response):
    st.write(llm_response['result'])
    #st.write("---------------------------")

qa_template = """You are a youtube video's transcript expert. Use the given context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
must add youtubelink, publish date and title.

Context: {context}

Question: {question}
Answer:
"""

def loadModel(question):
    try:
        question = f"{question}"
        st.write(f"You asked: {question}")
        modelname=os.getenv("modelname")
        pc = Pinecone()
        index_name=os.getenv("pinecode_index_name")
        embedding_function = HuggingFaceEmbeddings(model_name=modelname)
        docsearch=PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embedding_function)
        
        retriever = docsearch.as_retriever(search_type="mmr",search_kwargs={'k':1})
        # Define the prompt template for Q&A
        qa_prompt_template = PromptTemplate.from_template(qa_template)

        # Define the RetrievalQ&A chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt_template},
        )


        # Perform retrieval Q&A
        response=qa_chain({"query": f"{question}"})
        process_llm_response(response)
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
st.title("Generative AI khanacademy youtube channel chatbot")

# Big question bar
st.write("Last update on 24 June 2024, 4583 video's are available for question and answers")
st.write("This project is not yet funded and using very minimal resources")
question = st.text_input("Ask any question regarding the khanacademy youtube channel video's.")

# Submit button
if st.button("Submit"):
    with st.spinner("Please wait......"):
        #st.write(f"You asked: {question}")
        loadModel(question)
    



# Display images in a grid
st.subheader("khanacademy youtube video's")
st.write("For full transcripts in English and other languages, email me at leodeveloper@gmail.com.")
page_number = st.number_input('Page Number', min_value=1, max_value=(len(data) // PAGE_SIZE) + 1, value=1)
start_index = (page_number - 1) * PAGE_SIZE
end_index = start_index + PAGE_SIZE

for i, item in enumerate(data[start_index:end_index]):
    if i % 3 == 0:
        cols = st.columns(3)
    cols[i % 3].image(decrypt_string(item['thumbnail_url'],key))
    cols[i % 3].markdown(f"<div class='truncate-title'>{item['title']}</div>", unsafe_allow_html=True)
    cols[i % 3].write(f"{item['publish_date'].strftime('%d %b %Y')} - <a href='https://youtubetranslate.streamlit.app/?source={item['source']}'>Translate</a>", unsafe_allow_html=True)
    

# If there are more than PAGE_SIZE items, add pagination
if len(data) > PAGE_SIZE:
    st.write("Pagination:")
    total_pages = (len(data) // PAGE_SIZE) + 1
    st.write(f"Page {page_number} of {total_pages}")


# Footer
st.markdown('---')
st.write('All copyrights are reserved 2024. For more information, contact [leodeveloper@gmail.com](mailto:leodeveloper@gmail.com).')




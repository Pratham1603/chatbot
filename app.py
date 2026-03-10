from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_API_KEY_2 = os.getenv("HUGGINGFACE_API_KEY_2")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
os.environ["HUGGINGFACE_API_KEY_2"] = HUGGINGFACE_API_KEY_2

embeddings = download_embeddings()

index_name = "mokshfit"

# Load the existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=512,
    temperature=0.2,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY_2 # Add this parameter
)

chat_model = ChatHuggingFace(llm=endpoint)

PROMPT = PromptTemplate(
    template=prompt_template + "\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model , # <--- Using native ChatHuggingFace
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

#####################################################################################

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])

def chat():
    
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    response = qa_chain.invoke({"query": msg})
    result = response["result"]  # ← move this BEFORE jsonify
    
    print("Response : ", result)
    return jsonify({"answer": result})

#####################################################################################

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
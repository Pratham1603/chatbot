from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.schema import Document
import re
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Extracting data from JSON file and creating documents
def load_json_file(file_path):
    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:

        # ---- Minimal Metadata ----
        metadata = {
            "handle": item.get("Handle"),
            "title": item.get("Title"),
            "type": item.get("Type"),
            "price": item.get("Variant Price"),
            "published": item.get("Published"),
            "Image URL": item.get('Image Src')
        }

        # ---- Rich Page Content ----
        page_content = f"""
        Product Title: {item.get('Title')}
        Product Handle: {item.get('Handle')}
        Product Type: {item.get('Type')}
        Vendor: {item.get('Vendor')}
        Price: {item.get("Variant Price")}
        Color: {item.get('Color')}
        Size: {item.get('Size')}
        Fabric Type: {item.get('Fabric_Type')}
        GSM: {item.get('GSM')}
        Fit Type: {item.get('Fit_Type')}
        Care Instructions: {item.get('Care_Instructions')}
        Tags: {item.get('Tags')}
        Image Position: {item.get('Image Position')}
        Image Alt Text: {item.get('Image Alt Text')}
        """

        documents.append(
            Document(
                metadata=metadata,
                page_content=page_content.strip()
            )
        )

    return documents


# 2. Since the page content is not cleaned like \n next page , etc...
def clean_product_text(text):

    # --- FIX WORDS BROKEN BY HYPHEN + NEWLINE ---
    text = re.sub(r'(\w+)-\s*[\r\n]+\s*(\w+)', r'\1\2', text)

    # --- PROTECT PARAGRAPHS ---
    text = text.replace('\n\n', '[[PARA]]')

    # --- REMOVE SINGLE NEWLINES/TABS ---
    text = text.replace('\n', '.').replace('\r', ' ').replace('\t', ' ')

    # --- RESTORE PARAGRAPHS ---
    text = text.replace('[[PARA]]', '\n\n')

    # --- REMOVE EXTRA SPACES ---
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# 3. Clean the page content of all documents
def clean_documents(documents):

    for doc in documents:
        doc.page_content = clean_product_text(doc.page_content)

    return documents

# 4. Download Embeddings Model  
def download_embeddings():
    # Using langchain_huggingface as required by your requirements.txt
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
    )
    return embeddings
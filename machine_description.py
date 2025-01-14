**Data pipeline that extracts and cleans information from PDF files and an Excel spreadsheet about machines. This data is then processed and indexed in a Pinecone vector store Database, allowing for advanced querying and interaction through an AI chatbot specialized in generating machine description.**
# Importing Libraries
!pip install PyPDF2
!pip install pinecone-client
!pip install langchain-pinecone
!pip install langchain-community
!pip install langchain_groq
import pandas as pd
import os
import PyPDF2
from pinecone import Pinecone
from pinecone import ServerlessSpec
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tqdm.autonotebook import tqdm
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
df=pd.read_excel(r"MachineTbl_Data.xlsx")
df
df1=df[["MachineCategoryId, 
"Another manufacturer" ,
"Another machine model",
"MachineTypeId",
"Year",
"MachineConditionId", 
"MachineAvailabilityId", 
"Location",
"Description"]]
# Define mappings
availability_mapping = {
    1: "Immediately",
    2: "Less than 30 Days",
    3: "More than 30 Days",
    4: "Immediately from stock"
}

condition_mapping = {
   1: "InStock",
   2: "Running",
   3: "Rebuilt",
   4: "In Transit",
   5: "In Production",
   6: "Excellent"
}
df1['MachineAvailabilityId'] = df1['MachineAvailabilityId'].map(availability_mapping).fillna(df1['MachineAvailabilityId'])
if condition_mapping:
    df1['MachineConditionId'] = df1['MachineConditionId'].map(condition_mapping).fillna(df1['MachineConditionId'])
manufacturers_df = pd.read_excel(r"MasterManufacture.xlsx")
models_df = pd.read_excel(r"model_mapping.xlsx")
models_df.drop(["categorys id","ManufacturerId"],inplace=True,axis=1)
models_df.duplicated().sum()
df1=pd.merge(df1,manufacturers_df,how='left',left_on='ManufacturerId',right_on='ManufacturerId')
df1=pd.merge(df1,models_df,how='left',left_on='MachineModelId',right_on='MachineModelId')
df1['Manufacturer_name'] = df1['Manufacturer_name'].fillna('').astype(str)
df1['Model Name'] = df1['Model Name '].fillna('').astype(str)
df1['Machine_Name'] = df1['Manufacturer_name'] + " " + df1['Model Name']
# PDF EXTRACTION
main_folder_path = r'C:\Users\Nenya\Desktop\Machine dala\Machine_dala_PDFs\PDFs'

all_text = ""

for root, dirs, files in os.walk(main_folder_path):
    for file in files:
        if file.endswith('.pdf'):
            pdf_file_path = os.path.join(root, file)
            print(f"Extracting text from: {pdf_file_path}")

            with open(pdf_file_path, 'rb') as file_obj:
                reader = PyPDF2.PdfReader(file_obj)
                for page in reader.pages:
                    all_text += page.extract_text() + "\n"

with open('extracted_text_all.txt', 'w', encoding='utf-8') as f:
    f.write(all_text)
lines = all_text.splitlines()
cleaned_lines = []
patterns_to_remove = [
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    r'©|\u00A9|\u00AE|®|™',
    r'^[0-9\-\.]+$',
    r'Page \d+( of \d+)?',
    r'^\s*\d+\s*$',
    r'www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}',
    r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
]
combined_pattern = '|'.join(patterns_to_remove)
for line in lines:
    line = line.strip()
    if not line:
        continue

    line = re.sub(combined_pattern, '', line, flags=re.IGNORECASE)

    if len(line.strip()) < 2:
        continue

    if re.match(r'^\s*\d+(\.\d+)?\s*(mm|cm|in|ft)\s*[×x]\s*\d+(\.\d+)?\s*(mm|cm|in|ft)\s*$', line):
        continue

    if re.match(r'^\s*(Phone|Tel|Fax|Mobile)[:.]?\s*[\+]?[\d\-\s\(\)]+$', line, re.IGNORECASE):
        continue
    if re.match(r'^\s*\d{5,}([- ]?\d{4,})?\s*$', line):
        continue

    if re.match(r'^\s*(Version|Date|Rev|Updated?)?\s*:?\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\s*$', line, re.IGNORECASE):
        continue

    if re.match(r'^\s*\d+(\.\d+)?\s*(kg|g|oz|lb|m|ft|in|cm|mm|mL|L|gal|mph|km/h)\s*$', line, re.IGNORECASE):
        continue

    if any(header in line.lower() for header in ['confidential', 'all rights reserved', 'proprietary']):
        continue

    line = ' '.join(line.split())

    if line.strip():
        cleaned_lines.append(line)

cleaned_text = '\n'.join(cleaned_lines)
cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
cleaned_text = cleaned_text.strip()
with open('extracted_text_all.txt', 'w', encoding='utf-8') as f:
    f.write(cleaned_text)
with open('extracted_text_all.txt', 'r') as f:
    clean=f.read()
# VECTOR DATABASE
**Use Only When database is created for first time .Otherwise delete the database first and then run this code**
# pc.create_index(name='Aiproject', dimension=384, metric="cosine",spec=ServerlessSpec(cloud='aws',region='us-east-1'))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)

doc = Document(page_content=cleaned_text)
docs=text_splitter.split_documents([doc])

import os
os.environ['PINECONE_API_KEY'] = 'pineconeapi'
# docs

# docsearch=PineconeVectorStore.from_documents(docs,embeddings, index_name="Aiproject")

**Use Only when database is created and you are running the code again**
index=pc.Index('machine-dala')
emptydocs=[]
docsearch=PineconeVectorStore.from_documents(emptydocs,embeddings, index_name="machine-dala")

retriever = docsearch.as_retriever()
grok_api="grok_api"
llm=ChatGroq(
    api_key=grok_api,
    model="llama-3.2-3b-preview",
    temperature=0,
)
# PROMPT
description_template = ChatPromptTemplate([
    ("system", """You are Avesh, an AI assistant who is an expert in printing and packaging machinery. Write in a friendly, authentic way and description is for an online platform that deals with print and packaging machinery. Keep it natural and engaging while sharing your expertise. Write a single flowing paragraph of at least 200 words.

    Style guide:
    - Give a description of the machine in detail
    - Focus more on machine specifications and features
    - Maintain industry credibility
    - The description is for an online platform that deals with print and packaging machinery
    - Add helpful context about what makes this machine special in everyday use

    Context: {context}

    Remember:
    - Keep it flowing as one paragraph
    - Include actual usability and performance details
    - Minimum 200 words
    - Stay authentic but professional
    - Do not include any unknown information"""),
    ("human", "Tell me about the {machine_name} from {manufacturer}."),
])

pdf_based_template = """
Based on the following context about a machine and its details, generate a comprehensive description:
Context from documentation: {context}
Machine Details:
- Manufacturer: {manufacturer}
- Model: {model}
- Condition: {condition}
- Availability: {availability}
- Year: {year}
- Location: {location}

Generate a detailed description of this machine, incorporating both the context information and the provided details.
"""

llm_only_template = """
Generate a comprehensive description for {machine_name} from this {manufacturer}:

Focus on typical capabilities, features, and applications of this type of machine, considering the manufacturer's reputation and the model specifications.
-The description is for an online platform that deals with print and packaging machinery
-Keep it flowing as one paragraph
-Minimum 200 words
"""

pdf_prompt = ChatPromptTemplate.from_template(pdf_based_template)
llm_prompt = ChatPromptTemplate.from_template(llm_only_template)
def generate_authentic_description(row):
    machine_name = row['Machine_Name']

    search_results = retriever.get_relevant_documents(machine_name)

    relevant_contexts = []
    for doc in search_results:
        if is_relevant(machine_name, doc.page_content):
            relevant_contexts.append(doc.page_content)

    context = " ".join(relevant_contexts)

    if context:
        messages = description_template.format_messages(
            context=context,
            machine_name=machine_name,
            manufacturer=row['Manufacturer_name'],
            model=row['Model Name'],
            condition=row['MachineConditionId'],
            availability=row['MachineAvailabilityId'],
            year=row['Year'],
            location=row['Location']
        )
        template_type = "pdf"
    else:
        messages = llm_prompt.format_messages(
            machine_name=machine_name,
            manufacturer=row['Manufacturer_name'],
            model=row['Model Name'],
            condition=row['MachineConditionId'],
            availability=row['MachineAvailabilityId'],
            year=row['Year'],
            location=row['Location']
        )
        template_type = "llm"

    response = llm.invoke(messages)
    description = response.content

    return pd.Series({'Generated_Description': description, 'Template_Type': template_type})
# Set the number of rows to sample
df_sample = df1.copy()
print("Generating authentic machine descriptions...")
df_sample[['Generated_Description', 'Template_Type']] = df_sample.apply(generate_authentic_description, axis=1)
chain = llm | pdf_prompt
df_sample.to_excel("trial11.xlsx")
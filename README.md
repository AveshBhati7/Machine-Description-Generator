# Machine-Description-Generator

Overview
This project implements a data pipeline that extracts and cleans information from PDF files and an Excel spreadsheet about machines. The cleaned data is then processed and indexed in a Pinecone vector store database, enabling advanced querying and interaction through an AI chatbot specialized in generating machine descriptions.

Features
Data Extraction: Extracts text from PDF files and reads data from Excel spreadsheets.
Data Cleaning: Cleans the extracted text to remove irrelevant information and noise.
Data Processing: Merges and processes data from multiple sources, including mapping machine conditions and availability.
Vector Database: Indexes the processed data in a Pinecone vector store for efficient querying.
AI Chatbot: Utilizes a language model to generate detailed machine descriptions based on user queries.

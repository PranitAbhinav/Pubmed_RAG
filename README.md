# PubMed RAG with Qdrant + AWS Bedrock

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** for medical research papers.  
It uses **Amazon Bedrock** for embeddings and summarization, and **Qdrant** as a vector database for semantic search.  

The goal: search PubMed abstracts/paragraphs stored in Qdrant and generate concise, medically accurate summaries.

---

## 🚀 Features
- **Chunk & Embed**: Text is split into chunks and embedded using **Amazon Titan v2** embeddings.
- **Vector Search**: Qdrant is used for similarity search over PubMed chunks.
- **Summarization**: Retrieved documents are summarized using **Anthropic Claude-v2** via Bedrock.
- **CLI Tool**: Query PubMed-like data directly from the terminal.

---

## 📂 Project Structure


├── query.py # CLI for search & summarization of answers \
├── generate_embeddings.py # use bedrock agent to generate embedings and push to Qdrant db \
├── pubmed_data.py # script for ingesting CSV data and parse paper content from xml files\
├── awsauth.py # Stores credentials & API keys (gitignored, not committed)\
└── README.md

---

## 🖥️ Usage
### 1. Ingest PubMed data into Qdrant



#### If you have a CSV file with paper content (e.g. pubmed_with_paragraphs.csv):

```
python3 generate_embeddings.py
```

This will:
Create/recreate a Qdrant collection.
Chunk paragraphs into ~500 words.
Embed with Titan-v2.
Upload embeddings to Qdrant.

### 2. To run a query and get responses from the summarization ,model:

```
python3 query.py --query "What are the latest findings about nab-paclitaxel therapy?" --collection <your qdrant db collection> --top-k 5
```



## 🔑 Authentication

Inside awsauth.py enter your AWS credentials and Qdrant db credentials : 
``` 
AWS_ACCESS_KEY_ID = "your-aws-key"
AWS_SECRET_ACCESS_KEY = "your-aws-secret"
AWS_SESSION_TOKEN = "your-session-token"   # if applicable

qdranturl = "https://your-qdrant-instance"
qdrantapikey = "your-qdrant-api-key"

```


## 🔮 Roadmap

 Add support for Claude v3 or other LLMs

 Integrate FastAPI for a web service

 Extend ingestion for full-text PDFs via PubMed Central



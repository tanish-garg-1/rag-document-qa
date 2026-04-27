\# 📚 RAG Document Question Answering System



A production-style \*\*Multimodal Conversational RAG\*\* system built with FastAPI, FAISS, Gemini, and Groq.



\---



\## 🧠 System Overview



Upload documents (PDF, DOCX, TXT, Images) and ask questions about them in a conversational interface. The system uses \*\*Retrieval Augmented Generation (RAG)\*\* to find relevant content and generate accurate, cited answers.



\---



\## ✨ Features



\- 📄 \*\*Multimodal document support\*\* — PDF, DOCX, TXT, PNG, JPG, WEBP

\- 🖼️ \*\*Image understanding\*\* — Gemini Vision API describes images and diagrams

\- 🧩 \*\*Smart chunking\*\* — RecursiveCharacterTextSplitter with metadata

\- 🔢 \*\*Gemini Embeddings\*\* — `gemini-embedding-2` (3072 dimensions)

\- 📦 \*\*Persistent FAISS vector store\*\* — survives server restarts

\- 🔍 \*\*MMR Retrieval\*\* — Maximal Marginal Relevance for diverse results

\- 🔄 \*\*Query Rewriting\*\* — Uses chat history to rewrite vague follow-ups

\- 🧠 \*\*Conversational Memory\*\* — Stores last 20 messages, uses last 8 for inference

\- ⚡ \*\*Groq Streaming\*\* — Ultra-fast LLM responses streamed token by token

\- 📚 \*\*Citations\*\* — Every answer includes chunk-level source references

\- 🐳 \*\*Docker ready\*\* — Full containerized deployment



\---



\## ⚙️ Tech Stack



| Component | Technology |

|---|---|

| Backend | FastAPI + Uvicorn |

| Frontend | Streamlit |

| Vector DB | FAISS (local, persistent) |

| Embeddings | Google Gemini (`gemini-embedding-2`) |

| LLM | Groq (`llama-3.3-70b-versatile`) |

| Vision | Gemini Vision (`gemini-2.0-flash-lite`) |

| Chunking | LangChain RecursiveCharacterTextSplitter |

| Deployment | Docker + Docker Compose |



\---



\## 📁 Project Structure



rag\_project/

│

├── app/

│   ├── main.py                  # FastAPI app entry point

│   ├── routes/

│   │   ├── upload.py            # POST /upload, POST /clear

│   │   ├── query.py             # POST /query, GET /history

│   │

│   ├── services/

│   │   ├── document\_loader.py   # PDF, DOCX, TXT, Image loaders

│   │   ├── chunking.py          # Text splitting with metadata

│   │   ├── embeddings.py        # Gemini embedding calls

│   │   ├── vector\_store.py      # FAISS index management

│   │   ├── retriever.py         # MMR retrieval pipeline

│   │   ├── llm.py               # Groq streaming + query rewriting

│   │   ├── memory.py            # Conversation buffer memory

│   │   ├── citation.py          # Citation generation

│   │

│   ├── utils/

│   │   ├── file\_utils.py        # File saving, validation

│   │   ├── constants.py         # All config constants

│

├── data/

│   ├── uploads/                 # Uploaded documents

│   ├── faiss\_index/             # Persisted FAISS index

│

├── frontend/

│   ├── streamlit\_app.py         # Streamlit chat UI

│

├── Dockerfile                   # Backend container

├── Dockerfile.streamlit         # Frontend container

├── docker-compose.yml           # Multi-container orchestration

├── requirements.txt

├── .env                         # API keys (never commit!)

├── .gitignore

└── README.md



\---



\## 🚀 Quick Start



\### 1. Clone the repository



```bash

git clone https://github.com/yourusername/rag-document-qa.git

cd rag-document-qa

```



\### 2. Set up environment variables



```bash

cp .env.example .env

```



Edit `.env` and add your API keys:



```env

GEMINI\_API\_KEY=your\_gemini\_api\_key\_here

GROQ\_API\_KEY=your\_groq\_api\_key\_here

```



\### 3. Run with Docker (recommended)



```bash

docker compose up --build

```



\- \*\*Frontend:\*\* http://localhost:8501

\- \*\*Backend API:\*\* http://localhost:8000

\- \*\*API Docs:\*\* http://localhost:8000/docs



\### 4. Run locally (without Docker)



```bash

\# Create virtual environment

python -m venv venv

venv\\Scripts\\activate  # Windows

source venv/bin/activate  # Mac/Linux



\# Install dependencies

pip install -r requirements.txt



\# Start backend

uvicorn app.main:app --reload



\# Start frontend (new terminal)

streamlit run frontend/streamlit\_app.py

```



\---



\## 🌐 API Endpoints



| Method | Endpoint | Description |

|---|---|---|

| `GET` | `/` | Health check |

| `GET` | `/stats` | FAISS index statistics |

| `POST` | `/upload` | Upload and index documents |

| `POST` | `/query` | Query documents (streaming) |

| `GET` | `/history` | View conversation memory |

| `DELETE` | `/history` | Clear conversation memory |

| `POST` | `/clear` | Clear FAISS vector index |



\---



\## 🔐 Environment Variables



| Variable | Description |

|---|---|

| `GEMINI\_API\_KEY` | Google Gemini API key |

| `GROQ\_API\_KEY` | Groq API key |



Get your keys from:

\- Gemini: https://aistudio.google.com/

\- Groq: https://console.groq.com/



\---



\## 🧪 Testing the System



1\. Start the system (Docker or local)

2\. Upload a PDF via the sidebar

3\. Ask a question about the document

4\. Try a follow-up question to test memory

5\. Check citations in the response

6\. Restart the server and verify FAISS persists



\---



\## 📝 How It Works



User uploads document

↓

Document parsed (text + images via Gemini Vision)

↓

Text split into chunks (500 chars, 50 overlap)

↓

Chunks embedded via Gemini (3072 dim vectors)

↓

Vectors stored in FAISS index (persisted to disk)

↓

User asks question

↓

Query rewritten using chat history

↓

Query embedded → FAISS similarity search

↓

MMR reranking for diversity (top 4 chunks)

↓

Chunks + history sent to Groq LLM

↓

Streaming response returned with citations

↓

Response + user query stored in memory



\---



\## 🙏 Acknowledgements



\- \[FastAPI](https://fastapi.tiangolo.com/)

\- \[Streamlit](https://streamlit.io/)

\- \[FAISS](https://github.com/facebookresearch/faiss)

\- \[Google Gemini](https://ai.google.dev/)

\- \[Groq](https://groq.com/)

\- \[LangChain](https://langchain.com/)




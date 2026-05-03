# ⚖️ AI Lawyer — Egyptian Labor Law GraphRAG Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge\&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge\&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge\&logo=streamlit)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-008CC1?style=for-the-badge\&logo=neo4j)
![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-GPT--4.1-0078D4?style=for-the-badge\&logo=microsoftazure)
![GraphRAG](https://img.shields.io/badge/RAG-GraphRAG-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-Educational-green?style=for-the-badge)

### 🧠 Intelligent Legal Assistant powered by Hybrid GraphRAG

Arabic legal AI assistant specialized in **Egyptian Labor Law No. 14 of 2025**.
The system combines:

✅ Semantic Search
✅ Graph Retrieval
✅ Explicit Legal Article Routing
✅ Neo4j Graph Traversal
✅ Azure OpenAI Generation

</div>

---

# 📌 Overview

AI Lawyer is an advanced Arabic legal assistant designed to help users:

* Understand Egyptian labor law.
* Retrieve legal articles accurately.
* Ask legal questions in natural Arabic language.
* Navigate relationships between legal materials.
* Explore definitions and cross-references using GraphRAG.

Unlike traditional chatbots, this system does not rely only on embeddings.
It combines:

* **Semantic Retrieval**
* **Explicit Article Retrieval**
* **Graph Traversal**
* **Definition Expansion**
* **Cross-Reference Expansion**

This significantly improves retrieval precision and reduces hallucinations.

---

# 🚀 Key Features

## 🧠 Hybrid GraphRAG Pipeline

The system combines:

| Component         | Purpose                         |
| ----------------- | ------------------------------- |
| Vector Search     | Semantic similarity retrieval   |
| Neo4j Graph       | Relationship-aware retrieval    |
| Explicit Routing  | Direct article lookup           |
| Definitions Graph | Legal terminology expansion     |
| References Graph  | Linked legal material traversal |

---

## 📚 Explicit Legal Article Retrieval

The assistant can detect explicit article references automatically.

### Example

```arabic
اعرض المادة 70 بالكامل
```

The system:

1. Detects article references.
2. Extracts legal article numbers.
3. Routes the query directly to Neo4j.
4. Retrieves legal text without relying only on semantic similarity.

---

## 🔗 Graph Traversal

The project uses Neo4j relationships such as:

```text
(:Article)-[:REFERENCES]->(:Article)
(:Article)-[:USES_TERM]->(:Definition)
(:Article)-[:NEXT]->(:Article)
```

This enables:

* Cross-reference discovery
* Legal relationship navigation
* Definition-aware retrieval
* Connected article exploration

---

## 📖 Legal Definitions Expansion

The assistant automatically expands legal definitions.

### Example

```arabic
ما تعريف العامل؟
```

The system retrieves legal definitions directly from the graph database.

---

## 🎤 Voice Input Support

Users can ask legal questions using voice input.

Supported Features:

* Arabic speech recognition
* Real-time transcription
* Voice-to-query interaction

---

## 🕸️ Interactive Graph Visualization

The UI visualizes:

* Legal articles
* Relationships
* References
* Definitions

Using interactive Neo4j-style graph rendering.

---

## ⚡ Streaming Responses

The assistant streams generated responses gradually for better UX.

---

# 🏗️ System Architecture

```text
User Question
      ↓
Intent Detection
      ↓
Explicit Legal Routing
      ↓
Embedding Generation
      ↓
Neo4j Vector Search
      ↓
Graph Expansion
      ├── REFERENCES
      ├── USES_TERM
      └── NEXT
      ↓
Context Building
      ↓
Azure OpenAI GPT-4.1
      ↓
Final Legal Answer
```

---

# 🧠 Retrieval Pipeline

## 1️⃣ Query Routing

Regex-based routing detects:

* Explicit article requests
* Definitions
* Cross-reference queries

### Examples

```arabic
المادة 48
ما تعريف الأجر؟
ما المواد المرتبطة بالمادة 70؟
```

---

## 2️⃣ Embedding Generation

Azure OpenAI embeddings are generated using:

```text
text-embedding-3-small
```

---

## 3️⃣ Neo4j Vector Search

The system performs vector search across:

* Articles
* Article Segments
* Definitions

---

## 4️⃣ Graph Expansion

The graph traversal stage expands retrieval using:

### REFERENCES

Retrieve legally connected articles.

### USES_TERM

Retrieve legal definitions related to retrieved articles.

### NEXT

Navigate sequential legal content.

---

## 5️⃣ Context Construction

The system prioritizes retrieved chunks in the following order:

```text
Explicit Retrieval
↓
Primary Semantic Retrieval
↓
Definitions Expansion
↓
Cross References
```

---

## 6️⃣ Answer Generation

The final response is generated using:

```text
Azure OpenAI GPT-4.1
```

with deterministic legal prompting.

---

# 🖥️ Tech Stack

| Layer              | Technology             |
| ------------------ | ---------------------- |
| Backend API        | FastAPI                |
| Frontend           | Streamlit              |
| Graph Database     | Neo4j                  |
| LLM                | Azure OpenAI GPT-4.1   |
| Embeddings         | text-embedding-3-small |
| Visualization      | streamlit-agraph       |
| Speech Recognition | streamlit-mic-recorder |
| Language           | Python                 |

---

# 📂 Project Structure

```text
ai-lawyer/
│
├── api/
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   └── core/
│
├── app/
│   ├── rag.py
│   └── streamlit_app.py
│
├── data/
│   ├── chunks.jsonl
│   └── chunks_embedded.jsonl
│
├── src/
│   ├── embed_azure.py
│   ├── load_neo4j.py
│   ├── normalize.py
│   ├── crossrefs.py
│   └── chunk.py
│
├── .env
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ai-lawyer.git
cd ai-lawyer
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv new_env
```

### Windows

```bash
new_env\Scripts\activate
```

### Linux / Mac

```bash
source new_env/bin/activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

---

# ▶️ Running the Project

## Start FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

## Start Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

# 📸 Example Questions

## Legal Definitions

```arabic
ما تعريف العامل؟
```

```arabic
ما تعريف الأجر؟
```

---

## Explicit Article Retrieval

```arabic
اعرض المادة 70 بالكامل
```

```arabic
ما عقوبة مخالفة المادة 120؟
```

---

## Legal Rights

```arabic
ما مدة الإجازة السنوية للعامل؟
```

```arabic
هل يجوز فصل العامل؟
```

---

## Graph Relationships

```arabic
ما المواد المرتبطة بالمادة 70؟
```

---

# 🧪 API Example

## POST /ask

```json
{
  "question": "ما تعريف العامل؟"
}
```

---

## Example Response

```json
{
  "question": "ما تعريف العامل؟",
  "answer": "...",
  "sources": [...],
  "elapsed_ms": {
    "route": 1,
    "embed": 540,
    "retrieve": 820,
    "generate": 1200
  }
}
```

---

# 🩺 Health Monitoring

The backend exposes a dedicated health endpoint:

```text
GET /health
```

Example Response:

```json
{
  "api": "running",
  "status": "healthy"
}
```

This endpoint is useful for:

* Deployment monitoring
* Docker/Kubernetes health checks
* API uptime validation
* CI/CD pipelines
* Production observability

---

# 📝 Logging System

The project includes a structured logging system for backend observability.

Features:

* Request logging
* Error tracking
* Retrieval monitoring
* Debugging support
* Performance tracing

Example Logs:

```text
2026-05-03 17:49:50 - INFO - Question received
2026-05-03 17:49:52 - INFO - RAG response generated successfully
2026-05-03 17:49:52 - ERROR - Error occurred while processing question
```

The logging layer helps monitor:

* Incoming legal queries
* Retrieval pipeline execution
* Azure OpenAI requests
* Neo4j operations
* System failures

---

# 📊 Explainable AI Features

The system exposes:

* Retrieval timing metrics
* Explicit article detection
* Source transparency
* Similarity scores
* Graph relationship visualization

This improves explainability and trustworthiness.

---

# 🔐 Hallucination Reduction

The assistant reduces hallucinations using:

* Explicit legal retrieval
* Deterministic prompting
* Graph-based context expansion
* Source-grounded generation

---

# 🌍 Why This Project Matters

Legal information is often:

* Complex
* Hard to navigate
* Difficult for non-specialists
* Full of interconnected references

This project demonstrates how GraphRAG can improve:

✅ Legal accessibility
✅ Retrieval precision
✅ Explainability
✅ Context-aware reasoning
✅ Arabic legal AI systems

---

# 🎓 Educational Value

This project demonstrates practical usage of:

* Retrieval-Augmented Generation (RAG)
* GraphRAG
* Neo4j Graph Databases
* Azure OpenAI
* FastAPI APIs
* Streamlit Frontends
* Arabic NLP
* Semantic Search
* Hybrid Retrieval Systems

---

# 👨‍💻 Contributors

Built as a Graduation Project focused on:

* Legal AI
* GraphRAG Systems
* Arabic NLP
* Explainable AI

---

# 📄 Disclaimer

This project is for educational and experimental purposes only.
It is not a replacement for professional legal consultation.

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
🧠 Contribute improvements

---

<div align="center">

## ⚖️ AI Lawyer — GraphRAG for Egyptian Labor Law

Built with ❤️ using FastAPI, Neo4j, Azure OpenAI, and Streamlit.

</div>

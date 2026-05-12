# PDF Knowledge Base API

A high-performance RAG (Retrieval-Augmented Generation) API for ingesting, indexing, and querying PDF documents. Built with FastAPI, SQLModel (PostgreSQL), ChromaDB, and Arq (Redis).

## 🚀 Features

- **Asynchronous Ingestion Pipeline**: Background workers handle PDF parsing (PyMuPDF), semantic chunking, and vector embedding.
- **Semantic Search**: High-performance vector retrieval powered by ChromaDB.
- **Multi-Provider LLM Integration**: Flexible RAG queries via LiteLLM (Gemini, OpenAI, Anthropic, etc.).
- **Layered Architecture**: Clean separation between API endpoints, business services, and data repositories.
- **Crash Recovery**: Automatic detection and recovery of "stuck" processing tasks on application startup.
- **Schema Management**: Robust database migrations using Alembic.
- **Streaming Support**: Real-time answer generation using Server-Sent Events (SSE).
- **Fully Dockerized**: Pre-configured environment for development and production.

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Async/Await, Type Hints)
- **Database**: [PostgreSQL](https://www.postgresql.org/) with [SQLModel](https://sqlmodel.tiangolo.com/) (SQLAlchemy 2.0 based)
- **Migrations**: [Alembic](https://alembic.sqlalchemy.org/)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
- **Background Tasks**: [Arq](https://github.com/samuelcolvin/arq) (Redis-based)
- **PDF Parsing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- **LLM Integration**: [LiteLLM](https://docs.litellm.ai/)
- **Linting/Formatting**: [Ruff](https://beta.ruff.rs/)

## 🏗️ Project Structure

```text
├── migrations/          # Alembic database migrations
├── app/
│   ├── api/             # API Layer: Routers and Annotated dependencies
│   ├── core/            # Infrastructure Layer: Config, DB Engine, Lifespan, Queue
│   ├── integrations/    # External Adapters: LLM (LiteLLM) and VectorStore (Chroma)
│   ├── lib/             # Utility Layer: Document parsers and chunkers
│   ├── models/          # Domain Entities: SQLModel database models
│   ├── repositories/    # Data Access Layer: Repository pattern implementations
│   ├── schemas/         # Transfer Layer: Pydantic request/response models
│   ├── services/        # Logic Layer: Business orchestration and RAG pipeline
│   ├── workers/         # Background Layer: Arq worker tasks and settings
│   └── main.py          # App entry point & lifecycle management
├── tests/               # Pytest suite with isolated DB/Redis setups
├── alembic.ini          # Migration configuration
├── docker-compose.yml   # Multi-service orchestration (API, Worker, DB, Redis, Chroma)
└── Dockerfile           # API and Worker container definition
```

## 🏁 Getting Started

### 1. Prerequisites
- Docker and Docker Compose
- LLM API Key (e.g., Google Gemini, OpenAI)

### 2. Quick Start (Docker)
1. **Clone and Configure**:
   ```bash
   git clone https://github.com/ayo6706/pdf-rag-system.git
   cd pdf-rag-system
   cp .env.example .env
   ```
   Edit `.env` and add your `GOOGLE_API_KEY` (or other provider keys).

2. **Launch Services**:
   ```bash
   docker-compose up --build
   ```
   This starts the API (port 8000), Worker, PostgreSQL, ChromaDB, and Redis. Migrations are applied automatically on startup.

### 3. Local Development (Manual)
1. **Setup Venv & Dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Migrations**:
   ```bash
   alembic upgrade head
   ```

3. **Run API**:
   ```bash
   fastapi dev app/main.py
   ```

4. **Run Worker**:
   ```bash
   arq app.workers.ingestion.WorkerSettings
   ```

## 📖 API Documentation

Once running, access the interactive docs:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/documents/upload` | Upload a PDF for processing |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `POST` | `/api/v1/query` | Ask a question (supports streaming) |
| `GET` | `/api/v1/health` | Service and infrastructure health check |

## 🧪 Testing

The project uses `pytest` for integration and unit testing.
```bash
pytest
```

---
*Developed with a focus on clean architecture, high-performance async patterns, and production-grade observability.*

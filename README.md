# RAG Evaluation Bot

A Retrieval-Augmented Generation (RAG) application that answers questions from PDF documents using semantic search, ChromaDB, and Google's Gemini API. The project includes a custom chat interface and an evaluation harness for measuring retrieval quality and comparing chunking and reranking strategies.

## Features

- **PDF-based Question Answering:** Ask questions about AI, machine learning, large language models, RAG, and statistics.
- **Document Processing:** Extract text from PDFs and split documents into chunks.
- **Semantic Retrieval:** Generate embeddings with `all-MiniLM-L6-v2` and retrieve relevant passages using ChromaDB.
- **LLM Answer Generation:** Generate answers using the Gemini API and retrieved document context.
- **Source References:** Return source document names and page information.
- **Interactive Chat UI:** Use a custom HTML, CSS, and JavaScript interface.
- **REST API:** Expose chatbot functionality through FastAPI and OpenAPI documentation.
- **Retrieval Evaluation:** Evaluate retrieval using Hit Rate@5 and Mean Reciprocal Rank (MRR).
- **Chunking Experiments:** Compare chunk sizes and overlap configurations.
- **Reranking Experiments:** Compare vector-search ranking with CrossEncoder reranking.
- **Evaluation Diagnostics:** Inspect candidate rankings and investigate retrieval failures.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| LLM | Google Gemini API |
| Embedding model | Sentence Transformers — `all-MiniLM-L6-v2` |
| Reranking model | CrossEncoder — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector database | ChromaDB |
| PDF processing | PyPDF |
| Frontend | HTML, CSS, JavaScript |
| Evaluation | Python, JSON, CSV |
| API documentation | Swagger UI / OpenAPI |

## Architecture

1. Load PDF documents from `data/documents/`.
2. Extract document text and divide it into chunks.
3. Generate embeddings for the chunks.
4. Store vectors and metadata in ChromaDB.
5. Embed the user's question and retrieve relevant chunks.
6. Pass retrieved context to Gemini to generate an answer.
7. Return the answer and source references through FastAPI.
8. Run offline evaluation experiments to measure retrieval performance and compare alternative configurations.

## Project Structure

```text
rag-evaluation-bot/
├── data/
│   └── documents/
│       ├── AI & ML.pdf
│       ├── LLM.pdf
│       ├── RAG.pdf
│       └── Statistics.pdf
├── evaluation/
│   ├── compare_chunking.py
│   ├── compare_query_rewriting.py
│   ├── compare_reranking.py
│   ├── diagnose_reranking.py
│   ├── evaluate_answers.py
│   ├── evaluate_labeled_reranking.py
│   ├── evaluate_mrr.py
│   ├── evaluate_retrieval.py
│   ├── relevance_labels.json
│   ├── test_answers.json
│   ├── test_questions.json
│   └── chunking_results.csv
├── src/
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
├── tests/
├── app.py
├── index.html
├── script.js
├── style.css
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*Note: `requirements.txt` and `.env.example` are deployment files to be added. The structure above is the intended repository structure.*

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag-evaluation-bot
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Obtain an API key through Google AI Studio. Never commit `.env` or expose your API key in public source code.

### 5. Start the backend

```bash
uvicorn app:app --reload
```

Open the API documentation at:

`http://127.0.0.1:8000/docs`

### 6. Open the frontend

Open `index.html` using VS Code Live Server or another local static server.

Ensure the frontend API URL points to the running backend. For deployment, configure it to use the deployed backend URL.

## Running the Evaluation

Run evaluation commands from the project root with the virtual environment activated.

### Retrieval evaluation

```bash
python -m evaluation.evaluate_retrieval
```

Evaluates 25 benchmark questions against their expected source documents and, where provided, expected page numbers.

### MRR evaluation

```bash
python -m evaluation.evaluate_mrr
```

Evaluates the ranking of expected source documents using Mean Reciprocal Rank.

### Chunking comparison

```bash
python -m evaluation.compare_chunking
```

Compares different chunk sizes and overlap configurations.

### Reranking comparison

```bash
python -m evaluation.compare_reranking
```

Compares baseline vector retrieval against CrossEncoder reranking using a candidate depth of 20 and final top-k of 5.

### Reranking diagnostics

```bash
python -m evaluation.diagnose_reranking
```

Investigates cases where the baseline retrieves an expected source/page label but reranking removes it from the final top five.

### Answer evaluation

```bash
python -m evaluation.evaluate_answers
```

Runs the existing answer-evaluation workflow. This requires the Gemini API key and may incur API usage depending on the configured model and account limits.

## Evaluation Results

### 1. Retrieval benchmark

The retrieval benchmark contains 25 questions across four PDF documents.

| Metric | Result |
|---|---:|
| Source Hit Rate@5 | 100% (25/25) |
| Expected Page Hit Rate@5 | 100% (14/14) |

These results mean that the expected source document appeared in the top five results for all 25 questions, and the expected page appeared in the top five for all 14 questions with page labels.

### 2. Chunking experiment

Three chunk configurations were compared using the same 25 questions.

| Configuration | Chunk size | Overlap | Chunks | Source Hit Rate@5 | Source MRR@5 |
|---|---:|---:|---:|---:|---:|
| A | 500 | 0 | 374 | 100% | 0.838 |
| B | 250 | 50 | 834 | 96% | 0.845 |
| C | 800 | 100 | 253 | 100% | 0.887 |

The larger-chunk configuration had the highest source MRR in this experiment. However, the benchmark is small and uses source labels as relevance proxies, so these results are preliminary rather than conclusive.

### 3. Reranking experiment

The CrossEncoder reranker scored the top 20 vector-search candidates and selected the final five results.

| Metric | Baseline | Reranked |
|---|---:|---:|
| Source Hit Rate@5 | 100% | 100% |
| Source MRR@5 | 0.921 | 0.920 |
| Expected Page Hit Rate@5 | 100% (14/14) | 92.9% (13/14) |
| Expected Page MRR@5 | 0.881 | 0.768 |

**Finding:** Reranking did not improve the measured results on this benchmark. Baseline retrieval remains the preferred default for the current implementation.

A diagnostic experiment also identified a question where the expected page was present among the top 20 candidates but fell to rank 10 after reranking.

### 4. Initial answer-quality evaluation

An earlier five-question evaluation produced the following results:

| Metric | Initial result |
|---|---:|
| Retrieval Hit Rate@5 | 100% (5/5) |
| MRR | 0.767 |
| Average LLM-judge answer quality | 5.00/5 |
| Average chatbot request latency | 14.01 seconds |
| Successful answer evaluations | 5/5 |

These are preliminary results from a separate five-question evaluation, not the 25-question retrieval benchmark. The answer-quality score was generated by an LLM judge and should not be interpreted as proof of perfect answers.

## Evaluation Methodology and Limitations

- **Hit Rate@5:** Measures whether an expected source or page label appears among the top five retrieved chunks.
- **MRR:** Rewards higher rankings for results matching the expected labels.
- **Chunking comparison:** Measures how chunk size and overlap affect retrieval results.
- **Reranking comparison:** Evaluates whether a CrossEncoder improves the ordering of retrieved candidates.
- **Answer evaluation:** Uses the existing LLM-judge workflow to assess generated answers and record latency.

The benchmark currently uses expected source and page labels as relevance proxies. These metrics do not establish that every retrieved passage directly supports the answer. The benchmark is relatively small, and further human-verified passage relevance labels and broader answer-quality tests would strengthen the evaluation.

## Deployment

The application is being prepared for deployment using Render.

The deployment configuration must provide:

- A Python runtime and the required dependencies.
- The `GEMINI_API_KEY` environment variable, configured through the hosting provider.
- A working frontend-to-backend API URL.
- Access to the required PDF documents and a reproducible vector-index initialization process.

**Important:** The local ChromaDB directory is excluded from Git. Deployment must rebuild or otherwise initialize the vector index from the available documents. Free hosting instances may have limited memory, startup time, and ephemeral storage, so deployment behavior must be verified after configuration.

## Future Improvements

- Add human-verified passage-level relevance labels.
- Expand the benchmark with more questions and unanswerable queries.
- Evaluate answer faithfulness and citation correctness independently.
- Test retrieval and reranking with a stronger, validated relevance benchmark.
- Improve cold-start performance and persistent vector-index handling.
- Monitor latency and failures in the deployed application.

## Security

- Store API keys in environment variables.
- Keep `.env`, virtual environments, and local ChromaDB files out of Git.
- Do not expose API keys in frontend JavaScript.
- Never publish documents unless you have permission to share them.

## License

Add a license if you intend to distribute the project publicly. Verify that you have permission to redistribute the source PDFs before publishing them.

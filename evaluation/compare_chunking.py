
import csv
import chromadb
from pathlib import Path

from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents
from src.embeddings import create_embeddings


QUESTIONS = [
    {"question": "What is the Central Limit Theorem?",
     "source": "Statistics.pdf"},
    {"question": "What is Retrieval-Augmented Generation?",
     "source": "RAG.pdf"},
    {"question": "What is an embedding in machine learning?",
     "source": "AI & ML.pdf"},
    {"question": "What is a large language model?",
     "source": "LLM.pdf"},
    {"question": "What is the purpose of chunking in RAG?",
     "source": "RAG.pdf"},
    {"question": "What are the trade-offs of using small and large chunks?",
     "source": "RAG.pdf", "page": 6},
    {"question": "How does chunk overlap help retrieval, and what costs does it introduce?",
     "source": "RAG.pdf", "page": 6},
    {"question": "Why is embedding similarity not proof of factual support?",
     "source": "RAG.pdf", "page": 7},
    {"question": "Why can increasing the top-k value hurt RAG answer quality?",
     "source": "RAG.pdf", "page": 10},
    {"question": "What is the difference between answer correctness and faithfulness?",
     "source": "RAG.pdf", "page": 13},
    {"question": "What is the difference between sparse and dense retrieval?",
     "source": "RAG.pdf", "page": 8},
    {"question": "How does BM25 rank documents?",
     "source": "RAG.pdf", "page": 8},
    {"question": "What is the purpose of reranking in a RAG pipeline?",
     "source": "RAG.pdf", "page": 10},
    {"question": "What is the difference between Hit Rate@k and Recall@k?",
     "source": "RAG.pdf", "page": 12},
    {"question": "What does Mean Reciprocal Rank measure?",
     "source": "RAG.pdf", "page": 12},
    {"question": "Why should retrieval quality and answer quality be evaluated separately?",
     "source": "RAG.pdf", "page": 13},
    {"question": "Why should RAG evaluation include unanswerable questions?",
     "source": "RAG.pdf", "page": 14},
    {"question": "What is the difference between population and sample?",
     "source": "Statistics.pdf"},
    {"question": "What is the difference between variance and standard deviation?",
     "source": "Statistics.pdf"},
    {"question": "What is the difference between covariance and correlation?",
     "source": "Statistics.pdf"},
    {"question": "What is the role of embeddings in machine learning?",
     "source": "AI & ML.pdf"},
    {"question": "How does a large language model generate text?",
     "source": "LLM.pdf"},
    {"question": "What is the purpose of tokenization in language models?",
     "source": "LLM.pdf"},
    {"question": "What is the role of a vector store in RAG?",
     "source": "RAG.pdf", "page": 9},
    {"question": "Why is chunking strategy important for retrieval quality?",
     "source": "RAG.pdf", "page": 6},
]

CONFIGURATIONS = [
    {"name": "A: 500/0", "size": 500, "overlap": 0},
    {"name": "B: 250/50", "size": 250, "overlap": 50},
    {"name": "C: 800/100", "size": 800, "overlap": 100},
]

TOP_K = 5


def main():
    documents = load_pdfs()
    client = chromadb.EphemeralClient()
    experiment_results = []

    labelled_questions = [
        item for item in QUESTIONS if "page" in item
    ]

    print(f"Loaded {len(documents)} pages.")
    print(f"Benchmark questions: {len(QUESTIONS)}")
    print(f"Page-labelled questions: {len(labelled_questions)}")
    print("Using temporary ChromaDB collections.")
    print("The existing chatbot database will not be modified.")

    for config in CONFIGURATIONS:
        chunks = chunk_documents(
            documents,
            chunk_size=config["size"],
            chunk_overlap=config["overlap"],
        )

        collection_name = (
            f"experiment_{config['size']}_{config['overlap']}"
        )

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = create_embeddings(texts)

        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                }
                for chunk in chunks
            ],
        )

        source_hits = 0
        source_reciprocal_ranks = []
        page_hits = 0
        page_reciprocal_ranks = []

        for item in QUESTIONS:
            query_embedding = create_embeddings(
                [item["question"]]
            )[0]

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(TOP_K, len(chunks)),
                include=["metadatas"],
            )

            metadatas = results["metadatas"][0]

            source_ranks = [
                rank
                for rank, metadata in enumerate(metadatas, start=1)
                if metadata["source"] == item["source"]
            ]

            if source_ranks:
                source_hits += 1
                source_reciprocal_ranks.append(1 / source_ranks[0])
            else:
                source_reciprocal_ranks.append(0)

            if "page" in item:
                page_ranks = [
                    rank
                    for rank, metadata in enumerate(metadatas, start=1)
                    if metadata["source"] == item["source"]
                    and metadata["page"] == item["page"]
                ]

                if page_ranks:
                    page_hits += 1
                    page_reciprocal_ranks.append(1 / page_ranks[0])
                else:
                    page_reciprocal_ranks.append(0)

        source_hit_rate = source_hits / len(QUESTIONS)
        source_mrr = (
            sum(source_reciprocal_ranks) / len(QUESTIONS)
        )

        page_hit_rate = (
            page_hits / len(labelled_questions)
            if labelled_questions else 0
        )
        page_mrr = (
            sum(page_reciprocal_ranks) / len(labelled_questions)
            if labelled_questions else 0
        )

        print("\n" + "=" * 45)
        print(f"Configuration: {config['name']}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Source Hit Rate@{TOP_K}: {source_hit_rate:.2%}")
        print(f"Source MRR@{TOP_K}: {source_mrr:.3f}")
        print(
            f"Page Hit Rate@{TOP_K}: "
            f"{page_hit_rate:.2%} ({page_hits}/{len(labelled_questions)})"
        )
        print(f"Page-based MRR@{TOP_K}: {page_mrr:.3f}")

        experiment_results.append({
            "configuration": config["name"],
            "chunk_size": config["size"],
            "chunk_overlap": config["overlap"],
            "total_chunks": len(chunks),
            "top_k": TOP_K,
            "question_count": len(QUESTIONS),
            "page_labelled_question_count": len(labelled_questions),
            "source_hit_rate": round(source_hit_rate, 4),
            "source_mrr": round(source_mrr, 4),
            "page_hit_rate": round(page_hit_rate, 4),
            "page_mrr": round(page_mrr, 4),
        })

        client.delete_collection(collection_name)

    output_path = Path("evaluation/chunking_results.csv")

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=experiment_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(experiment_results)

    print(f"\nResults saved to: {output_path}")
    print("Experiment complete.")


if __name__ == "__main__":
    main()

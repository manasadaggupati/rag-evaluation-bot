from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = []

    for document in documents:
        for chunk in splitter.create_documents(
            [document["text"]],
            metadatas=[{
                "source": document["source"],
                "page": document["page"],
            }],
        ):
            chunks.append({
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "start_index": chunk.metadata["start_index"],
            })

    return chunks
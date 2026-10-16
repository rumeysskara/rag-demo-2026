"""Dokümanları okuyup v1 ve v2/v3 için Chroma koleksiyonlarını oluşturur. Demodan önce BİR KEZ çalıştırın."""

import config
from pipelines.common import chunk_by_section, chunk_fixed, embed_documents, get_client, read_documents


def rebuild(client, name, ids, texts, metadatas, titles):
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    collection = client.create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    collection.add(ids=ids, documents=texts, metadatas=metadatas,
                   embeddings=embed_documents(texts, titles))
    print(f"  {name}: {len(ids)} parça")


def main():
    documents = read_documents()
    client = get_client()
    print(f"Embedding modeli: {config.EMBED_MODEL}")

    # v1: sabit boyutlu parçalar
    ids, texts, metas = [], [], []
    for source, text in documents.items():
        for i, chunk in enumerate(chunk_fixed(text)):
            ids.append(f"{source}#{i}")
            texts.append(chunk)
            metas.append({"source": source, "section": f"parça {i + 1}"})
    rebuild(client, config.V1_COLLECTION, ids, texts, metas, titles=None)

    # v2/v3: madde bazlı parçalar + metadata + bağlam satırı
    chunks = [c for source, text in documents.items() for c in chunk_by_section(source, text)]
    rebuild(
        client, config.V2_COLLECTION,
        ids=[f"{c['source']}#{i}" for i, c in enumerate(chunks)],
        texts=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"], "updated_at": c["updated_at"]}
                   for c in chunks],
        titles=[c["title"] for c in chunks],
    )
    print(f"İndeks hazır: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()

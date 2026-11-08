from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# Le modèle qui donne une "adresse" à chaque texte
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Ramasser les documents, les ranger, préparer la recherche
documents = SimpleDirectoryReader("data/procedures").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=2)

def chercher(question: str) -> str:
    """Renvoie les passages les plus proches, avec leur source et leur score."""
    resultats = retriever.retrieve(question)
    return "\n\n".join(
        f"[Source : {r.metadata.get('file_name')} | score {r.score:.2f}]\n{r.text}"
        for r in resultats
    )

if __name__ == "__main__":
    print(chercher("timeout sur la base Oracle, les virements SEPA échouent"))
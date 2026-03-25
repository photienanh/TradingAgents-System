import chromadb
from chromadb.config import Settings
from openai import OpenAI


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.embedding = "text-embedding-3-small"
        self.client = OpenAI(base_url=config["backend_url"])
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        
        response = self.client.embeddings.create(
            model=self.embedding, input=text
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []

        if not documents or not documents[0]:
            return []

        first_documents = documents[0]
        first_metadatas = metadatas[0] if metadatas else []
        first_distances = distances[0] if distances else []

        matched_results = []
        for i in range(len(first_documents)):
            metadata = first_metadatas[i] if i < len(first_metadatas) else {}
            distance = first_distances[i] if i < len(first_distances) else 1.0
            matched_results.append(
                {
                    "matched_situation": first_documents[i],
                    "recommendation": metadata.get("recommendation", ""),
                    "similarity_score": 1 - distance,
                }
            )

        return matched_results

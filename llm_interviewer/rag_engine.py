import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeRAG:
    def __init__(self):
        # We use TF-IDF instead of deep embeddings to avoid PyTorch/CUDA memory conflicts 
        # with Whisper, ensuring 0 VRAM usage and absolute stability.
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks = []
        self.chunk_vectors = None

    def build_index(self, resume_parsed: dict):
        """
        Takes the parsed resume JSON and builds a TF-IDF vector index.
        """
        self.chunks = []
        
        # Extract text chunks from the parsed resume
        if "experience" in resume_parsed:
            for exp in resume_parsed.get("experience", []):
                self.chunks.append(f"Experience at {exp.get('company', '')} as {exp.get('role', '')}: {exp.get('description', '')}")
                
        if "projects" in resume_parsed:
            for proj in resume_parsed.get("projects", []):
                self.chunks.append(f"Project {proj.get('name', '')}: {proj.get('description', '')}")
                
        if "skills" in resume_parsed:
            skills = ", ".join(resume_parsed.get("skills", {}).get("all", []))
            if skills:
                self.chunks.append(f"Technical Skills: {skills}")
                
        if "education" in resume_parsed:
            for edu in resume_parsed.get("education", []):
                self.chunks.append(f"Education: {edu.get('degree', '')} at {edu.get('institution', '')}")

        if not self.chunks:
            self.chunks.append("Resume data is minimal.")

        # Build TF-IDF Vectors
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        print(f"[RAG] Built TF-IDF index with {len(self.chunks)} chunks from Resume.")

    def get_relevant_context(self, query: str, top_k: int = 2) -> str:
        """
        Retrieves the top_k most relevant resume chunks for the given query using Cosine Similarity.
        """
        if self.chunk_vectors is None or not self.chunks:
            return ""
            
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # Get indices of top_k highest similarities
        # If all similarities are 0 (no keyword match), just return empty
        if np.all(similarities == 0):
            return ""
            
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_texts = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # slight threshold to ensure relevance
                relevant_texts.append(self.chunks[idx])
                
        return "\n".join(relevant_texts)

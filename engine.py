import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

class SimpleVectorStore:
    def __init__(self):
        self.vectors = None
        self.documents = []

    def add(self, embeddings, documents):
        # We normalize vectors so we can use Dot Product for Cosine Similarity
        self.vectors = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.documents = documents

    def search(self, query_emb):
        query_norm = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(self.vectors, query_norm)
        top_idx = np.argmax(similarities)
        return self.documents[top_idx], float(similarities[top_idx])

class SemanticEngine:
    def __init__(self, n_clusters=12):
        # all-MiniLM-L6-v2 is lightweight and perfect for a "fast" service
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = SimpleVectorStore()
        # GMM handles the "Fuzzy" part: it gives probabilities, not just one label
        self.clusterer = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42)
        self.corpus = []

    def clean_text(self, text):
        # Remove Usenet headers/quotes to focus on the semantic core of the message
        text = re.sub(r'(From|Subject|Reply-To|Lines|Organization|Expires|Distribution):.*', '', text)
        text = re.sub(r'(\n|>|---).*', ' ', text)
        return " ".join(text.split())

    def initialize(self):
        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        self.corpus = [self.clean_text(t) for t in data.data if len(t) > 100]
        embeddings = self.encoder.encode(self.corpus, batch_size=64, show_progress_bar=True)
        self.vector_store.add(embeddings, self.corpus)
        self.clusterer.fit(embeddings)

    def get_cluster_data(self, embedding):
        probs = self.clusterer.predict_proba(embedding.reshape(1, -1))[0]
        return int(np.argmax(probs)), probs
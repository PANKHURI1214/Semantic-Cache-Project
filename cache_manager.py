import numpy as np

class CacheManager:
    def __init__(self, threshold=0.60):
        self.threshold = threshold
        self.cache_store = []
        self.hits = 0
        self.misses = 0

    def check_cache(self, query_emb, query_cluster):
        if not self.cache_store:
            self.misses += 1
            return None

       
        relevant_entries = [e for e in self.cache_store if e['cluster_id'] == query_cluster]
        
        if not relevant_entries:
            self.misses += 1
            return None

        best_score = -1
        best_match = None
        query_norm = query_emb / np.linalg.norm(query_emb)

        for entry in relevant_entries:
            entry_norm = entry['emb'] / np.linalg.norm(entry['emb'])
            score = np.dot(query_norm, entry_norm)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.threshold:
            self.hits += 1
            return {
                "matched_query": best_match['text'],
                "similarity_score": round(float(best_score), 3),
                "result": best_match['result'],
                "dominant_cluster": best_match['cluster_id']
            }
        
        self.misses += 1
        return None

    def update(self, query_text, query_emb, result, cluster_id):
        self.cache_store.append({
            "text": query_text,
            "emb": query_emb,
            "result": result,
            "cluster_id": cluster_id
        })

    def get_stats(self):
        total = self.hits + self.misses
        return {
            "total_entries": len(self.cache_store),
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0
        }

    def flush(self):
        self.cache_store = []
        self.hits = 0

        self.misses = 0

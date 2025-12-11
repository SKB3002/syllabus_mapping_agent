"""
curriculum_comparator.py  (FINAL VERSION – STABLE & MEMORY SAFE)

✔ Auto-detects CSV / XLSX
✔ Dynamic board names in final output
✔ Handles ANY number of columns (even different between boards)
✔ Cleans & preprocesses everything
✔ Embedding Similarity
✔ Topic Similarity
✔ Concept Extraction (KeyBERT)
✔ Clustering (FAISS → sklearn fallback)
✔ Agreement Engine (4 votes)
✔ Match Percentage Score (0–100%)
✔ TopicName included if available
✔ Prevents MemoryError by forcing string-safe export values
"""

import os
import logging
from typing import Optional, Dict
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("CurriculumComparator")


class CurriculumComparator:

    def __init__(
        self,
        path_boardA: str,
        path_boardB: str,
        boardA_name: str = "Board A",
        boardB_name: str = "Board B",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        num_topics: int = 12,
        concept_top_n: int = 8,
        pca_components: int = 10,
        random_state: int = 42,
        thresholds: Optional[Dict[str, float]] = None,
    ):

        self.pathA = path_boardA
        self.pathB = path_boardB

        # User-defined names
        self.boardA_name = boardA_name
        self.boardB_name = boardB_name

        # Model configuration
        self.embedding_model_name = embedding_model_name
        self.num_topics = num_topics
        self.concept_top_n = concept_top_n
        self.pca_components = pca_components
        self.random_state = random_state

        # Agreement thresholds
        if thresholds is None:
            thresholds = {"emb": 0.50, "topic": 0.40, "concept": 0.30}
        self.thresholds = thresholds

        # Data holders
        self.boardA = pd.DataFrame()
        self.boardB = pd.DataFrame()

        self.emb_model = None
        self.kw_model = None

        self.embA = None
        self.embB = None

        self.emb_sim_matrix = None
        self.topic_sim_matrix = None
        self.concept_sim_matrix = None

        self.clusterA = None
        self.clusterB = None
        self.cluster_match_matrix = None

        self.results_df = None

    # -----------------------------------------------------------
    # CSV/XLSX Auto Loader
    # -----------------------------------------------------------
    def load_any(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def load_data(self):
        if not os.path.exists(self.pathA):
            raise FileNotFoundError(self.pathA)
        if not os.path.exists(self.pathB):
            raise FileNotFoundError(self.pathB)

        logger.info("Loading Board A and Board B files...")
        self.boardA = self.load_any(self.pathA)
        self.boardB = self.load_any(self.pathB)

        # Standardize column types
        self.boardA = self._ensure_str(self.boardA)
        self.boardB = self._ensure_str(self.boardB)

        logger.info(f"Board A loaded with shape: {self.boardA.shape}")
        logger.info(f"Board B loaded with shape: {self.boardB.shape}")

    def _ensure_str(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in df.columns:
            df[c] = df[c].astype(str).fillna("")
        return df

    # -----------------------------------------------------------
    # Preprocessing — only formats known columns but optional
    # -----------------------------------------------------------
    def preprocess(self):
        for df in (self.boardA, self.boardB):
            if "CourseName" in df.columns:
                df["CourseName_clean"] = df["CourseName"].str.lower().str.strip()

            if "ChapterName" in df.columns:
                df["ChapterName_clean"] = df["ChapterName"].str.lower().str.strip()

            if "GradeID" in df.columns:
                df["GradeID"] = df["GradeID"].astype(str).str.strip()

    # -----------------------------------------------------------
    # Build Combined Text using ALL columns (dynamic)
    # -----------------------------------------------------------
    def build_combined_text(self):

        def combine(row):
            cols = sorted(row.index)
            parts = []
            for c in cols:
                val = str(row[c]).strip()
                if val != "":
                    parts.append(val)
            return " . ".join(parts)

        self.boardA["combined"] = self.boardA.apply(combine, axis=1)
        self.boardB["combined"] = self.boardB.apply(combine, axis=1)

    # -----------------------------------------------------------
    # Sentence Embeddings
    # -----------------------------------------------------------
    def _load_embedding_model(self):
        if self.emb_model:
            return
        from sentence_transformers import SentenceTransformer
        self.emb_model = SentenceTransformer(self.embedding_model_name)

    def generate_embeddings(self):
        self._load_embedding_model()
        self.embA = self.emb_model.encode(
            self.boardA["combined"].tolist(), normalize_embeddings=True
        )
        self.embB = self.emb_model.encode(
            self.boardB["combined"].tolist(), normalize_embeddings=True
        )

    def compute_embedding_similarity(self):
        self.emb_sim_matrix = np.matmul(self.embA, self.embB.T)

    # -----------------------------------------------------------
    # Topic Modeling
    # -----------------------------------------------------------
    def compute_topic_similarity(self):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.metrics.pairwise import cosine_similarity

        texts = list(self.boardA["combined"]) + list(self.boardB["combined"])
        bow = CountVectorizer(stop_words="english").fit_transform(texts)

        lda = LatentDirichletAllocation(
            n_components=self.num_topics, random_state=self.random_state
        )
        topics = lda.fit_transform(bow)

        A = topics[: len(self.boardA)]
        B = topics[len(self.boardA):]

        self.topic_sim_matrix = cosine_similarity(A, B)

    # -----------------------------------------------------------
    # Concept Extraction + Jaccard
    # -----------------------------------------------------------
    def _load_kw(self):
        if self.kw_model:
            return
        from keybert import KeyBERT
        self.kw_model = KeyBERT(model=self.embedding_model_name)

    def extract_concepts(self):
        self._load_kw()

        def extract(text):
            if not text.strip():
                return []
            kws = self.kw_model.extract_keywords(
                text,
                top_n=self.concept_top_n,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
            )
            return [k for k, _ in kws]

        self.boardA["concepts"] = self.boardA["combined"].apply(extract)
        self.boardB["concepts"] = self.boardB["combined"].apply(extract)

    @staticmethod
    def jaccard(a, b):
        A, B = set(a), set(b)
        if not A and not B:
            return 0
        return len(A & B) / len(A | B)

    def compute_concept_similarity_matrix(self):
        nA, nB = len(self.boardA), len(self.boardB)
        mat = np.zeros((nA, nB))

        for i in range(nA):
            for j in range(nB):
                mat[i][j] = self.jaccard(
                    self.boardA.loc[i, "concepts"],
                    self.boardB.loc[j, "concepts"],
                )

        self.concept_sim_matrix = mat

    # -----------------------------------------------------------
    # Clustering (FAISS → sklearn fallback)
    # -----------------------------------------------------------
    def cluster_embeddings(self):
        from sklearn.decomposition import PCA

        all_emb = np.vstack([self.embA, self.embB])
        reduced = PCA(n_components=self.pca_components).fit_transform(all_emb)

        lenA = len(self.boardA)
        lenB = len(self.boardB)

        # Try FAISS
        try:
            import faiss
            reduced32 = reduced.astype("float32")
            k = max(int(np.sqrt(len(reduced))), 5)
            km = faiss.Kmeans(d=reduced32.shape[1], k=k, niter=20)
            km.train(reduced32)
            _, labels = km.index.search(reduced32, 1)
            labels = labels.flatten()
        except:
            from sklearn.cluster import KMeans
            k = max(int(np.sqrt(len(reduced))), 5)
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(reduced)

        self.clusterA = labels[:lenA]
        self.clusterB = labels[lenA: lenA + lenB]

        mat = np.zeros((lenA, lenB), dtype=bool)
        for i in range(lenA):
            for j in range(lenB):
                mat[i][j] = self.clusterA[i] == self.clusterB[j]

        self.cluster_match_matrix = mat

    # -----------------------------------------------------------
    # Agreement Engine (Weighted Score + Match Verdict)
    # -----------------------------------------------------------
    def run_agreement_engine(self):
        logger.info("Running Agreement Engine...")
    
        w_emb, w_topic, w_concept, w_cluster = 0.45, 0.25, 0.20, 0.10
    
        results = []
    
        for i, rowA in self.boardA.iterrows():
            for j, rowB in self.boardB.iterrows():
    
                emb = self.emb_sim_matrix[i][j]
                topic = self.topic_sim_matrix[i][j]
                concept = self.concept_sim_matrix[i][j]
                cluster_flag = bool(self.cluster_match_matrix[i][j])
    
                vote_emb = emb >= self.thresholds["emb"]
                vote_topic = topic >= self.thresholds["topic"]
                vote_concept = concept >= self.thresholds["concept"]
                vote_cluster = cluster_flag
    
                score = sum([vote_emb, vote_topic, vote_concept, vote_cluster])
    
                verdict = (
                    "STRONG MATCH" if score >= 3
                    else "POSSIBLE MATCH" if score == 2
                    else "NO MATCH"
                )
    
                # 🚀 NEW: Skip NO MATCH to prevent giant file sizes
                if verdict == "NO MATCH":
                    continue
    
                # Weighted score
                final_pct = (
                    emb * w_emb +
                    topic * w_topic +
                    concept * w_concept +
                    (1 if cluster_flag else 0) * w_cluster
                ) * 100
    
                topicA = str(rowA.get("TopicName", "N/A"))
                topicB = str(rowB.get("TopicName", "N/A"))
    
                results.append({
                    "BoardA_Index": int(i),
                    "BoardB_Index": int(j),
    
                    f"{self.boardA_name} Chapter": str(rowA.get("ChapterName", "N/A")),
                    f"{self.boardB_name} Chapter": str(rowB.get("ChapterName", "N/A")),
    
                    f"{self.boardA_name} TopicName": topicA,
                    f"{self.boardB_name} TopicName": topicB,
    
                    "Embedding_Sim": float(emb),
                    "Topic_Sim": float(topic),
                    "Concept_Sim": float(concept),
                    "Cluster_Match": cluster_flag,
    
                    "Votes_Agreed": score,
                    "Verdict": verdict,
                    "Match_Percentage": round(final_pct, 2)
                })
    
        self.results_df = pd.DataFrame(results)


    # -----------------------------------------------------------
    # Run Everything
    # -----------------------------------------------------------
    def run_all(self):
        self.load_data()
        self.preprocess()
        self.build_combined_text()
        self.generate_embeddings()
        self.compute_embedding_similarity()
        self.compute_topic_similarity()
        self.extract_concepts()
        self.compute_concept_similarity_matrix()
        self.cluster_embeddings()
        self.run_agreement_engine()

    # -----------------------------------------------------------
    # Export Results
    # -----------------------------------------------------------
    def export_results(self, out_path="agreement_output.csv"):
        ext = os.path.splitext(out_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            self.results_df.to_excel(out_path, index=False)
        else:
            self.results_df.to_csv(out_path, index=False)

        logger.info(f"Results exported: {out_path}")

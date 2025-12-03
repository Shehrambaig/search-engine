import os, json, time, pickle, ssl
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# SSL workaround for NLTK downloads (had issues on Mac)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
     ssl._create_default_https_context = _create_unverified_https_context
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
     nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
      nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
     nltk.download('wordnet', quiet=True)
class LRUCache:
    def __init__(self, capacity=100):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
    def get(self, key):
        if key not in self.cache:
            self.misses += 1
            return None
        self.hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # remove the older content
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_queries': total,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)}
class DocumentPreprocessor:
    def __init__(self, use_stopwords=True, use_stemming=False, use_lemmatization=True):
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

        if use_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
              self.stop_words = set()

        self.stemmer = PorterStemmer() if use_stemming else None
        if use_lemmatization:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        else:
              self.nlp = None

    def preprocess_text(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalnum()]
        if self.use_stopwords:
             tokens = [t for t in tokens if t not in self.stop_words]
        # either stem or lemmatize, not both
        if self.use_stemming and self.stemmer:
               tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.use_lemmatization and self.nlp:
            doc = self.nlp(' '.join(tokens))
            tokens = [token.lemma_ for token in doc]
        return tokens
    def preprocess_documents(self, documents):
        return [self.preprocess_text(doc) for doc in documents]
class HybridIRSystem:
    # I have implemented hybrid system combining bm25 + neural reranking
    # I want to use bm25 for fast retrieval of candidates,
    # then rerank top results with sentence transformers for better relevance
    def __init__(self, bm25_k1=1.5,bm25_b=0.75,neural_model='all-MiniLM-L6-v2',top_k_candidates=100,final_top_k=10,cache_size=100,hybrid_alpha=0.5):
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.top_k_candidates = top_k_candidates
        self.final_top_k = final_top_k
        self.hybrid_alpha = hybrid_alpha
        self.preprocessor = None
        self.bm25 = None
        self.neural_model = None
        self.documents_df = None
        self.tokenized_corpus = None
        self.original_texts = None
        self.cache = LRUCache(capacity=cache_size)
        self.neural_model_name = neural_model
        self.query_times = []
    def load_data(self, csv_path):
        print(f"Loading data from {csv_path}...")
        self.documents_df = pd.read_csv(csv_path, encoding='latin-1')
        self.documents_df['full_text'] = (
                self.documents_df['Article'].fillna('') + ' ' +
                self.documents_df['Heading'].fillna('') + ' ' +
                self.documents_df['NewsType'].fillna(''))
        self.documents_df['doc_id'] = range(len(self.documents_df))
        self.original_texts = self.documents_df['full_text'].tolist()
        print(f"Loaded {len(self.documents_df)} documents")
        return self.documents_df
    def build_index(self, use_stopwords=True, use_stemming=False, use_lemmatization=True):
        print("Building index...")
        start = time.time()
        self.preprocessor = DocumentPreprocessor(
            use_stopwords=use_stopwords,
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization)
        print("Preprocessing docs (this takes a while)...")
        self.tokenized_corpus = self.preprocessor.preprocess_documents(self.original_texts)
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
        print(f"Loading neural model: {self.neural_model_name}...")
        self.neural_model = SentenceTransformer(self.neural_model_name)
        elapsed = time.time() - start
        print(f"Index built in {elapsed:.2f}s")
        return {
            'index_time': elapsed,
            'num_documents': len(self.tokenized_corpus),
            'preprocessing': {
                'stopwords': use_stopwords,
                'stemming': use_stemming,
                'lemmatization': use_lemmatization}}
    def retrieve_bm25(self, query):
        query_tokens = self.preprocessor.preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:self.top_k_candidates]
        top_scores = scores[top_idx]
        return list(zip(top_idx, top_scores))
    def rerank_neural(self, query, candidates):
        if not candidates:
            return []
        candidate_idx = [idx for idx, _ in candidates]
        candidate_texts = [self.original_texts[idx] for idx in candidate_idx]
        q_emb = self.neural_model.encode([query])
        doc_embs = self.neural_model.encode(candidate_texts)
        sims = cosine_similarity(q_emb, doc_embs)[0]
        results = []
        max_bm25 = np.max([s for _, s in candidates])
        for i, (doc_id, bm25_score) in enumerate(candidates):
            neural_score = sims[i]
            bm25_norm = bm25_score / (max_bm25 + 1e-10)
            hybrid = self.hybrid_alpha * bm25_norm + (1 - self.hybrid_alpha) * neural_score
            results.append((doc_id, hybrid, neural_score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.final_top_k]
    def search(self, query, use_cache=True, use_reranking=True):
        start = time.time()
        # check cache first
        cache_key = f"{query}_{use_reranking}_{self.hybrid_alpha}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached['from_cache'] = True
                cached['query_time'] = time.time() - start
                return cached
        bm25_start = time.time()
        candidates = self.retrieve_bm25(query)
        bm25_time = time.time() - bm25_start
        if use_reranking:
            rerank_start = time.time()
            results = self.rerank_neural(query, candidates)
            rerank_time = time.time() - rerank_start
        else:
            results = [(idx, score, 0.0) for idx, score in candidates[:self.final_top_k]]
            rerank_time = 0.0
        result_docs = []
        for doc_id, hybrid_score, neural_score in results:
            doc = self.documents_df.iloc[doc_id]
            result_docs.append({
                'doc_id': int(doc_id),
                'article': doc['Article'],
                'date': doc['Date'],
                'heading': doc['Heading'],
                'news_type': doc['NewsType'],
                'hybrid_score': float(hybrid_score),
                'neural_score': float(neural_score),
                'text_snippet': self.original_texts[doc_id][:200] + '..'})
        query_time = time.time() - start
        self.query_times.append(query_time)
        result = {
            'query': query,
            'results': result_docs,
            'num_results': len(result_docs),
            'query_time': query_time,
            'bm25_time': bm25_time,
            'rerank_time': rerank_time,
            'from_cache': False}
        # cache the result
        if use_cache:
            self.cache.put(cache_key, result)
        return result
    def save_index(self, save_dir='data/processed'):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving index to {save_dir}...")
        with open(f'{save_dir}/bm25_index.pkl', 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(f'{save_dir}/tokenized_corpus.pkl', 'wb') as f:
            pickle.dump(self.tokenized_corpus, f)
        with open(f'{save_dir}/preprocessor_config.json', 'w') as f:
            json.dump({
                'use_stopwords': self.preprocessor.use_stopwords,
                'use_stemming': self.preprocessor.use_stemming,
                'use_lemmatization': self.preprocessor.use_lemmatization}, f)
        self.documents_df.to_csv(f'{save_dir}/documents.csv', index=False)
        print("Index saved")
    def load_index(self, load_dir='data/processed'):
        print(f"Loading index from {load_dir}...")
        with open(f'{load_dir}/bm25_index.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(f'{load_dir}/tokenized_corpus.pkl', 'rb') as f:
            self.tokenized_corpus = pickle.load(f)
        with open(f'{load_dir}/preprocessor_config.json', 'r') as f:
              config = json.load(f)
        self.preprocessor = DocumentPreprocessor(**config)
        self.documents_df = pd.read_csv(f'{load_dir}/documents.csv')
        self.original_texts = self.documents_df['full_text'].tolist()
        # still need to load the neural model
        print(f"Loading neural model: {self.neural_model_name}...")
        self.neural_model = SentenceTransformer(self.neural_model_name)
        print("Done loading")
    def get_stats(self):
        avg_qt = np.mean(self.query_times) if self.query_times else 0
        return {'num_documents': len(self.documents_df) if self.documents_df is not None else 0,'num_queries': len(self.query_times),'avg_query_time': avg_qt,'cache_stats': self.cache.get_stats(),
            'bm25_params': {
                'k1': self.bm25_k1,
                'b': self.bm25_b},
            'hybrid_alpha': self.hybrid_alpha}
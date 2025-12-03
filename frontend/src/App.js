import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryTime, setQueryTime] = useState(null);
  const [fromCache, setFromCache] = useState(false);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`);
      setStats(response.data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/search`, {
        query: query,
        use_reranking: true,
        use_cache: true
      });

      setResults(response.data.results);
      setQueryTime(response.data.query_time);
      setFromCache(response.data.from_cache);
      fetchStats();
    } catch (err) {
      setError(err.response?.data?.detail || 'Search failed');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>Hybrid Search Engine</h1>
        <p>BM25 + Neural Reranking with caching</p>
      </header>

      <div className="container">
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search articles..."
            className="search-input"
            disabled={loading}
          />
          <button type="submit" disabled={loading} className="search-button">
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {queryTime && (
          <div className="query-info">
            Query time: {(queryTime * 1000).toFixed(2)}ms
            {fromCache && <span className="cache-badge">From Cache</span>}
          </div>
        )}

        {error && <div className="error">{error}</div>}

        {results.length > 0 && (
          <div className="results">
            <h2>{results.length} Results</h2>
            {results.map((doc, idx) => (
              <div key={idx} className="result-card">
                <h3>{doc.heading}</h3>
                <p className="article-text">{doc.article.substring(0, 200)}...</p>
                <div className="meta">
                  <span>{doc.date}</span>
                  <span className="news-type">{doc.news_type}</span>
                </div>
                <div className="scores">
                  <span>Hybrid: {doc.hybrid_score.toFixed(3)}</span>
                  <span>Neural: {doc.neural_score.toFixed(3)}</span>
                  <span className="doc-id">ID: {doc.doc_id}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {stats && (
          <div className="stats">
            <h3>System Stats</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Documents</span>
                <span className="stat-value">{stats.num_documents}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Queries</span>
                <span className="stat-value">{stats.num_queries}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Avg Query Time</span>
                <span className="stat-value">{(stats.avg_query_time * 1000).toFixed(2)}ms</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Cache Hit Rate</span>
                <span className="stat-value">{(stats.cache_stats.hit_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

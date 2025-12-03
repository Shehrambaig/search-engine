import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def setup_dark_theme():
    """Configure matplotlib for professional black & grey theme"""
    plt.style.use('dark_background')

    # Grey scale color palette
    colors = {
        'bg': '#0a0a0a',           # Almost black background
        'fg': '#e0e0e0',           # Light grey for text
        'grid': '#2a2a2a',         # Dark grey for grid
        'bars': ['#c0c0c0', '#a0a0a0', '#808080', '#606060'],  # Grey shades for bars
        'positive': '#b0b0b0',     # Light grey for positive values
        'negative': '#505050',     # Dark grey for negative values
    }

    # Apply settings
    plt.rcParams.update({
        'figure.facecolor': colors['bg'],
        'axes.facecolor': colors['bg'],
        'axes.edgecolor': '#606060',
        'axes.labelcolor': colors['fg'],
        'axes.titlecolor': colors['fg'],
        'xtick.color': colors['fg'],
        'ytick.color': colors['fg'],
        'grid.color': colors['grid'],
        'text.color': colors['fg'],
        'legend.facecolor': colors['bg'],
        'legend.edgecolor': '#606060',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    return colors


class IREvaluator:
    def __init__(self, ir_system):
        self.ir_system = ir_system
        self.results = defaultdict(list)

    def precision_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / k if k > 0 else 0.0

    def recall_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0

    def average_precision(self, retrieved: List[int], relevant: List[int]) -> float:
        if not relevant:
            return 0.0
        score = 0.0
        num_relevant = 0

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant += 1
                score += num_relevant / i

        return score / len(relevant) if len(relevant) > 0 else 0.0

    def reciprocal_rank(self, retrieved: List[int], relevant: List[int]) -> float:
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    def evaluate_query(self, query: str, relevant_docs: List[int],
                       use_reranking: bool = True) -> Dict:
        result = self.ir_system.search(query, use_reranking=use_reranking, use_cache=False)
        retrieved = [doc['doc_id'] for doc in result['results']]
        metrics = {
            'query': query,
            'precision@5': self.precision_at_k(retrieved, relevant_docs, 5),
            'precision@10': self.precision_at_k(retrieved, relevant_docs, 10),
            'recall@10': self.recall_at_k(retrieved, relevant_docs, 10),
            'average_precision': self.average_precision(retrieved, relevant_docs),
            'reciprocal_rank': self.reciprocal_rank(retrieved, relevant_docs),
            'query_time': result['query_time'],
            'num_retrieved': len(retrieved),
            'num_relevant': len(relevant_docs),
            'relevant_retrieved': len(set(retrieved) & set(relevant_docs))
        }

        return metrics

    def evaluate_queries(self, test_queries: List[Dict], use_reranking: bool = True) -> Dict:
        all_metrics = []
        print(f"\n=== Evaluating {len(test_queries)} queries ===")
        print(f"Reranking: {'ON' if use_reranking else 'OFF'}")
        for i, test_case in enumerate(test_queries, 1):
            print(f"Query {i}/{len(test_queries)}: {test_case['query']}")
            metrics = self.evaluate_query(
                test_case['query'],
                test_case['relevant_docs'],
                use_reranking=use_reranking)
            all_metrics.append(metrics)
        aggregate = {
            'mean_precision@5': np.mean([m['precision@5'] for m in all_metrics]),
            'mean_precision@10': np.mean([m['precision@10'] for m in all_metrics]),
            'mean_recall@10': np.mean([m['recall@10'] for m in all_metrics]),
            'MAP': np.mean([m['average_precision'] for m in all_metrics]),
            'MRR': np.mean([m['reciprocal_rank'] for m in all_metrics]),
            'avg_query_time': np.mean([m['query_time'] for m in all_metrics]),
            'total_queries': len(test_queries),
            'use_reranking': use_reranking}

        return {
            'aggregate': aggregate,
            'per_query': all_metrics}

    def ablation_study(self, test_queries: List[Dict]) -> Dict:
        print("\n=== Running Ablation Study ===")
        results = {}
        configs = [
            {'name': 'Full (Stopwords + Lemma)', 'stopwords': True, 'stemming': False, 'lemma': True},
            {'name': 'No Stopwords', 'stopwords': False, 'stemming': False, 'lemma': True},
            {'name': 'Stemming', 'stopwords': True, 'stemming': True, 'lemma': False},
            {'name': 'No Processing', 'stopwords': False, 'stemming': False, 'lemma': False}]

        for config in configs:
            print(f"\nTesting: {config['name']}")
            self.ir_system.build_index(
                use_stopwords=config['stopwords'],
                use_stemming=config['stemming'],
                use_lemmatization=config['lemma'])
            with_rerank = self.evaluate_queries(test_queries, use_reranking=True)
            without_rerank = self.evaluate_queries(test_queries, use_reranking=False)

            results[config['name']] = {
                'config': config,
                'with_reranking': with_rerank['aggregate'],
                'without_reranking': without_rerank['aggregate']}

        return results

    def test_bm25_parameters(self, test_queries: List[Dict]) -> Dict:
        print("testing bm25")
        results = {}
        param_configs = [
            {'k1': 1.2, 'b': 0.75},
            {'k1': 1.5, 'b': 0.75},
            {'k1': 2.0, 'b': 0.75},
            {'k1': 1.5, 'b': 0.5},
            {'k1': 1.5, 'b': 1.0}]

        for params in param_configs:
            config_name = f"k1={params['k1']}, b={params['b']}"
            print(f"\nTesting: {config_name}")
            self.ir_system.bm25_k1 = params['k1']
            self.ir_system.bm25_b = params['b']
            self.ir_system.build_index()
            eval_results = self.evaluate_queries(test_queries, use_reranking=True)
            results[config_name] = eval_results['aggregate']

        return results

    def generate_report(self, results: Dict, output_dir: str = 'results'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{output_dir}/plots').mkdir(exist_ok=True)
        print(f"\n=== Generating Report ===")
        with open(f'{output_dir}/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        self._plot_metric_comparison(results, output_dir)
        self._plot_ablation_results(results, output_dir)
        self._generate_summary_report(results, output_dir)
        print(f"Report saved to {output_dir}/")

    def _plot_metric_comparison(self, results: Dict, output_dir: str):
        if 'ablation_study' not in results:
            return

        # Setup dark theme
        colors = setup_dark_theme()

        ablation_results = results['ablation_study']
        configs = list(ablation_results.keys())

        metrics_with = {
            'P@5': [ablation_results[c]['with_reranking']['mean_precision@5'] for c in configs],
            'P@10': [ablation_results[c]['with_reranking']['mean_precision@10'] for c in configs],
            'MAP': [ablation_results[c]['with_reranking']['MAP'] for c in configs],
            'MRR': [ablation_results[c]['with_reranking']['MRR'] for c in configs],
        }

        metrics_without = {
            'P@5': [ablation_results[c]['without_reranking']['mean_precision@5'] for c in configs],
            'P@10': [ablation_results[c]['without_reranking']['mean_precision@10'] for c in configs],
            'MAP': [ablation_results[c]['without_reranking']['MAP'] for c in configs],
            'MRR': [ablation_results[c]['without_reranking']['MRR'] for c in configs],
        }
        x = np.arange(len(configs))
        width = 0.15

        fig, ax = plt.subplots(figsize=(14, 6))

        # Use grey color palette
        bar_colors = colors['bars']
        for i, (metric, values) in enumerate(metrics_with.items()):
            ax.bar(x + i * width, values, width, label=f'{metric} (with rerank)',
                   color=bar_colors[i % len(bar_colors)], alpha=0.9, edgecolor='#404040')

        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Ablation Study: Metric Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend(framealpha=0.9)
        ax.grid(axis='y', alpha=0.2, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/ablation_comparison.png', dpi=300, facecolor=colors['bg'])
        plt.close()
        # Second plot: Reranking impact
        fig, ax = plt.subplots(figsize=(10, 6))

        map_improvement = [
            (ablation_results[c]['with_reranking']['MAP'] -
             ablation_results[c]['without_reranking']['MAP']) * 100
            for c in configs
        ]

        # Grey scale colors for positive/negative
        bar_colors = [colors['positive'] if x > 0 else colors['negative'] for x in map_improvement]
        ax.bar(configs, map_improvement, color=bar_colors, alpha=0.9, edgecolor='#404040')
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('MAP Improvement (%)', fontweight='bold')
        ax.set_title('Reranking Impact on MAP', fontweight='bold', pad=20)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.axhline(y=0, color='#808080', linestyle='-', linewidth=1.5)
        ax.grid(axis='y', alpha=0.2, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/reranking_impact.png', dpi=300, facecolor=colors['bg'])
        plt.close()

    def _plot_ablation_results(self, results: Dict, output_dir: str):
        if 'bm25_parameters' not in results:
            return

        # Setup dark theme
        colors = setup_dark_theme()

        param_results = results['bm25_parameters']

        configs = list(param_results.keys())
        map_scores = [param_results[c]['MAP'] for c in configs]
        mrr_scores = [param_results[c]['MRR'] for c in configs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: MAP scores
        ax1.bar(range(len(configs)), map_scores, alpha=0.9,
                color='#a0a0a0', edgecolor='#404040')
        ax1.set_xlabel('BM25 Configuration', fontweight='bold')
        ax1.set_ylabel('MAP Score', fontweight='bold')
        ax1.set_title('Mean Average Precision by BM25 Parameters',
                      fontweight='bold', pad=20)
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.2, linestyle='--')

        # Right plot: MRR scores
        ax2.bar(range(len(configs)), mrr_scores, alpha=0.9,
                color='#808080', edgecolor='#404040')
        ax2.set_xlabel('BM25 Configuration', fontweight='bold')
        ax2.set_ylabel('MRR Score', fontweight='bold')
        ax2.set_title('Mean Reciprocal Rank by BM25 Parameters',
                      fontweight='bold', pad=20)
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.2, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/bm25_parameters.png', dpi=300, facecolor=colors['bg'])
        plt.close()

    def _generate_summary_report(self, results: Dict, output_dir: str):
        with open(f'{output_dir}/summary_report.txt', 'w') as f:
            f.write("HYBRID IR SYSTEM EVALUATION REPORT")
            f.write("SYSTEM CONFIGURATION")
            stats = self.ir_system.get_stats()
            f.write(f"Number of documents: {stats['num_documents']}")
            f.write(f"BM25 parameters: k1={stats['bm25_params']['k1']}, b={stats['bm25_params']['b']}")
            f.write(f"Hybrid alpha: {stats['hybrid_alpha']}")
            f.write(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
            if 'ablation_study' in results:
                f.write("ABLATION STUDY RESULTS")

                for config_name, config_results in results['ablation_study'].items():
                    f.write(f"\n{config_name}:\n")

                    with_rerank = config_results['with_reranking']
                    without_rerank = config_results['without_reranking']

                    f.write(f"  With Reranking:\n")
                    f.write(f"    MAP: {with_rerank['MAP']:.4f}")
                    f.write(f"    MRR: {with_rerank['MRR']:.4f}")
                    f.write(f"    P@10: {with_rerank['mean_precision@10']:.4f}")

                    f.write(f"  Without Reranking:")
                    f.write(f"    MAP: {without_rerank['MAP']:.4f}")
                    f.write(f"    MRR: {without_rerank['MRR']:.4f}\n")
                    f.write(f"    P@10: {without_rerank['mean_precision@10']:.4f}\n")

                    improvement = ((with_rerank['MAP'] - without_rerank['MAP']) /
                                   without_rerank['MAP'] * 100) if without_rerank['MAP'] > 0 else 0
                    f.write(f"  MAP Improvement: {improvement:+.2f}%\n")
            if 'bm25_parameters' in results:
                f.write("\n\nBM25 PARAMETER TUNING\n")
                f.write("-" * 80 + "\n")

                best_config = max(results['bm25_parameters'].items(),
                                  key=lambda x: x[1]['MAP'])

                f.write(f"\nBest Configuration: {best_config[0]}\n")
                f.write(f"  MAP: {best_config[1]['MAP']:.4f}\n")
                f.write(f"  MRR: {best_config[1]['MRR']:.4f}\n")
                f.write(f"  P@10: {best_config[1]['mean_precision@10']:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Summary report saved to {output_dir}/summary_report.txt")
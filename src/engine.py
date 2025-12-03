
import os, sys, yaml
from main import HybridIRSystem

k1_default = 1.6
b_default = 0.75
model_default = 'all-MiniLM-L6-v2'


class InteractiveSearch:
    def __init__(self, cfg='../config.yaml'):
        self.sys = None
        self.cfg = {}
        if os.path.exists(cfg):
            with open(cfg) as f:
                self.cfg = yaml.safe_load(f)

    def setup(self):
        # bm25 stuff
        bm25_cfg = self.cfg.get('bm25', {}) or k1_default
        k1 = bm25_cfg.get('k1', k1_default)
        b = bm25_cfg.get('b', b_default)

        neural_cfg = self.cfg.get('neural', {})
        model = neural_cfg.get('model_name', model_default)

        ret = self.cfg.get('retrieval', {})
        top_k = ret.get('top_k_candidates', 100)
        final_k = ret.get('final_top_k', 10)

        cache_cfg = self.cfg.get('cache', {})
        cache_sz = cache_cfg.get('capacity', 100)

        hybrid_cfg = self.cfg.get('hybrid', {})
        alpha = hybrid_cfg.get('alpha', 0.5)

        self.sys = HybridIRSystem(
            bm25_k1=k1, bm25_b=b,
            neural_model=model,
            top_k_candidates=top_k,
            final_top_k=final_k,
            cache_size=cache_sz,
            hybrid_alpha=alpha)

        data = self.cfg.get('data', {})
        idx_dir = data.get('processed_dir', 'data/processed')
        idx_file = f'{idx_dir}/bm25_index.pkl'

        if os.path.exists(idx_file):
            # print("Loading existing index...")
            self.sys.load_index(idx_dir)
        else:
            csv = data.get('raw_csv', '/Users/shehrambaig/PycharmProjects/Search-engine/data/raw/Articles.csv')
            if not os.path.exists(csv):
                print("CSV not found!")
                sys.exit(1)

            self.sys.load_data(csv)
            pp = self.cfg.get('preprocessing', {})

            self.sys.build_index(
                use_stopwords=pp.get('use_stopwords', True),
                use_stemming=pp.get('use_stemming', False),  # stemming is kinda meh
                use_lemmatization=pp.get('use_lemma', True))

            self.sys.save_index(idx_dir)

    def search(self, q):
        results = self.sys.search(q)

        if not results.get('results'):
            print("Nothing found :(")
            return

        # show results
        for i, doc in enumerate(results['results'][:self.sys.final_top_k], 1):
            print(f"{i}. {doc['heading']}")
            # truncate long articles
            article_text = doc['article'][:150] + "..." if len(doc['article']) > 150 else doc['article']
            print(f"   {article_text}")
            print(f"   {doc['date']} | {doc['news_type']}")

            # scores for debugging
            h_score = doc['hybrid_score']
            n_score = doc['neural_score']
            print(f"   hybrid={h_score:.3f}, neural={n_score:.3f} [id={doc['doc_id']}]")
            print()

    def run(self):
        self.setup()

        print("Enter queries (q to quit):")
        while True:
            try:
                query = input("> ").strip()
                if query == 'q' or query == 'quit':
                    break

                if query:
                    self.search(query)

            except KeyboardInterrupt:
                print("\nBye!")
                break
            # catch other errors but don't do anything special
            except:
                print("Something went wrong, try again")


if __name__ == '__main__':
    app = InteractiveSearch()
    app.run()
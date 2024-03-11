from operator import itemgetter

import nltk
from gensim.parsing.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from math import sqrt, log

from BM25 import BM25
from utils import *

B_TITLE = 0.5
K1_TITLE = 1.1
B_TEXT = 0.45
K1_TEXT = 1.2
MAX_DOCS = 100
MAX_DOC_LEN = 100_000
TITLE_WEIGHT = 0.2


class Search:
    STEM_BUCKET_NAME = 'bgu-ir-ass3-fab-stem'
    PROJECT_NAME = 'ir-ass3-414111'

    def __init__(self) -> None:
        nltk.download('stopwords')
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = english_stopwords.union(corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        self.stemmer = PorterStemmer()
        self.inverted_title = load_pickle('index_title.pkl', self.STEM_BUCKET_NAME, self.PROJECT_NAME)
        self.inverted_text = load_pickle('index_text.pkl', self.STEM_BUCKET_NAME, self.PROJECT_NAME)
        self.page_views = load_pickle('pageviews_log.pkl', self.STEM_BUCKET_NAME, self.PROJECT_NAME)
        self.pagerank = load_pickle('normalized_pagerank_iter10.pkl', self.STEM_BUCKET_NAME, self.PROJECT_NAME)
        self.doc_title = load_pickle('doc_title.pkl', self.STEM_BUCKET_NAME, self.PROJECT_NAME)

    def retrieve_posting_list(self, query_word: str, bucket_name: str, inverted):
        return inverted.read_a_posting_list(base_dir='.', w=query_word, bucket_name=bucket_name)

    def weights(self, doc_id, bm25_score, def_rank=0.2, def_views=3.7e-6):
        pagerank = self.pagerank.get(doc_id, def_rank)
        page_views = self.page_views.get(doc_id, def_views)

        adjusted_pagerank = sqrt(pagerank + 1)
        adjusted_bm25_score = (bm25_score ** 3)
        adjusted_pageviews = log(page_views, 2)
        return adjusted_pagerank * adjusted_bm25_score * adjusted_pageviews

    def get_scores(self, tokens, bucket_name, inverted, k1, b, func=None):
        token_pl = {
            token: self.retrieve_posting_list(token, bucket_name, inverted)
            for token in tokens
        }
        bm25 = BM25(token_pl, inverted, k1=k1, b=b)
        scores = bm25.calculate_bm25()
        if func is not None:
            scores = list(map(lambda x: (x[0], func(x[1])), scores))
        return scores

    def search_query(self, query: str):
        stemmer = self.stemmer
        print('begin')
        tokens = [token.group() for token in self.RE_WORD.finditer(query.lower())]
        tokens = [token for token in tokens if token not in self.all_stopwords]
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        print('pl')

        scores_text = self.get_scores(
            stemmed_tokens,
            self.STEM_BUCKET_NAME,
            self.inverted_text,
            k1=K1_TEXT,
            b=B_TEXT,
        )
        scores_title = self.get_scores(
            stemmed_tokens,
            self.STEM_BUCKET_NAME,
            self.inverted_title,
            k1=K1_TITLE,
            b=B_TITLE,
            func=lambda x: x * TITLE_WEIGHT,
        )
        if len(scores_text) > MAX_DOC_LEN:
            sorted_scores = sorted(scores_text, key=itemgetter(1), reverse=True)
            scores_text = sorted(sorted_scores[:MAX_DOC_LEN], key=itemgetter(0), reverse=True)

        print('reduce together')
        scores = reduce_by_key([scores_title, scores_text])
        scores = list(map(lambda x: (x[0], self.weights(x[0], x[1])), scores))

        scores = sorted(scores, key=itemgetter(1), reverse=True)
        return [(str(doc_id), score) for doc_id, score in scores[:MAX_DOCS]]

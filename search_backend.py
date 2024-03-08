import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import re
from math import sqrt

from BM25 import BM25
from utils import *



class Search:
    BUCKET_NAME = 'bgu-ir-ass3-fab'
    PROJECT_NAME = 'ir-ass3-414111'

    def __init__(self) -> None:
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = english_stopwords.union(corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        self.inverted_title = load_pickle('inverted_title_v1.pkl', 'postings_gcp/title', self.BUCKET_NAME, self.PROJECT_NAME)
        self.inverted_text = load_pickle('inverted_text_v1.pkl', 'postings_gcp/text', self.BUCKET_NAME, self.PROJECT_NAME)
        self.page_views = load_pickle('pageviews_log.pkl', 'dicts_pickles', self.BUCKET_NAME, self.PROJECT_NAME)

    def retrieve_posting_list(self, query_word: str, bucket_name: str, inverted):
        """Retrieve the posting list for a query word.
        Args:
            query_word: the query word
            bucket_name: the name of the bucket
            inverted: the inverted index
        Returns:
            the posting list
        """
        return inverted.read_a_posting_list(base_dir='.', w=query_word, bucket_name=bucket_name)

    def calculate_bm25_per_title_term(self, token, doc_id_tf):
        """Calculate the BM25 score for a term in a document.
        Args:
            token: the term
            doc_id_tf: a tuple (doc_id, tf)
            inverted: the inverted index for title
        Returns:
            the BM25 score
        """
        k1, b = 1.2, 0.5
        return self.calculate_bm25_per_term(token, doc_id_tf, self.inverted_title, k1=k1, b=b)

    def calculate_bm25_per_text_term(self, token, doc_id_tf):
        """Calculate the BM25 score for a term in a document.
        Args:
            token: the term
            doc_id_tf: a tuple (doc_id, tf)
            inverted: the inverted index for text
        Returns:
            the BM25 score
        """
        k1, b = 1.2, 0.5
        return self.calculate_bm25_per_term(token, doc_id_tf, self.inverted_text, k1=k1, b=b)    

    def search_query(self, query: str, pagerank_normalized):
        """Query the inverted index.
        Args:
            query: the query string
            bucket_name: the name of the bucket
            inverted_title: the inverted index for title
            inverted_text: the inverted index for text
            pagerank_normalized: the normalized pagerank
        Returns:
            a list of tuples (doc_id, score)
        """
        print('begin')
        tokens = [token.group() for token in self.RE_WORD.finditer(query.lower())]
        tokens = [token for token in tokens if token not in self.all_stopwords]
        print('pl')
        token_pl_title = {
            token: self.retrieve_posting_list(token, self.BUCKET_NAME, self.inverted_title)
            for token in tokens
        }
        token_pl_text = {
            token: self.retrieve_posting_list(token, self.BUCKET_NAME, self.inverted_text)
            for token in tokens
        }

        bm25_title = BM25(token_pl_title, self.inverted_title, k1=1.1, b=0.5)
        bm25_text = BM25(token_pl_text, self.inverted_text, k1=1.2, b=0.5)

        scores_title = bm25_title.calculate_bm25()
        # bm25_title = list(map(lambda x: (x[0], x[1]), bm25_title))
        scores_text = bm25_text.calculate_bm25()
        # bm25_text = list(map(lambda x: (x[0], x[1]), bm25_text))
        print('reduce together')
        scores = reduce_by_key([scores_title, scores_text])
        scores = list(map(lambda x: (x[0], sqrt(pagerank_normalized.get(x[0], 0.2) + 1) * (x[1] ** 3) * self.page_views.get(x[0], 3.7e-6)), scores))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [(str(doc_id), score) for doc_id, score in scores[:100]]

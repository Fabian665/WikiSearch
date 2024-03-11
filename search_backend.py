from operator import itemgetter
from typing import Dict, Callable, Optional

import nltk
from gensim.parsing.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from math import sqrt, log

from custom_types import DocId, Score, PostingList, Tokens
from inverted_index_gcp import InvertedIndex

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
    """Search class for the search engine.
    """
    STEM_BUCKET_NAME = 'bgu-ir-ass3-fab-stem'
    PROJECT_NAME = 'ir-ass3-414111'
    BUCKET_PROJECT = STEM_BUCKET_NAME, PROJECT_NAME

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
        self.inverted_title: InvertedIndex = load_pickle('index_title.pkl', *self.BUCKET_PROJECT)
        self.inverted_text: InvertedIndex = load_pickle('index_text.pkl', *self.BUCKET_PROJECT)
        self.page_views: Dict[DocId, float] = load_pickle('pageviews_log.pkl', *self.BUCKET_PROJECT)
        self.pagerank: Dict[DocId, float] = load_pickle('normalized_pagerank_iter10.pkl', *self.BUCKET_PROJECT)
        self.doc_title: Dict[DocId, str] = load_pickle('doc_title.pkl', *self.BUCKET_PROJECT)

    @staticmethod
    def retrieve_posting_list(query_word: str, bucket_name: str, inverted: InvertedIndex) -> PostingList:
        """Retrieve a posting list from the inverted index.
        Args:
            query_word: the query word.
            bucket_name: the bucket name.
            inverted: the inverted index.
        Returns:
            The posting list of the query word.
        """
        return inverted.read_a_posting_list(base_dir='.', w=query_word, bucket_name=bucket_name)

    def weights(self, doc_id: DocId, bm25_score: Score) -> Score:
        """Calculate the score of the given document based on predetermined weights.
        Args:
            doc_id: the document id.
            bm25_score: the BM25 score of the document.
        Returns:
            The score of the document.
            """
        pagerank = self.pagerank.get(doc_id, 0.2)
        page_views = self.page_views.get(doc_id, 3.7e-6)

        adjusted_pagerank = sqrt(pagerank + 1)
        adjusted_bm25_score = (bm25_score ** 3)
        adjusted_pageviews = log(page_views, 2)
        return adjusted_pagerank * adjusted_bm25_score * adjusted_pageviews

    def get_scores(self,
                   tokens: Tokens,
                   bucket_name: str,
                   inverted: InvertedIndex,
                   k1: float,
                   b: float,
                   func: Optional[Callable] = None
                   ) -> RankedPostingList:
        """Get the BM25 scores of the given tokens.
        Args:
            tokens: the tokens.
            bucket_name: the bucket name.
            inverted: the inverted index.
            k1: the k1 hyperparameter.
            b: the b hyperparameter.
            func: the function to apply to the scores.
        Return:
            The BM25 scores of the given tokens.
        """
        token_pl = {
            token: self.retrieve_posting_list(token, bucket_name, inverted)
            for token in tokens
        }
        bm25 = BM25(token_pl, inverted, k1=k1, b=b)
        scores = bm25.calculate_bm25()
        if func is not None:
            scores = list(map(lambda x: (x[0], func(x[1])), scores))
        return scores

    def search_query(self, query: str) -> RankedPostingList:
        """Search the given query.
        Args:
            query: the query.
        Returns:
            The ranked posting list of the query.
        """
        stemmer = self.stemmer
        tokens = [token.group() for token in self.RE_WORD.finditer(query.lower())]
        tokens = [token for token in tokens if token not in self.all_stopwords]
        stemmed_tokens: Tokens = [stemmer.stem(token) for token in tokens]

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

        scores = reduce_by_key([scores_title, scores_text])
        scores = list(map(lambda x: (x[0], self.weights(x[0], x[1])), scores))

        scores = sorted(scores, key=itemgetter(1), reverse=True)
        return [(doc_id, score) for doc_id, score in scores[:MAX_DOCS]]

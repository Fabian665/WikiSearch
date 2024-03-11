from typing import Tuple, Dict, List

from custom_types import PostingList, DocId, Score, RankedPostingList
from inverted_index_gcp import InvertedIndex
from utils import reduce_by_key


class BM25:
    """BM25 ranking algorithm implementation.
    """
    UNMEANINGFUL = {'consid', 'could', 'describ', 'done', 'explain', 'find',
                    'identifi', 'known', 'locat', 'might', 'uncov', 'whose'}

    def __init__(self,
                 token_posting_list: Dict[str, PostingList],
                 inverted_index: InvertedIndex,
                 k1: float = 1.2,
                 b: float = 0.5
                 ) -> None:
        """Initialize the BM25 algorithm.
        Args:
            token_posting_list: a dictionary of token to posting list.
            inverted_index: the inverted index.
            k1: hyperparameter for the BM25 algorithm.
            b: hyperparameter for the BM25 algorithm.
            """
        self.doc_len = inverted_index.doc_len
        self.average_doc_length = inverted_index.avdl
        self.idf_bm25 = inverted_index.idf_bm25
        self.token_posting_list = token_posting_list
        self.k1 = k1
        self.b = b

    def calculate_bm25_per_term(self,
                                doc_id_tf: Tuple[DocId, Score],
                                token: str
                                ) -> Score:
        """Calculate the BM25 score for a given doc and term.
        Args:
            doc_id_tf: a tuple of doc_id and term frequency.
            token: the token for which the BM25 score is calculated.
        Returns:
            The BM25 score for the given doc and term.
            """
        doc_id, tf = doc_id_tf
        B = (1 - self.b) + (self.b * self.doc_len.get(doc_id, self.average_doc_length))
        tf_ij = ((self.k1 + 1) / (B * self.k1 + tf))
        return tf_ij * self.idf_bm25[token] * (1 if token not in self.UNMEANINGFUL else 0.75)

    def calculate_bm25_per_posting_list(self) -> List[RankedPostingList]:
        """Calculate the BM25 score for each posting list.
        Returns:
            A list of BM25 scores for each posting list.
        """
        bm25_per_doc = []
        for token, pl in self.token_posting_list.items():
            curr_term = [
                (doc_id, self.calculate_bm25_per_term((doc_id, tf), token))
                for doc_id, tf in pl
            ]
            bm25_per_doc.append(curr_term)
        return bm25_per_doc

    def calculate_bm25(self) -> RankedPostingList:
        """Calculate the BM25 score for the entire corpus.
        Returns:
            A list of BM25 scores for the entire corpus.
        """
        print('claculating BM25')
        postings_lists = self.calculate_bm25_per_posting_list()
        print('reducing')
        return reduce_by_key(postings_lists)

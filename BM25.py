from utils import reduce_by_key


class BM25:
    UNMEANINGFUL = {'consid', 'could', 'describ', 'done', 'explain', 'find',
                    'identifi', 'known', 'locat', 'might', 'uncov', 'whose'}
    def __init__(self, posting_list, inverted_index, k1=1.2, b=0.5, penalty=0.75):
        self.doc_len = inverted_index.doc_len
        self.avdl = inverted_index.avdl
        self.idf_bm25 = inverted_index.idf_bm25
        self.posting_list = posting_list
        self.k1 = k1
        self.b = b
        self.penalty = penalty

    def calculate_bm25_per_term(self, doc_id_tf, token):
        """Calculate the BM25 score for a term in a document.
        Args:
            token: the term
            doc_id_tf: a tuple (doc_id, tf)
        Returns:
            the BM25 score
        """
        doc_id, tf = doc_id_tf
        B = (1 - self.b) + (self.b * self.doc_len.get(doc_id, self.avdl))
        tf_ij = ((self.k1 + 1) / (B * self.k1 + tf))
        return tf_ij * self.idf_bm25[token] * (1 if token not in self.UNMEANINGFUL else self.penalty)

    def calculate_bm25_per_posting_list(self):
        """Calculate the BM25 score for a posting list.
        Args:
            token_pl: a dictionary of term:pl posting list
        Returns:
            a list of posting lists
        """
        bm25_per_doc = []
        for token, pl in self.posting_list.items():
            curr_term = [
                (doc_id, self.calculate_bm25_per_term((doc_id, tf), token))
                for doc_id, tf in pl
            ]
            bm25_per_doc.append(curr_term)
        return bm25_per_doc

    def calculate_bm25(self):
        """Calculate the BM25 score for a posting list.
        Args:
            token_pl: a dictionary of term:pl posting list
        Returns:
            a list of posting lists
        """
        print('claculating BM25')
        postings_lists = self.calculate_bm25_per_posting_list()
        print('reducing')
        return reduce_by_key(postings_lists)



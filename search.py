import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import re
from math import sqrt


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def retrieve_posting_list(query_word: str, bucket_name: str, inverted):
    """Retrieve the posting list for a query word.
    Args:
        query_word: the query word
        bucket_name: the name of the bucket
        inverted: the inverted index
    Returns:
        the posting list
    """
    return inverted.read_a_posting_list(base_dir='.', w=query_word, bucket_name=bucket_name)

def calculate_bm25_per_title_term(token, doc_id_tf, inverted):
    """Calculate the BM25 score for a term in a document.
    Args:
        token: the term
        doc_id_tf: a tuple (doc_id, tf)
        inverted: the inverted index for title
    Returns:
        the BM25 score
    """
    k1, b = 1.2, 0.5
    return calculate_bm25_per_term(token, doc_id_tf, inverted, k1=k1, b=b)

def calculate_bm25_per_text_term(token, doc_id_tf, inverted):
    """Calculate the BM25 score for a term in a document.
    Args:
        token: the term
        doc_id_tf: a tuple (doc_id, tf)
        inverted: the inverted index for text
    Returns:
        the BM25 score
    """
    k1, b = 1.2, 0.5
    return calculate_bm25_per_term(token, doc_id_tf, inverted, k1=k1, b=b)

def calculate_bm25_per_term(token, doc_id_tf, inverted, **kwargs):
    """Calculate the BM25 score for a term in a document.
    Args:
        token: the term
        doc_id_tf: a tuple (doc_id, tf)
        inverted: the inverted index
        **kwargs: the BM25 hyper-parameters
    Returns:
        the BM25 score
    """
    k1, b = kwargs['k1'], kwargs['b']
    # k3 = kwargs['k3']
    doc_id, tf = doc_id_tf
    B = (1 - b) + (b * inverted.doc_len.get(doc_id, inverted.avdl))
    tf_ij = ((k1 + 1) / (B * k1 + tf))
    return tf_ij * inverted.idf_bm25[token]


def remove_empty_postings(postings_lists):
    """Remove empty postings lists from a list of postings lists.
    Args:
        postings_lists: a list of postings lists, each postings list is a list of tuples (doc_id, score)
    """
    bad_indices = []
    for index, pl in enumerate(postings_lists):
        if len(pl) == 0:
            bad_indices.append(index)
    for bad_index in bad_indices[::-1]:
        del postings_lists[bad_index]

def refresh_items(items, gens, minimum):
    """Refresh the items list by replacing the minimum item with the next item from the corresponding generator.
    Args:
        items: a list of tuples (doc_id, score)
        gens: a list of generators
        minimum: the minimum doc_id in the items list
    """
    for index, item in enumerate(items):
        if item[0] == minimum:
            try:
                items[index] = next(gens[index])
            except StopIteration:
                items[index] = (float('inf'), 0)


def reduce_by_key(postings_lists):
    """Reduce a list of postings lists to a single postings list by summing the socre values of the same doc_id.
    Args:
        postings_lists: a list of postings lists, each postings list is a list of tuples (doc_id, score)
    Returns:
        a single postings list
    """
    remove_empty_postings(postings_lists)
    gens = [(item for item in pl) for pl in postings_lists]
    items = [next(gen) for gen in gens]

    combined = []
    while any(score != 0 for _, score in items) and len(postings_lists) > 0:
        doc_id, _ = min(items)
        lst = [(_doc_id, _score) for _doc_id, _score in items if _doc_id == doc_id]
        combined.append((doc_id, sum(score for _, score in lst)))

        refresh_items(items, gens, doc_id)
    return combined

def calculate_bm25_per_posting_list(token_pl, bm25_func, inverted):
    """Calculate the BM25 score for a posting list.
    Args:
        token_pl: a dictionary of term:pl posting list
    Returns:
        a list of posting lists
    """
    bm25_per_doc = []
    for token in token_pl.keys():
        curr_term = []
        for doc_id, tf in token_pl[token]:
            curr_term.append((doc_id, bm25_func(token, (doc_id, tf), inverted)))
        bm25_per_doc.append(curr_term)
    return bm25_per_doc
    

def query(query: str, bucket_name: str, inverted_title, inverted_text, pagerank_normalized):
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
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [token for token in tokens if token not in all_stopwords]
    print('pl')
    token_pl_title = {token: retrieve_posting_list(token, bucket_name, inverted_title) for token in tokens}
    token_pl_text = {token: retrieve_posting_list(token, bucket_name, inverted_text) for token in tokens}

    print('bm25 title')
    bm25_per_doc_title = calculate_bm25_per_posting_list(token_pl_title, calculate_bm25_per_title_term, inverted_title)

    print('bm25 text')
    bm25_per_doc_text = calculate_bm25_per_posting_list(token_pl_text, calculate_bm25_per_text_term, inverted_text)

    print('reduce title')
    bm25_title = reduce_by_key(bm25_per_doc_title)
    print('reduce text')
    bm25_title = list(map(lambda x: (x[0], x[1]), bm25_title))
    bm25_text = reduce_by_key(bm25_per_doc_text)
    print('reduce together')
    # bm25_text = list(map(lambda x: (x[0], x[1]), bm25_text))
    bm25 = reduce_by_key([bm25_title, bm25_text])
    bm25 = list(map(lambda x: (x[0], sqrt(pagerank_normalized.get(x[0],0.2) + 1) * (x[1] ** 3)), bm25))

    return sorted(bm25, key=lambda x: x[1], reverse=True)


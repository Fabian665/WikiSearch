import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path, PurePosixPath
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
import numpy as np
from math import log

PROJECT_ID = 'YOUR-PROJECT-ID-HERE'
def get_bucket(bucket_name):
    return storage.Client(PROJECT_ID).bucket(bucket_name)

def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)

# Let's start with a small block size of 30 bytes just to test things out. 
BLOCK_SIZE = 1999998

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (_open(str(self._base_dir / f'{name}_{i:03}.bin'), 
                                'wb', self._bucket) 
                          for i in itertools.count())
        self._f = next(self._file_gen)
           
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:  
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, 'name') else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = PurePosixPath(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            f_name = str(self._base_dir / f_name)
            if f_name not in self._open_files:
                self._open_files[f_name] = _open(f_name, 'rb', self._bucket)
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)
  
    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False 

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
    def __init__(self, docs={}, page_rank=None):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores inverse document frequency per term
        self.idf = defaultdict(float)
        self.idf_bm25 = defaultdict(float)
        # stores PageRank score per document
        self.pagerank = defaultdict(float) if page_rank is None else page_rank
        # stores normalized PageRank score per document
        self.pagerank_normalized = defaultdict(float) if page_rank is None else self.get_normalized_page_rank()
        # stores posting list per term while building the index (internally), 
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # stores document length per document
        self.doc_len = defaultdict(int)
        # mapping a term to posting file locations, which is a list of 
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are 
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents 
        # the number of bytes from the beginning of the file where the posting list
        # starts. 
        self.posting_locs = defaultdict(list)
        self.corpus_size = 0
        self.avdl = 0

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage 
            side-effects).
        """
        raise NotImplementedError
        self.__corpus_size += 1
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        self.doc_len[doc_id] = sum(w2cnt.values())
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def get_idf(self):
        """ Calculate the inverse document frequency of each term in the index.
        """
        return {w: log(self.corpus_size / df) for w, df in self.df.items()}
    
    def get_idf_bm25(self):
        """ Calculate the inverse document frequency of each term in the index.
        """
        return {w: log((self.corpus_size + 1)/ df) for w, df in self.df.items()}

    def set_idf(self):
        self.idf = self.get_idf()
        
    def set_idf_bm25(self):
        self.idf_bm25 = self.get_idf_bm25()

    def get_normalized_term_freq(self, w, doc_id):
        """ Returns the normalized term frequency of term `w` in document `doc_id`.
        """
        return self._posting_list[w][doc_id] / self.doc_len[doc_id]
    
    def get_normalized_page_rank(self):
        """ Returns the normalized page rank of all of the documents
        """
        scores = list(self.pagerank.values())
        robust_measure = np.percentile(scores, 99.995)
        return {doc_id: min(score / robust_measure, 1.0) for doc_id, score in self.pagerank.items()}

    def write_index(self, base_dir, name, bucket_name=None):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        if state.get('_posting_list', None) is not None:
            del state['_posting_list']
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def read_a_posting_list(self, base_dir, w, bucket_name=None):
        posting_list = []
        if not w in self.posting_locs:
            return posting_list
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list
    
    def get_tf_in_doc(self, w, doc_id, bucket_name=None):
        data = self.read_a_posting_list(base_dif, w, bucket_name)
        low = 0
        high = len(data)
        
        while low <= high:
            mid = (low + high) // 2
            if data[mid][0] == target_doc_id:
                return data[mid]
            elif data[mid][0] < target_doc_id:
                low = mid + 1
            else:
                high = mid - 1

        return None  
            
    

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        
        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl: 
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)
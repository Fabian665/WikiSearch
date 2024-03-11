import pickle
from typing import Generator, List, Any

from google.cloud import storage
from Types import RankedPosting, RankedPostingList


def load_pickle(filename: str, bucket_name: str, project_name: str) -> Any:
    """
    Load a pickled object from a Google Cloud Storage bucket.

    This function retrieves a pickled object from a specified Google Cloud Storage bucket and returns it.
    The object is expected to be located in the 'pickles' directory of the bucket.

    Args:
        filename (str): The name of the file to be loaded.
        bucket_name (str): The name of the Google Cloud Storage bucket where the file is located.
        project_name (str): The name of the Google Cloud project associated with the bucket.

    Returns:
        The unpickled object that was stored in the file.
    """
    file_path = f"pickles/{filename}"
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    with bucket.blob(file_path).open('rb') as file:
        return pickle.loads(file.read())


def refresh_items(items: List[RankedPosting], gens: List[Generator[RankedPosting, Any, None]], minimum: int) -> None:
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


def remove_empty_postings(postings_lists: List[RankedPostingList]) -> None:
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


def reduce_by_key(postings_lists: List[RankedPostingList]) -> RankedPostingList:
    """Reduce a list of postings lists to a single postings list by summing the score values of the same doc_id.
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

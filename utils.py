import pickle
from google.cloud import storage


def load_pickle(filename, bucket_name, project_name):
    file_path = f"pickles/{filename}"
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    with bucket.blob(file_path).open('rb') as file:
        return pickle.loads(file.read())


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

from typing import List, Tuple, TypeAlias


DocId = int
Tf = int
Score = float

Posting: TypeAlias = Tuple[DocId, Tf]
RankedPosting: TypeAlias = Tuple[DocId, Score]

PostingList: TypeAlias = List[Posting]
RankedPostingList: TypeAlias = List[RankedPosting]
Tokens = List[str]

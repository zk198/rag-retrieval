def reciprocal_rank(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def test_reciprocal_rank_decreases():
    assert reciprocal_rank(1) > reciprocal_rank(2)

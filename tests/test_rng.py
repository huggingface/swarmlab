import random

from swarmlab.rng import derive


def draws(r: random.Random, n: int = 8) -> list[float]:
    return [r.random() for _ in range(n)]


def test_same_labels_same_stream():
    assert draws(derive(3, "agent", "a001")) == draws(derive(3, "agent", "a001"))


def test_different_seed_or_label_differs():
    base = draws(derive(3, "world"))
    assert draws(derive(4, "world")) != base
    assert draws(derive(3, "schedule", 1)) != draws(derive(3, "schedule", 2))
    assert draws(derive(3, "agent", "a000")) != draws(derive(3, "agent", "a001"))
    assert draws(derive(3, "private", "a000")) != draws(derive(3, "agent", "a000"))


def test_label_boundaries_matter():
    assert draws(derive(1, "ab")) != draws(derive(1, "a", "b"))
    assert draws(derive(1, "a", "b")) != draws(derive(1, "ab", ""))


def test_streams_are_independent():
    # consuming one stream heavily does not shift another
    expected = draws(derive(7, "topology", 3))
    w = derive(7, "world")
    for _ in range(1000):
        w.random()
    assert draws(derive(7, "topology", 3)) == expected


def test_global_random_untouched():
    random.seed(123)
    state = random.getstate()
    derive(1, "world").random()
    assert random.getstate() == state

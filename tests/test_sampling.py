from supahmlx.generation.sampling import Sampler, SamplingConfig


def test_sampling_seeded_determinism():
    logits = [0.1, 2.0, 0.5, -1.0]
    s1 = Sampler(seed=42)
    s2 = Sampler(seed=42)
    cfg = SamplingConfig(temperature=0.9, top_p=0.95, top_k=3)
    assert [s1.sample(logits, cfg) for _ in range(5)] == [s2.sample(logits, cfg) for _ in range(5)]


def test_sampling_greedy_and_filters():
    logits = [0.0, 1.0, 2.0, 0.5]
    greedy = Sampler(seed=1).sample(logits, SamplingConfig(temperature=0, greedy=True))
    assert greedy == 2
    chosen = Sampler(seed=0).sample(logits, SamplingConfig(temperature=1.0, top_k=1, top_p=1.0))
    assert chosen == 2

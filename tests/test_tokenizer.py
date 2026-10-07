from supahmlx.tokenizer.simple import SimpleTokenizer


def test_tokenizer_round_trip_normalized_whitespace():
    tokenizer = SimpleTokenizer()
    text = "hello   supahmlx world"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == "hello supahmlx world"

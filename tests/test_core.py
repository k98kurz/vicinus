from vicinus import (
    tokenize, rank, select
)
import unittest


class TestCore(unittest.TestCase):
    original = "This. Is. A. Test. Get that through your thick head, numb-skull!"
    def test_tokenize_returns_stripped_words(self):
        tokens = tokenize(self.original)
        assert type(tokens) is list, type(tokens)
        assert all([t.islower() for t in tokens]), tokens
        assert all([t.isalnum() for t in tokens]), tokens

    def test_rank_returns_sorted_list_of_tuples(self):
        fn = lambda x: (len(x) + len(self.original))/len(self.original)
        texts = {
            0: "Shortest",
            1: "A medium length text",
            2: "This is the longest text by far.",
        }
        rankings = rank(fn, texts)
        assert type(rankings) is list, type(rankings)
        assert all([type(t) is tuple for t in rankings]), [type(t) for t in rankings]
        sorted_rankings = sorted(rankings, reverse=True)
        assert rankings == sorted_rankings, (rankings, sorted_rankings)
        assert rankings[0][1] == 2, rankings

    def test_select_returns_top_k(self):
        fn = lambda x: (len(x) + len(self.original))/len(self.original)
        texts = {
            0: "Shortest",
            1: "A medium length text",
            2: "This is the longest text by far.",
        }
        rankings = rank(fn, texts)
        s1 = select(rankings, 1)
        assert type(s1) is list, type(s1)
        assert len(s1) == 1, s1
        assert s1[0][0] == max([t[0] for t in rankings])
        s2 = select(rankings, 2)
        assert type(s2) is list, type(s2)
        assert len(s2) == 2, s2


if __name__ == '__main__':
    unittest.main()

from fixture import DocDB
from vicinus import (
    hamming_distance, hamming_similarity,
    levenshtein_distance, levenshtein_similarity
)
import unittest


class TestHammingLev(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docdb = DocDB()

    def test_hamming_distance_returns_int_that_counts_differences(self):
        hd = hamming_distance(self.docdb.get(0).title, self.docdb.get(1).title)
        assert type(hd) is int, type(hd)
        assert hamming_distance("hello world", "hxllo world") == 1
        assert hamming_distance("hello world", "hxxlo world") == 2
        assert hamming_distance("hello world", "hxxlo worl") == 3
        assert hamming_distance("hello world", "ello world") == 10

    def test_hamming_distance_normalized_returns_float(self):
        hd = hamming_distance(
            self.docdb.get(0).title, self.docdb.get(1).title, True
        )
        assert type(hd) is float, type(hd)
        hd = hamming_distance("test", "", True)
        assert hd == 1.0, hd

    def test_hamming_similarity_returns_float_that_is_1_for_same(self):
        hs = hamming_similarity(self.docdb.get(0).title, self.docdb.get(1).title)
        assert type(hs) is float, type(hs)
        assert hamming_similarity("test", "test") == 1.0

    def test_levenshtein_distance_returns_int_that_counts_differences(self):
        ld = levenshtein_distance(self.docdb.get(0).title, self.docdb.get(1).title)
        assert type(ld) is int, type(ld)
        assert levenshtein_distance("hello world", "hxllo world") == 1
        assert levenshtein_distance("hello world", "hxxlo world") == 2
        assert levenshtein_distance("hello world", "hxxlo worl") == 3
        assert levenshtein_distance("hello world", "ello world") == 1

    def test_levenshtein_distance_normalized_returns_float(self):
        ld = levenshtein_distance(
            self.docdb.get(0).title, self.docdb.get(1).title, True
        )
        assert type(ld) is float, type(ld)
        ld = levenshtein_distance("test", "", True)
        assert ld == 1.0, ld

    def test_levenshtein_similarity_returns_float_is_1_for_same(self):
        ls = levenshtein_similarity(
            self.docdb.get(0).title, self.docdb.get(1).title
        )
        assert type(ls) is float, type(ls)
        assert levenshtein_similarity("test", "test") == 1.0


if __name__ == '__main__':
    unittest.main()

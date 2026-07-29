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

    def test_hamming_distance_returns_int(self):
        hd = hamming_distance(self.docdb.get(0).title, self.docdb.get(1).title)
        assert type(hd) is int, type(hd)

    def test_hamming_distance_normalized_returns_float(self):
        hd = hamming_distance(
            self.docdb.get(0).title, self.docdb.get(1).title, True
        )
        assert type(hd) is float, type(hd)

    def test_hamming_similarity_returns_float(self):
        hs = hamming_similarity(self.docdb.get(0).title, self.docdb.get(1).title)
        assert type(hs) is float, type(hs)

    def test_levenshtein_distance_returns_int(self):
        ld = levenshtein_distance(self.docdb.get(0).title, self.docdb.get(1).title)
        assert type(ld) is int, type(ld)

    def test_levenshtein_distance_normalized_returns_float(self):
        ld = levenshtein_distance(
            self.docdb.get(0).title, self.docdb.get(1).title, True
        )
        assert type(ld) is float, type(ld)

    def test_levenshtein_similarity_returns_float(self):
        ls = levenshtein_similarity(
            self.docdb.get(0).title, self.docdb.get(1).title
        )
        assert type(ls) is float, type(ls)


if __name__ == '__main__':
    unittest.main()

from fixture import DocDB
from vicinus import (
    n_grams, jaccard_index
)
import unittest


class TestNGJaccard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docdb = DocDB()

    def test_n_grams_returns_set_of_str_with_correct_size(self):
        ngs = n_grams(self.docdb.get(0).text)
        assert type(ngs) is set, type(ngs)
        assert all([len(n) == 3 for n in ngs]), ngs

        ngs = n_grams(self.docdb.get(0).text, N=2)
        assert type(ngs) is set, type(ngs)
        assert all([len(n) == 2 for n in ngs]), ngs

    def test_n_grams_are_all_alphanumeric(self):
        ngs = n_grams(self.docdb.get(0).text)
        assert all([n.isalnum() for n in ngs]), ngs

    def test_jaccard_index_returns_float_between_0_and_1(self):
        ngs0 = n_grams(self.docdb.get(0).text)
        ngs1 = n_grams(self.docdb.get(1).text)

        ji00 = jaccard_index(ngs0, ngs0)
        assert type(ji00) is float, type(ji00)
        # should be 100% similar
        assert ji00 == 1.0, ji00

        ji01 = jaccard_index(ngs0, ngs1)
        assert type(ji01) is float, type(ji01)
        # should be less than 100% similar
        assert 0.0 <= ji01 < 1.0, ji01


if __name__ == '__main__':
    unittest.main()

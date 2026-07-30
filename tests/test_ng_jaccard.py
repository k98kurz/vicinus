from fixture import DocDB
from vicinus import (
    n_grams, jaccard_index, rank,
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

    def test_higher_N_improves_jaccard_index_accuracy(self):
        query = "spiritual disease illness"
        ngs2 = {i: n_grams(d.text, N=2) for i, d in enumerate(self.docdb.docs)}
        qng2 = n_grams(query, N=2)
        r2 = rank(lambda ng: jaccard_index(qng2, ng), ngs2)
        d2 = [self.docdb.get(r[1]).title for r in r2[:2]]
        has_spirits = 'spirits-cause-disease.txt' in d2
        has_cold_flu = 'common-cold-vs-flu.txt' in d2
        assert not (has_spirits and has_cold_flu)

        ngs3 = {i: n_grams(d.text, N=3) for i, d in enumerate(self.docdb.docs)}
        qng3 = n_grams(query, N=3)
        r3 = rank(lambda ng: jaccard_index(qng3, ng), ngs3)
        d3 = [self.docdb.get(r[1]).title for r in r3[:2]]
        has_spirits = 'spirits-cause-disease.txt' in d3
        has_cold_flu = 'common-cold-vs-flu.txt' in d3
        assert not (has_spirits and has_cold_flu)

        ngs5 = {i: n_grams(d.text, N=5) for i, d in enumerate(self.docdb.docs)}
        qng5 = n_grams(query, N=5)
        r5 = rank(lambda ng: jaccard_index(qng5, ng), ngs5)
        d5 = [self.docdb.get(r[1]).title for r in r5[:2]]
        has_spirits = 'spirits-cause-disease.txt' in d5
        has_cold_flu = 'common-cold-vs-flu.txt' in d5
        assert has_spirits and has_cold_flu


if __name__ == '__main__':
    unittest.main()

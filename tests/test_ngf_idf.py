from fixture import DocDB
from vicinus import (
    ngf_index, ng_count, ngf, ngf_rank, ngf_select,
    ngf_idf_setup, ngf_idf_query, ngf_idf_rank, ngf_idf_select,
    SparseVector,
)
import unittest


class TestNGFIDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docdb = DocDB()

    def test_ngf_index_returns_int(self):
        assert type(ngf_index('0')) is int, ngf_index('0')
        assert ngf_index('0') == 0, ngf_index('0')
        assert ngf_index('z') == 35, ngf_index('z')

    def test_ng_count_returns_SparseVector_of_ints(self):
        vec = ng_count(self.docdb.get(0).text)
        assert isinstance(vec, SparseVector), type(vec)
        assert all([type(v) is int for v in vec.values()])

    def test_ngf_returns_SparseVector_of_floats(self):
        doc = self.docdb.get(0)
        vec = ngf(doc.text)
        assert isinstance(vec, SparseVector)
        assert all([type(v) is float for v in vec.values()])

    def test_ngf_cosine_similarity_between_0_and_1(self):
        vec0 = ngf(self.docdb.get(0).text)
        vec1 = ngf(self.docdb.get(1).text)
        sim = vec0.cosine_similarity(vec0)
        assert sim > 0.99999, sim # floating point occasionally just under 1.0
        assert 0.0 < vec0.cosine_similarity(vec1) < 1.0

    def test_ngf_rank_returns_list_tuple_float_int(self):
        vecs = [ngf(d.text) for d in self.docdb.docs]
        query = "spiritual disease"
        rankings = ngf_rank(query, vecs)
        assert type(rankings) is list, type(rankings)
        assert all([type(r) is tuple for r in rankings])
        assert all([len(r) == 2 for r in rankings])
        assert all([type(r[0]) is float for r in rankings]), 'score should be first'
        assert all([type(r[1]) is int for r in rankings]), 'index should be second'

    def test_ngf_rank_returns_sorted_by_score_descending(self):
        vecs = [ngf(d.text) for d in self.docdb.docs]
        query = "spiritual disease"
        rankings = ngf_rank(query, vecs)
        for i in range(len(rankings)-1):
            assert rankings[i][0] >= rankings[i+1][0]

    def test_ngf_select_returns_top_k_of_ngf_rank(self):
        vecs = [ngf(d.text) for d in self.docdb.docs]
        query = "spiritual disease"
        k = 3
        rankings = ngf_rank(query, vecs)
        rank_ids = [r[1] for r in rankings]
        selection = ngf_select(query, vecs, k=k)
        select_ids = [s[1] for s in selection]
        assert select_ids == rank_ids[:k], (rankings, selection)

    def test_ngf_rank_scores_decrease_and_become_more_accurate_with_n_gram_size(self):
        vecs = [ngf(d.text) for d in self.docdb.docs]
        query = "spiritual disease"
        vecs1 = [ngf(d.text, 2) for d in self.docdb.docs]
        vecs2 = [ngf(d.text, 5) for d in self.docdb.docs]
        query = "spiritual disease psychic illness"
        rankings1 = ngf_rank(query, vecs1, 2)
        rankings2 = ngf_rank(query, vecs2, 5)

        # decrease total scores
        total_score1 = sum([r[0] for r in rankings1])
        total_score2 = sum([r[0] for r in rankings2])
        assert total_score2 < total_score1, (total_score2, total_score1)

        # increase accuracy
        top_r1 = self.docdb.get(rankings1[0][1]).title
        assert 'spirit' not in top_r1, top_r1

        top_r2 = self.docdb.get(rankings2[0][1]).title
        assert 'spirit' in top_r2, top_r2

    def test_ngf_idf_setup_returns_tuple_SparseVector_list_float_SparseVectors(self):
        corpus = [d.text for d in self.docdb.docs]
        result = ngf_idf_setup(corpus)
        assert type(result) is tuple, type(result)
        assert len(result) == 2, len(result)
        idf, vecs = result
        assert type(idf) is SparseVector, type(idf)
        assert type(vecs) is list, type(vecs)
        assert all([type(v) is SparseVector for v in vecs])

    def test_ngf_idf_query_returns_float_SparseVector(self):
        corpus = [d.text for d in self.docdb.docs]
        idf, _vecs = ngf_idf_setup(corpus)
        query = "spiritual disease psychic illness"
        result = ngf_idf_query(query, idf)
        assert type(result) is SparseVector, type(result)
        assert all([type(v) is float for v in result.values()])

    def test_ngf_idf_rank_returns_list_tuple_float_int(self):
        corpus = [d.text for d in self.docdb.docs]
        idf, vecs = ngf_idf_setup(corpus)
        query = "spiritual disease psychic illness"
        rankings = ngf_idf_rank(query, idf, vecs)
        assert type(rankings) is list, type(rankings)
        assert all([type(r) is tuple for r in rankings])
        assert all([len(r) == 2 for r in rankings])
        assert all([type(r[0]) is float for r in rankings]), 'score should be first'
        assert all([type(r[1]) is int for r in rankings]), 'index should be second'

    def test_ngf_idf_rank_returns_sorted_by_score_descending(self):
        corpus = [d.text for d in self.docdb.docs]
        idf, vecs = ngf_idf_setup(corpus)
        query = "spiritual disease psychic illness"
        rankings = ngf_idf_rank(query, idf, vecs)
        for i in range(len(rankings)-1):
            assert rankings[i][0] >= rankings[i+1][0]

    def test_ngf_idf_select_returns_top_k_of_ngf_rank(self):
        corpus = [d.text for d in self.docdb.docs]
        idf, vecs = ngf_idf_setup(corpus)
        query = "spiritual disease psychic illness"
        k = 3
        rankings = ngf_idf_rank(query, idf, vecs)
        rank_ids = [r[1] for r in rankings]
        selection = ngf_idf_select(query, idf, vecs, k=k)
        select_ids = [s[1] for s in selection]
        assert select_ids == rank_ids[:k], (rankings, selection)


if __name__ == '__main__':
    unittest.main()

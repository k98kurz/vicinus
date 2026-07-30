from fixture import DocDB
from time import perf_counter
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
        cls.corpus = {i: d.text for i, d in enumerate(cls.docdb.docs)}
        cls.ngf_vecs = {i: ngf(d.text) for i, d in enumerate(cls.docdb.docs)}

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
        query = "spiritual disease"
        rankings = ngf_rank(query, self.ngf_vecs)
        assert type(rankings) is list, type(rankings)
        assert all([type(r) is tuple for r in rankings])
        assert all([len(r) == 2 for r in rankings])
        assert all([type(r[0]) is float for r in rankings]), 'score should be first'
        assert all([type(r[1]) is int for r in rankings]), 'index should be second'

    def test_ngf_rank_returns_sorted_by_score_descending(self):
        query = "spiritual disease"
        rankings = ngf_rank(query, self.ngf_vecs)
        for i in range(len(rankings)-1):
            assert rankings[i][0] >= rankings[i+1][0]

    def test_ngf_select_returns_top_k_of_ngf_rank(self):
        query = "spiritual disease"
        k = 3
        rankings = ngf_rank(query, self.ngf_vecs)
        rank_ids = [r[1] for r in rankings]
        selection = ngf_select(query, self.ngf_vecs, k=k)
        select_ids = [s[1] for s in selection]
        assert select_ids == rank_ids[:k], (rankings, selection)

    def test_ngf_rank_scores_decrease_and_become_more_accurate_with_n_gram_size(self):
        vecs1 = {i: ngf(d.text, 2) for i, d in enumerate(self.docdb.docs)}
        vecs2 = {i: ngf(d.text, 5) for i, d in enumerate(self.docdb.docs)}
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
        result = ngf_idf_setup(self.corpus)
        assert type(result) is tuple, type(result)
        assert len(result) == 3, len(result)
        idf, vecs, ngcs = result
        assert type(idf) is SparseVector, type(idf)
        assert type(vecs) is dict, type(vecs)
        assert all([type(v) is SparseVector for v in vecs.values()])
        assert type(ngcs) is dict, type(ngcs)
        assert all([type(v) is SparseVector for v in ngcs.values()])

    def test_ngf_idf_query_returns_float_SparseVector(self):
        idf, _vecs, _ngcs = ngf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        result = ngf_idf_query(query, idf)
        assert type(result) is SparseVector, type(result)
        assert all([type(v) is float for v in result.values()])

    def test_ngf_idf_rank_returns_list_tuple_float_int(self):
        idf, vecs, _ = ngf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        rankings = ngf_idf_rank(query, idf, vecs)
        assert type(rankings) is list, type(rankings)
        assert all([type(r) is tuple for r in rankings])
        assert all([len(r) == 2 for r in rankings])
        assert all([type(r[0]) is float for r in rankings]), 'score should be first'
        assert all([type(r[1]) is int for r in rankings]), 'index should be second'

    def test_ngf_idf_rank_returns_sorted_by_score_descending(self):
        idf, vecs, _ = ngf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        rankings = ngf_idf_rank(query, idf, vecs)
        for i in range(len(rankings)-1):
            assert rankings[i][0] >= rankings[i+1][0]

    def test_ngf_idf_select_returns_top_k_of_ngf_rank(self):
        idf, vecs, _ = ngf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        k = 3
        rankings = ngf_idf_rank(query, idf, vecs)
        rank_ids = [r[1] for r in rankings]
        selection = ngf_idf_select(query, idf, vecs, k=k)
        select_ids = [s[1] for s in selection]
        assert select_ids == rank_ids[:k], (rankings, selection)

    def test_ngf_idf_rank_scores_decrease_and_become_more_accurate_with_n_gram_size(self):
        idf1, vecs1, _ = ngf_idf_setup(self.corpus, N=2)
        idf2, vecs2, _ = ngf_idf_setup(self.corpus, N=5)
        query = "spiritual disease psychic illness"
        rankings1 = ngf_idf_rank(query, idf1, vecs1, 2)
        rankings2 = ngf_idf_rank(query, idf2, vecs2, 5)

        # decrease total scores
        total_score1 = sum([r[0] for r in rankings1])
        total_score2 = sum([r[0] for r in rankings2])
        assert total_score2 < total_score1, (total_score2, total_score1)

        # increase accuracy
        top_r1 = self.docdb.get(rankings1[0][1]).title
        assert 'spirit' not in top_r1, top_r1

        top_r2 = self.docdb.get(rankings2[0][1]).title
        assert 'spirit' in top_r2, top_r2

    def test_ngf_idf_setup_is_faster_when_reusing_ngcounts(self):
        without = []
        for i in range(3):
            start = perf_counter()
            _i, _v, ngcs = ngf_idf_setup(self.corpus)
            stop = perf_counter()
            without.append(stop - start)

        with_ngcs = []
        for i in range(3):
            start = perf_counter()
            _i, _v, _n = ngf_idf_setup(self.corpus, ng_counts=ngcs)
            stop = perf_counter()
            with_ngcs.append(stop - start)

        total_without = sum(without)
        total_with = sum(with_ngcs)
        print(f"{total_without=} {total_with=} speed_up={1-total_with/total_without}")
        assert total_without > total_with, (total_without, total_with)


if __name__ == '__main__':
    unittest.main()

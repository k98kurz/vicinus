from fixture import DocDB
from time import perf_counter
from vicinus.ctf_idf import (
    ctf, ctf_rank, ctf_select, SparseVector,
    ctf_idf_setup, ctf_idf_query, ctf_idf_rank, ctf_idf_select,
)
import gc
import sys
import unittest


_stat_test_size = 3

class TestCTFIDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docdb = DocDB()
        cls.corpus = {d.title: d.text for i, d in enumerate(cls.docdb.docs)}
        cls.ctf_vecs = {d.title: ctf(d.text) for i, d in enumerate(cls.docdb.docs)}

    def test_ctf_returns_SparseVector(self):
        vec = ctf(self.docdb.get(0).text)
        assert isinstance(vec, SparseVector), type(vec)

    def test_ctf_cosine_similarity_between_0_and_1(self):
        vec0 = ctf(self.docdb.get(0).text)
        vec1 = ctf(self.docdb.get(1).text)
        sim = vec0.cosine_similarity(vec0)
        assert sim > 0.99999, sim # floating point occasionally just under 1.0
        assert 0.0 < vec0.cosine_similarity(vec1) < 1.0

    def test_ctf_rank_returns_list_tuple_float_int(self):
        query = "agentic ai cyber threat"
        rankings = ctf_rank(query, self.ctf_vecs)
        print(f'rankings ({query}):')
        [print(f"\t{r}") for r in rankings]
        assert type(rankings) is list, type(rankings)
        assert all([type(r) is tuple for r in rankings])
        assert all([len(r) == 2 for r in rankings])
        assert all([type(r[0]) is float for r in rankings]), 'score should be first'
        assert all([type(r[1]) is str for r in rankings]), 'index should be second'

    def test_ctf_rank_returns_sorted_by_score_descending(self):
        query = "spiritual disease"
        rankings = ctf_rank(query, self.ctf_vecs)
        for i in range(len(rankings)-1):
            assert rankings[i][0] >= rankings[i+1][0]

    def test_ctf_select_returns_top_k_of_ctf_rank(self):
        query = "spiritual disease"
        k = 3
        rankings = ctf_rank(query, self.ctf_vecs)
        rank_ids = [r[1] for r in rankings]
        selection = ctf_select(query, self.ctf_vecs, k=k)
        select_ids = [s[1] for s in selection]
        assert select_ids == rank_ids[:k], (rankings, selection)

    def test_ctf_rank_scores_decrease_and_become_more_accurate_with_N(self):
        vecs1 = {d.title: ctf(d.text, 10) for i, d in enumerate(self.docdb.docs)}
        vecs2 = {d.title: ctf(d.text, 500) for i, d in enumerate(self.docdb.docs)}
        query = "spirit disease psychic illness"
        rankings1 = ctf_rank(query, vecs1, 10)
        rankings2 = ctf_rank(query, vecs2, 500)

        # decrease total scores
        total_score1 = sum([r[0] for r in rankings1])
        total_score2 = sum([r[0] for r in rankings2])
        assert total_score2 < total_score1, (total_score2, total_score1)

        # increase accuracy
        top_r1 = rankings1[0][1]
        assert 'spirit' not in top_r1, top_r1

        top_r2 = rankings2[0][1]
        assert 'spirit' in top_r2, top_r2

    def test_ctf_idf_setup_returns_tuple_SparseVector_list_float_SparseVectors(self):
        result = ctf_idf_setup(self.corpus)
        assert type(result) is tuple, type(result)
        assert len(result) == 3, len(result)
        idf, vecs, ctcs = result
        assert type(idf) is SparseVector, type(idf)
        assert type(vecs) is dict, type(vecs)
        assert all([type(v) is SparseVector for v in vecs.values()])
        assert type(ctcs) is dict, type(ctcs)
        assert all([type(v) is SparseVector for v in ctcs.values()])

    def test_ctf_idf_query_returns_float_SparseVector(self):
        idf, _vecs, _ctcs = ctf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        result = ctf_idf_query(query, idf)
        assert type(result) is SparseVector, type(result)
        assert all([type(v) is float for v in result.values()])

    def test_ctf_idf_rank_returns_list_tuple_float_int(self):
        idf, vecs, _ = ctf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        rankings = ctf_idf_rank(query, idf, vecs)
        assert type(rankings) is list, type(rankings)
        assert all([type(r) is tuple for r in rankings])
        assert all([len(r) == 2 for r in rankings])
        assert all([type(r[0]) is float for r in rankings]), 'score should be first'
        assert all([type(r[1]) is str for r in rankings]), 'index should be second'

    def test_ctf_idf_rank_returns_sorted_by_score_descending(self):
        idf, vecs, _ = ctf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        rankings = ctf_idf_rank(query, idf, vecs)
        for i in range(len(rankings)-1):
            assert rankings[i][0] >= rankings[i+1][0]

    def test_ctf_idf_select_returns_top_k_of_ctf_rank(self):
        idf, vecs, _ = ctf_idf_setup(self.corpus)
        query = "spiritual disease psychic illness"
        k = 3
        rankings = ctf_idf_rank(query, idf, vecs)
        rank_ids = [r[1] for r in rankings]
        selection = ctf_idf_select(query, idf, vecs, k=k)
        select_ids = [s[1] for s in selection]
        assert select_ids == rank_ids[:k], (rankings, selection)

    def test_ctf_idf_rank_scores_decrease_and_become_more_accurate_with_N(self):
        idf1, vecs1, _ = ctf_idf_setup(self.corpus, N=100)
        idf2, vecs2, _ = ctf_idf_setup(self.corpus, N=1000)
        query = "spirits disease psychic illness"
        rankings1 = ctf_idf_rank(query, idf1, vecs1, 100)
        rankings2 = ctf_idf_rank(query, idf2, vecs2, 1000)

        # decrease total scores
        total_score1 = sum([r[0] for r in rankings1])
        total_score2 = sum([r[0] for r in rankings2])
        assert total_score2 < total_score1, (total_score2, total_score1)

        # increase accuracy
        top_r1 = rankings1[0][1]
        assert 'spirit' not in top_r1, top_r1

        top_r2 = rankings2[0][1]
        assert 'spirit' in top_r2, top_r2

    def test_ctf_idf_setup_is_faster_when_reusing_ctcounts(self):
        without = []
        for i in range(_stat_test_size):
            gc.collect()
            start = perf_counter()
            _i, _v, ctcs = ctf_idf_setup(self.corpus)
            stop = perf_counter()
            without.append(stop - start)

        with_ctcs = []
        for i in range(_stat_test_size):
            gc.collect()
            start = perf_counter()
            _i, _v, _n = ctf_idf_setup(self.corpus, ct_counts=ctcs)
            stop = perf_counter()
            with_ctcs.append(stop - start)

        total_without = sum(without)
        total_with = sum(with_ctcs)
        avg_without = total_without / len(without)
        avg_with = total_with / len(with_ctcs)
        stdev_without = sum([
            (wo - avg_without)**2
            for wo in without
        ]) ** 0.5
        stdev_with = sum([
            (w - avg_with)**2
            for w in with_ctcs
        ]) ** 0.5
        print(
            f"ctcs: {total_without=} {total_with=} "
            f"speed_up={1-total_with/total_without}"
        )
        print(f"{avg_without=} {stdev_without=}")
        print(f"{avg_with=} {stdev_with=}")
        print(f"sample size: {_stat_test_size}")
        assert total_without > total_with, (total_without, total_with)



if __name__ == '__main__':
    unittest.main()

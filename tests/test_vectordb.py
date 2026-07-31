from fixture import DocDB
from time import perf_counter
from vicinus.core import tokenize
from vicinus.sparse_vector import SparseVector
from vicinus.vectordb import VectorDB, VDBMode, IDFMode
import gc
import unittest


class TestVectorDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docdb = DocDB()

    def test_VectorDB_default_config(self):
        vdb = VectorDB()
        assert vdb.mode is VDBMode.NGF, vdb.mode
        assert vdb.N == 5, vdb.N

    def test_VectorDB_JACCARD_mode_e2e(self):
        doc = self.docdb.get(self.docdb.find('horse-tinder-pitch.txt'))
        vdb = VectorDB(mode=VDBMode.JACCARD)
        assert len(vdb.corpus) == 0, len(vdb.corpus)
        assert len(vdb.vectors) == 0, len(vdb.vectors)

        with self.assertRaises(IndexError):
            vdb.get(doc.title)

        vdb.add(doc.title, doc.text)
        assert len(vdb.corpus) == 1, len(vdb.corpus)
        assert len(vdb.vectors) == 1, len(vdb.vectors)
        assert doc.title in vdb.corpus
        assert doc.title in vdb.vectors
        assert type(vdb.vectors[doc.title]) is set

        assert vdb.idf is None

        gotten = vdb.get(doc.title)
        assert type(gotten) is tuple, type(gotten)
        assert len(gotten) == 2, len(gotten)
        assert type(gotten[0]) is str, type(gotten[0])
        assert gotten[0] == doc.text
        assert type(gotten[1]) is list, type(gotten[1])
        assert len(gotten[1]) == 1, len(gotten[1])
        assert type(gotten[1][0]) is set, type(gotten[1][0])

        start = perf_counter()
        vdb.set_corpus({
            d.title: d.text
            for d in self.docdb.docs
        })
        end = perf_counter()
        assert len(vdb.corpus) > 1, len(vdb.corpus)
        assert len(vdb.vectors) > 1, len(vdb.corpus)
        assert doc.title in vdb.corpus
        assert doc.title in vdb.vectors
        assert type(vdb.vectors[doc.title]) is set
        vdb.get(doc.title)

        l1 = len(vdb.corpus)
        vdb.remove(doc.title)
        l2 = len(vdb.corpus)
        assert l2 < l1
        assert doc.title not in vdb.corpus

        with self.assertRaises(IndexError):
            vdb.get(doc.title)

        vdb.add(doc.title, doc.text)
        query = 'horse tinder dating app companions'
        results = vdb.search(query)
        assert type(results) is list, type(results)
        assert len(results) == 4, len(results)
        assert results[0][1] == doc.title, results
        print(f"JACCARD: {query=}")
        for r in results:
            print(r)
        print(
            f"ratio 1st/2nd: {(results[0][0]/results[1][0]):.2f}; "
            f"setup time: {(end-start):.2f}s"
        )

        times = []
        for d in self.docdb.docs:
            query = d.title + d.text[:100]
            start = perf_counter()
            results = vdb.search(query)
            end = perf_counter()
            assert results[0][1] == d.title, (d.title, results)
            times.append(end-start)

        avgtime = sum(times) / len(times)
        print(f"query time (avg): {avgtime:.4f}s\n")

    def test_VectorDB_NGF_mode_e2e(self):
        doc = self.docdb.get(self.docdb.find('horse-tinder-pitch.txt'))
        vdb = VectorDB(mode=VDBMode.NGF)
        assert len(vdb.corpus) == 0, len(vdb.corpus)
        assert len(vdb.vectors) == 0, len(vdb.vectors)

        with self.assertRaises(IndexError):
            vdb.get(doc.title)

        vdb.add(doc.title, doc.text)
        assert len(vdb.corpus) == 1, len(vdb.corpus)
        assert len(vdb.vectors) == 1, len(vdb.vectors)
        assert doc.title in vdb.corpus
        assert doc.title in vdb.vectors
        assert type(vdb.vectors[doc.title]) is SparseVector

        assert vdb.idf is None

        gotten = vdb.get(doc.title)
        assert type(gotten) is tuple, type(gotten)
        assert len(gotten) == 2, len(gotten)
        assert type(gotten[0]) is str, type(gotten[0])
        assert gotten[0] == doc.text
        assert type(gotten[1]) is list, type(gotten[1])
        assert len(gotten[1]) == 1, len(gotten[1])
        assert type(gotten[1][0]) is SparseVector, type(gotten[1][0])

        start = perf_counter()
        vdb.set_corpus({
            d.title: d.text
            for d in self.docdb.docs
        })
        end = perf_counter()
        assert len(vdb.corpus) > 1, len(vdb.corpus)
        assert len(vdb.vectors) > 1, len(vdb.corpus)
        assert doc.title in vdb.corpus
        assert doc.title in vdb.vectors
        assert type(vdb.vectors[doc.title]) is SparseVector
        vdb.get(doc.title)

        l1 = len(vdb.corpus)
        vdb.remove(doc.title)
        l2 = len(vdb.corpus)
        assert l2 < l1
        assert doc.title not in vdb.corpus

        with self.assertRaises(IndexError):
            vdb.get(doc.title)

        vdb.add(doc.title, doc.text)
        query = 'horse tinder dating app companions'
        results = vdb.search(query)
        assert type(results) is list, type(results)
        assert len(results) == 4, len(results)
        assert results[0][1] == doc.title, results
        print(f"NGF: {query=}")
        for r in results:
            print(r)
        print(
            f"ratio 1st/2nd: {(results[0][0]/results[1][0]):.2f}; "
            f"setup time: {(end-start):.2f}s"
        )

        times = []
        for d in self.docdb.docs:
            query = d.title + d.text[:100]
            start = perf_counter()
            results = vdb.search(query)
            end = perf_counter()
            assert results[0][1] == d.title, (d.title, results)
            times.append(end-start)

        avgtime = sum(times) / len(times)
        print(f"query time (avg): {avgtime:.4f}s\n")

    def test_VectorDB_NGF_IDF_mode_SAVE_SPACE_e2e(self):
        doc = self.docdb.get(self.docdb.find('horse-tinder-pitch.txt'))
        vdb = VectorDB(mode=VDBMode.NGF_IDF)
        assert vdb.idf_mode is IDFMode.SAVE_SPACE, vdb.idf_mode
        assert len(vdb.corpus) == 0, len(vdb.corpus)
        assert len(vdb.vectors) == 0, len(vdb.vectors)

        with self.assertRaises(IndexError):
            vdb.get(doc.title)

        vdb.add(doc.title, doc.text)
        assert len(vdb.corpus) == 1, len(vdb.corpus)
        assert len(vdb.vectors) == 1, len(vdb.vectors)
        assert doc.title in vdb.corpus
        assert doc.title in vdb.vectors
        assert type(vdb.vectors[doc.title]) is SparseVector
        
        assert vdb.idf is not None

        gotten = vdb.get(doc.title)
        assert type(gotten) is tuple, type(gotten)
        assert len(gotten) == 2, len(gotten)
        assert type(gotten[0]) is str, type(gotten[0])
        assert gotten[0] == doc.text
        assert type(gotten[1]) is list, type(gotten[1])
        assert len(gotten[1]) == 1, len(gotten[1])
        assert type(gotten[1][0]) is SparseVector, type(gotten[1][0])

        start = perf_counter()
        vdb.set_corpus({
            d.title: d.text
            for d in self.docdb.docs
        })
        end = perf_counter()
        assert len(vdb.corpus) > 1, len(vdb.corpus)
        assert len(vdb.vectors) > 1, len(vdb.corpus)
        assert doc.title in vdb.corpus
        assert doc.title in vdb.vectors
        assert type(vdb.vectors[doc.title]) is SparseVector
        vdb.get(doc.title)

        l1 = len(vdb.corpus)
        vdb.remove(doc.title)
        l2 = len(vdb.corpus)
        assert l2 < l1
        assert doc.title not in vdb.corpus

        with self.assertRaises(IndexError):
            vdb.get(doc.title)

        vdb.add(doc.title, doc.text)
        query = 'horse tinder dating app companions'
        results = vdb.search(query)
        assert type(results) is list, type(results)
        assert len(results) == 4, len(results)
        assert results[0][1] == doc.title, results
        print(f"NGF_IDF: {query=}")
        for r in results:
            print(r)
        print(
            f"ratio 1st/2nd: {(results[0][0]/results[1][0]):.2f}; "
            f"setup time: {(end-start):.2f}s"
        )

        times = []
        for d in self.docdb.docs:
            query = d.title + d.text[:100]
            start = perf_counter()
            results = vdb.search(query)
            end = perf_counter()
            assert results[0][1] == d.title, (d.title, results)
            times.append(end-start)

        avgtime = sum(times) / len(times)
        print(f"query time (avg): {avgtime:.4f}s\n")

    def test_VectorDB_NGF_IDF_SAVE_COUNTS_has_faster_recalculate(self):
        doc = self.docdb.get(0)
        vdb1 = VectorDB(mode=VDBMode.NGF_IDF, idf_mode=IDFMode.SAVE_SPACE)
        vdb2 = VectorDB(mode=VDBMode.NGF_IDF, idf_mode=IDFMode.SAVE_COUNTS)

        vdb1.set_corpus({
            d.title: d.text
            for d in self.docdb.docs
        })
        vdb2.set_corpus({
            d.title: d.text
            for d in self.docdb.docs
        })
        
        start1 = perf_counter()
        vdb1.remove(doc.title)
        vdb1.add(doc.title, doc.text)
        vdb1.remove(doc.title)
        vdb1.add(doc.title, doc.text)
        end1 = perf_counter()
        diff1 = end1 - start1

        start2 = perf_counter()
        vdb2.remove(doc.title)
        vdb2.add(doc.title, doc.text)
        vdb2.remove(doc.title)
        vdb2.add(doc.title, doc.text)
        end2 = perf_counter()
        diff2 = end2 - start2

        print(f"Recalculate, IDFMode.SAVE_SPACE: {diff1}")
        print(f"Recalculate, IDFMode.SAVE_COUNTS: {diff2}\n")

        assert diff2 < diff1, (diff1, diff2)


if __name__ == '__main__':
    unittest.main()

from typing import Iterator, ItemsView, ValuesView
from vicinus import SparseVector
import unittest


class TestSparseVector(unittest.TestCase):
    def test_SparseVector_raises_TypeError_for_invalid_input(self):
        vec = SparseVector()
        with self.assertRaises(TypeError):
            vec['a'] = 1
        with self.assertRaises(TypeError):
            vec[1] = 'a'
        with self.assertRaises(TypeError):
            vec[1.012] = 1

    def test_SparseVector_can_have_int_or_float_values(self):
        vec = SparseVector()
        vec[1] = 1
        vec[2] = 2.2
        assert vec[1] == 1
        assert vec[2] == 2.2

    def test_SparseVector_len(self):
        vec = SparseVector()
        assert len(vec) == 0
        for i in range(100):
            vec[i] = i+1
            assert len(vec) == i+1

    def test_SparseVector_setting_to_0_removes_index(self):
        vec = SparseVector()
        vec[0] = 1
        vec[1] = 1
        assert len(vec) == 2
        vec[0] = 0
        vec[1] = 0.0
        assert len(vec) == 0

    def test_SparseVector_iter_provides_keys(self):
        vec = SparseVector({i: i*10 for i in range(1, 10)})
        for k in vec:
            assert k // 10 == 0, k
            assert k in vec.keys()

    def test_SparseVector_addition_e2e(self):
        v1 = SparseVector({0: 1, 1: 1})
        v2 = SparseVector({1: 1, 2: 1})
        v3 = v1 + v2
        assert type(v3) is SparseVector, type(v3)
        assert v3 is not v1 and v3 is not v2
        assert v3.keys() == v1.keys() | v2.keys(), (v3.keys(), v1.keys() | v2.keys())
        assert v3[0] == 1, v3
        assert v3[1] == 2, v3
        assert v3[0] == 1, v3

    def test_SparseVector_keys_values_items(self):
        vec = SparseVector({i: i*10 for i in range(1, 10)})
        assert isinstance(vec.keys(), set), type(vec.keys())
        assert isinstance(vec.values(), ValuesView), type(vec.keys())
        assert isinstance(vec.items(), ItemsView), type(vec.keys())

    def test_SparseVector_norm_returns_float(self):
        vec = SparseVector({i: i*10 for i in range(1, 10)})
        assert type(vec.norm()) is float, vec.norm()

    def test_SparseVector_dot_product_returns_float(self):
        vec = SparseVector({i: i/10 for i in range(1, 10)})
        dp = vec.dot_product(vec)
        assert type(dp) is float, dp

    def test_SparseVector_cosine_similarity_returns_float(self):
        vec = SparseVector({i: i/10 for i in range(1, 10)})
        cs = vec.cosine_similarity(vec)
        assert type(cs) is float, cs

    def test_SparseVector_cosine_similarity_returns_1_for_same(self):
        vec = SparseVector({i: i/10 for i in range(1, 10)})
        cs = vec.cosine_similarity(vec)
        assert cs >= 0.99999, cs

    def test_SparseVector_cosine_similarity_returns_neg1_for_opposite(self):
        vec1 = SparseVector({0: 1, 1: 0})
        vec2 = SparseVector({0: -1, 1: 0})
        cs = vec1.cosine_similarity(vec2)
        assert cs <= -0.99999, cs

    def test_SparseVector_returns_0_for_unset_index(self):
        vec = SparseVector()
        assert vec[123] == 0
        assert vec.get(123) == 0

    def test_SparseVector_copy_returns_copy(self):
        sp1 = SparseVector({1: 2})
        sp2 = sp1.copy()
        assert sp1[1] == sp2[1]
        sp2[2] = 3
        assert sp1[2] == 0


if __name__ == '__main__':
    unittest.main()

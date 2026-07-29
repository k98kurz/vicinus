from vicinus.hamming import hamming_distance, hamming_similarity
from vicinus.helpers import rank, select
from vicinus.levenshtein import levenshtein_distance, levenshtein_similarity
from vicinus.ngf_idf import (
    ngf_index, ng_count, ngf, ngf_rank, ngf_select,
    ngf_idf_setup, ngf_idf_query, ngf_idf_rank, ngf_idf_select,
)
from vicinus.ngrams import n_grams, jaccard_index
from vicinus.samples import list_samples, get_sample
from vicinus.sparse_vector import SparseVector
from vicinus.tokenize import tokenize

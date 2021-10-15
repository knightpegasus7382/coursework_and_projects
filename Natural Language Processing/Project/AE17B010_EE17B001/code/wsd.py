import json
import numpy as np
from collections import Counter
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy


t1 = time.time()
#region
docs_file = open("output/" + "stopword_removed_docs.txt", 'r')
docs_json = json.load(docs_file)
docs_json = docs_json[:600]

queries_file = open("output/" + "stopword_removed_queries.txt", 'r')
queries_json = json.load(queries_file)

docs_file.close()
queries_file.close()

docs_file1 = open("output/" + "stopword_removed_docs.txt", 'r')
queries_file1 = open("output/" + "stopword_removed_queries.txt", 'r')
out_docs_json = json.load(docs_file1)#[:]
out_queries_json = json.load(queries_file1)#[:]

docs_file1.close()
queries_file1.close()
#endregion


word_tokens = [
    token for doc in docs_json for sentence in doc for token in sentence]
word_tokens.extend([token for query in queries_json for sentence in query for token in sentence])
word_types_counts = Counter(word_tokens)

word_types = list(word_types_counts.keys())
type_freqs = list(word_types_counts.values())

disambiguable_types = [typ for (i, typ) in enumerate(
    word_types) if type_freqs[i] > 10]

# region
docs_unseg = []
for doc in docs_json:
    docs_unseg.append([token for sentence in doc for token in sentence])

assoc_vecs = np.zeros((len(word_types), len(word_types)))
token_context_counts_cij = {}

for wtype in word_types:
    token_context_counts_cij[wtype] = {'context_vecs':[], 'doc_nums':[]}

for i, doc in enumerate(docs_unseg):
    context = doc

    types_and_counts = Counter(context)

    present_types = types_and_counts.keys()
    token_counts = types_and_counts.values()

    idxs = []
    for type_k in present_types:
        idx = word_types.index(type_k)
        idxs.append(idx)

    present_types = list(present_types)
    token_counts = np.array(list(token_counts))

    counts_vec_from_doc = np.zeros(len(word_types))
    counts_vec_from_doc[idxs] = token_counts

    delta_iks = counts_vec_from_doc.copy()

    for tok in set(doc):
        counts_vec_of_token = counts_vec_from_doc.copy()
        type_index = idxs[present_types.index(tok)]
        counts_vec_of_token[type_index] -= 1
        token_context_counts_cij[tok]['context_vecs'].append(counts_vec_of_token)
        token_context_counts_cij[tok]['doc_nums'].append(i)
        assoc_vecs[type_index] += delta_iks[type_index]*counts_vec_of_token
        del counts_vec_of_token
    if i % 100 == 0:
        print("doc", i, "done")

t2 = time.time()

print(assoc_vecs, assoc_vecs.size, np.count_nonzero(assoc_vecs))

t3 = time.time()
# endregion

assoc_vecs = csr_matrix(assoc_vecs)

cluster_vectors = {}

actual_disambiguated_types = []

print(len(docs_json), len(docs_unseg), len(word_tokens), len(word_types), len(disambiguable_types))



for type_no, wtype in enumerate(disambiguable_types):

    if type_no % 100 == 0:
        print(type_no)

    type_vecs_nums = token_context_counts_cij[wtype]
    type_doc_nums = type_vecs_nums['doc_nums']

    if len(type_doc_nums) <= 5:
        continue

    actual_disambiguated_types.append(wtype)

    token_context_vecs = type_vecs_nums['context_vecs']

    sparse_token_context_vecs = csr_matrix(token_context_vecs)
    wtype_vecs = (sparse_token_context_vecs*assoc_vecs).toarray()

    Z = hierarchy.linkage(wtype_vecs, method='average', metric='cosine')
    max_cophenet_distance = hierarchy.maxdists(Z)[-1]

    threshold_height = 0.95
    C = hierarchy.fcluster(Z, t=threshold_height *
                           max_cophenet_distance, criterion='distance')

    cluster_vectors[wtype] = {}
    for cluster in range(1, max(C)+1):
        cluster_vectors[wtype][cluster] = wtype_vecs[C==cluster]
    
    del wtype_vecs
    del Z
    del token_context_vecs
    del sparse_token_context_vecs


    for i, num in enumerate(type_doc_nums):
        new_out_doc = []
        for sentence in out_docs_json[num]:
            new_sentence= []
            for token in sentence:
                if token == wtype:
                    new_sentence.append(token)
                    new_sentence.append(token+str(C[i]))
                else:
                    new_sentence.append(token)
            new_out_doc.append(new_sentence)
            del new_sentence
        out_docs_json[num] = new_out_doc


disambiguatedDocs = out_docs_json
json.dump(disambiguatedDocs, open('output/' + "trial_disambiguated_docs.txt", 'w'))

t4 = time.time()

del token_context_counts_cij
del docs_json
del disambiguatedDocs
del out_docs_json

print(len(actual_disambiguated_types))
print("DOC ASSOC_VECS TIME = ", (t2-t1))
print("DOC DISAMBIGUATION TIME = ", (t4-t3))

#####################################################

queries_unseg = []
for query in queries_json:
    queries_unseg.append([token for sentence in query for token in sentence])

assoc_vecs_query = np.zeros((len(word_types), len(word_types)))
token_context_counts_cij_query = {}

t5 = time.time()

for i, query in enumerate(queries_unseg):

    types_counts = Counter(query)
    present_types = types_counts.keys()
    token_counts = types_counts.values()

    idxs = []
    for type_k in present_types:
        idx = word_types.index(type_k)
        idxs.append(idx)

    present_types = list(present_types)
    token_counts = np.array(list(token_counts))

    counts_vec_from_query = np.zeros(len(word_types))
    counts_vec_from_query[idxs] = token_counts

    delta_iks = counts_vec_from_query.copy()

    for tok in set(query):
        counts_vec_of_token = counts_vec_from_query.copy()
        type_index = idxs[present_types.index(tok)]
        counts_vec_of_token[type_index] -= 1
        token_context_counts_cij_query[(tok, i)] = counts_vec_of_token
        assoc_vecs_query[type_index] += delta_iks[type_index]*counts_vec_of_token
        del counts_vec_of_token

t6 = time.time()

assoc_vecs_query = csr_matrix(assoc_vecs_query)

for i, query in enumerate(out_queries_json):
    if i%10 == 0:
        print("query", i, "done")
    new_query = []
    for sentence in query:
        new_sentence = []
        for token in sentence:
            if token in actual_disambiguated_types:
                avg_cossim_to_clusters = []
                for cluster_no in range(1, len(cluster_vectors[token])+1):
                    clus_vecs = cluster_vectors[token][cluster_no]
                    token_vec = csr_matrix(token_context_counts_cij_query[(token, i)])*assoc_vecs_query
                    avg_cossim_to_clusters.append(np.mean(cosine_similarity(token_vec, clus_vecs)))
                predicted_cluster = np.argmax(avg_cossim_to_clusters)+1
                new_sentence.append(token)
                new_sentence.append(token+str(predicted_cluster))
            else:
                new_sentence.append(token)
        new_query.append(new_sentence)
    out_queries_json[i] = new_query

#region
"""query_tokens = [(token, i) for (i, query) in enumerate(queries_unseg) for token in query]

new_query_tokens = []
for token, i in query_tokens:
    if token in actual_disambiguated_types:
        avg_cossim = []
        for cluster_no in range(1, len(cluster_vectors[token])+1):
            clus_vecs = cluster_vectors[token][cluster_no]
            token_vec = csr_matrix(word_token_context_counts_cij_query[(token, i)])*assoc_vecs_query
            avg_cossim.append(np.mean(cosine_similarity(token_vec, clus_vecs)))
        predicted_cluster = np.argmax(avg_cossim)+1
        new_query_tokens.append(token+str(predicted_cluster))
    else:
        new_query_tokens.append(token)

query_lengths, sentence_lengths = [], []
for query in out_queries_json:
    query_lengths.append(len(query))
    for sentence in query:
        sentence_lengths.append(len(sentence))

intermediate_queries = [list(islice(iter(new_query_tokens), elem)) for elem in sentence_lengths]
out_queries_json = [list(islice(iter(intermediate_queries), elem)) for elem in query_lengths]
"""
#endregion

t7 = time.time()

disambiguatedQueries = out_queries_json
json.dump(disambiguatedQueries, open('output/' + "trial_disambiguated_queries.txt", 'w'))

del token_context_counts_cij_query
del queries_json
del out_queries_json

print("QUERY ASSOC VECS TIME = ", (t6-t5))
print("QUERY DISAMBIGUATION TIME = ", (t7-t6))
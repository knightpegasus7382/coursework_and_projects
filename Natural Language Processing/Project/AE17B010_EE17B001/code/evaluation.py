from util import *

# Add your import statements here
import numpy as np


class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here

		# PRECISION = TP/TP+FP
		returned_doc_IDs = query_doc_IDs_ordered[:k]
		precision = sum([ID in true_doc_IDs for ID in returned_doc_IDs])/len(returned_doc_IDs)
		
		# Print statement to check the output for each query and analyse search engine results
		# print(query_id, returned_doc_IDs, [ID in true_doc_IDs for ID in returned_doc_IDs])

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here

		sumPrecision = 0

		# Iterating through all queries to calculate mean Precision
		for doc_ids, query_id in zip(doc_IDs_ordered, query_ids):
			required_qrels = [qrel for qrel in qrels if int(qrel["query_num"])==query_id]
			true_doc_ids = [int(qrel["id"]) for qrel in required_qrels]
			sumPrecision += self.queryPrecision(doc_ids, query_id, true_doc_ids, k)

		meanPrecision = sumPrecision/len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		# RECALL = TP/TP+FN
		returned_doc_IDs = query_doc_IDs_ordered[:k]
		recall = sum([ID in true_doc_IDs for ID in returned_doc_IDs])/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here

		sumRecall = 0

		# Iterating through all queries to calculate mean Recall
		for doc_ids, query_id in zip(doc_IDs_ordered, query_ids):
			required_qrels = [qrel for qrel in qrels if int(qrel["query_num"])==query_id]
			true_doc_ids = [int(qrel["id"]) for qrel in required_qrels]
			sumRecall += self.queryRecall(doc_ids, query_id, true_doc_ids, k)

		meanRecall = sumRecall/len(query_ids)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here

		# F-SCORE = 2*PRECISION*RECALL / (PRECISION + RECALL)
		prec = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		rec = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		try:
			fscore = 2*prec*rec/(prec+rec)
		except ZeroDivisionError:
			# F-Score undefined if both precision and recall are zero
			fscore = "undefined"

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here

		sumFscore = 0
		
		# Iterating through all queries to calculate mean F-score
		for doc_ids, query_id in zip(doc_IDs_ordered, query_ids):
			required_qrels = [qrel for qrel in qrels if int(qrel["query_num"])==query_id]
			true_doc_ids = [int(qrel["id"]) for qrel in required_qrels]
			query_fscore = self.queryFscore(doc_ids, query_id, true_doc_ids, k)

			# Implicitly assigning an F-score of 0 to the undefined F-score cases, in the calculation of mean F-score
			if query_fscore != "undefined":
				sumFscore += query_fscore

		meanFscore = sumFscore/len(query_ids)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here

		# Selecting top k documents
		returned_doc_IDs = query_doc_IDs_ordered[:k]


		# Initialising an array to store the relevance scores of docs
		relevance_scores = np.zeros(len(query_doc_IDs_ordered))

		# Looking for the relevance rank of each document from cran_qrels.json
		# Calculating relevance score as 5 - relevance rank, because ranks range from 1 to 4.
		for true_doc_ID in true_doc_IDs:
			if int(true_doc_ID['id']) in query_doc_IDs_ordered:
				relevance_scores[query_doc_IDs_ordered.index(int(true_doc_ID['id']))] = 5-true_doc_ID['position']

		# Calculating the discount factors for all ranks
		req_relevance_scores = relevance_scores[:k]
		discounts = np.log(np.arange(1, len(req_relevance_scores)+1) + 1)/np.log(2)
		DCG = np.sum(req_relevance_scores/discounts)

		sorted_relevance_scores = np.sort(relevance_scores)[::-1]
		sorted_top_relevance_scores = sorted_relevance_scores[:k]
		IDCG = np.sum(sorted_top_relevance_scores/discounts)

		# If IDCG (and hence DCG) = 0, then no relevant document has shown up in the top k entries
		# So an nDCG of 0 is assigned
		if IDCG == 0:
			nDCG = 0
		else:
			nDCG = DCG/IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here

		sumNDCG = 0
		# Iterating through all queries to calculate mean nDCG
		for doc_ids, query_id in zip(doc_IDs_ordered, query_ids):
			required_qrels = [qrel for qrel in qrels if int(qrel["query_num"])==query_id]
			true_doc_ids = required_qrels
			query_NDCG = self.queryNDCG(doc_ids, query_id, true_doc_ids, k)
			sumNDCG += query_NDCG

		meanNDCG = sumNDCG/len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here

		sumPrecision = 0
		denom = 0

		# For a given value of k, we calculate precision at all ranks <= k

		for k_iter in range(1, k+1):
			# The precision valeu added to average precision calculation only if the latest document is relevant
			if query_doc_IDs_ordered[k_iter-1] in true_doc_IDs:
				sumPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k_iter)
				denom += 1
			
		# If no documents are relevant in the top k results, we simply assign an average precision value of 0
		if denom == 0:
			avgPrecision = 0
		else:
			avgPrecision = sumPrecision/denom

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here

		sumAvgPrecision = 0
		# Iterating through all queries to calculate the MAP
		for doc_ids, query_id in zip(doc_IDs_ordered, query_ids):
			required_qrels = [qrel for qrel in q_rels if int(qrel["query_num"])==query_id]
			true_doc_ids = [int(qrel["id"]) for qrel in required_qrels]
			sumAvgPrecision += self.queryAveragePrecision(doc_ids, query_id, true_doc_ids, k)

		meanAveragePrecision = sumAvgPrecision/len(query_ids)

		return meanAveragePrecision
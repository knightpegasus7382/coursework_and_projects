from util import *

# Add your import statements here

from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		#Fill in code here

		# Selecting the English stopwords
		stops = set(stopwords.words('english'))

		stopwordRemovedText = []
		for seq in text:
			# Only selecting words which are not stopwrods
			stopwordRemovedText.append([word for word in seq if word not in stops])


		return stopwordRemovedText




	
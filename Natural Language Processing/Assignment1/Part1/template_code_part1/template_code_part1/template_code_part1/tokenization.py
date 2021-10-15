from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here

		# Replacing commas and periods in sentences with whitespace, so as to accommodate
		# badly formatted sentence with no spaces after these punctuation marks
		text_commas_replaced = [sentence.replace(",", " ") for sentence in text]
		text_periods_replaced = [sentence.replace(".", " ") for sentence in text_commas_replaced]

		# Applying Naive Tokenization
		tokenizedText = [sentence.strip().split() for sentence in text_periods_replaced]

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here

		# Replacing commas and periods in sentences with whitespace, so as to accommodate
		# badly formatted sentence with no spaces after these punctuation marks
		text_commas_replaced = [sentence.replace(",", " ") for sentence in text]
		text_periods_replaced = [sentence.replace(".", " ") for sentence in text_commas_replaced]

		# Applying Penn-Treebank Tokenization
		tokenizedText = [TreebankWordTokenizer().tokenize(sentence) for sentence in text_periods_replaced]
		
		return tokenizedText
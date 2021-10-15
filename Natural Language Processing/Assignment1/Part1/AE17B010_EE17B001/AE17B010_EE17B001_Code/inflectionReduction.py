from util import *

# Add your import statements here

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here

		# Creating an object of the WordNet Lemmatizer class, to use for lemmatization
		lemmatizer = WordNetLemmatizer()
		reducedText = []
		for sentence in text:
			# Identifying the part-of-speech tags of each token in the sentence
			pos_list = [self.get_wordnet_pos(nltk.pos_tag(token)[0][1]) for token in sentence]
			# Lemmatizing each token using the part-of-speech info
			reducedText.append([lemmatizer.lemmatize(token, pos = part) for token, part in zip(sentence, pos_list)])
		
		return reducedText
	
	# Helper function to obtain the part-of-speech tag in required format to provide as input to lemmatizer
	def get_wordnet_pos(self, treebank_tag):
   		if treebank_tag.startswith('J'):
   		    return wordnet.ADJ
   		elif treebank_tag.startswith('V'):
   		    return wordnet.VERB
   		elif treebank_tag.startswith('N'):
   		    return wordnet.NOUN
   		elif treebank_tag.startswith('R'):
   		    return wordnet.ADV
   		else:
   		    return wordnet.NOUN



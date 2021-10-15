from util import *

# Add your import statements here
import nltk



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		"""
		segmentedText = []
		
		text = text.split(".")
		for sent in text: 
			if not sent : continue
			elif "?" in sent:
				sent = sent.split("?")
				for part in sent:
					if part : segmentedText.append(part.strip())
			else : 
				segmentedText.append(sent.strip())
		"""
		
		segmentedText = None
		# appending all proposed sentence boundaries with special character($) as we want to preserve punctuation
		text = text.strip()
		text = text.replace("?","?$")
		text = text.replace(".",".$")
		text = text.split("$")
		for i in range(len(text)-1,-1,-1):
			if not text[i] : text.pop(i) # removing empty strings
			else : text[i] = text[i].strip(" ") # removing unnecessary whitespace
		
		if len(text)>0 : segmentedText = text 
		return segmentedText
		
		




	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle') # pre-trained model for sentence segmentation
		segmentedText = sent_detector.tokenize(text.strip())
		
		return segmentedText
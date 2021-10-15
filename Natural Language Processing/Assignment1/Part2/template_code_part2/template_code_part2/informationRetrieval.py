from util import *

# Add your import statements here
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.tfidfvectorizer = None
        self.tfidf_model = None
        
    def preprocess_sentence(self, sentence, lemmatizer, stop_words):
        # NOT USED
        tokenized_sentence = word_tokenize(sentence.lower())
        lemmatized_sentence = [lemmatizer.lemmatize(token) for token in tokenized_sentence]
        final_sentence = filter(lambda i : i not in stop_words,lemmatized_sentence)
        return final_sentence
        
    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None
        #Fill in code here
        corpus = []
        index = defaultdict(set)
        for document_id, document in zip(docIDs,docs):
            all_document_sentences = ""
            for sentence in document:
                document_sentence = " ".join(sentence)
                all_document_sentences += document_sentence + " "
                for token in sentence : 
                    index[token].add(document_id)
            corpus.append(all_document_sentences)    
            
        self.index = index

        # Training the TF-IDF Vectorizer Model using the corpus

        tfidfvectorizer = TfidfVectorizer()
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)
        tfidf_tokens = tfidfvectorizer.get_feature_names()
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = docIDs, columns = tfidf_tokens)
        self.tfidfvectorizer = tfidfvectorizer
        self.tfidf_model = df_tfidfvect

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query


        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        #Fill in code here
        for query in queries:
            doc_ids = []
            scores = []
            q = ""
            for sentence in query:
                q = " ".join(sentence) + " "
            tokenized_query = self.tfidfvectorizer.transform([q])

            for doc_id in self.tfidf_model.index : 
                doc = self.tfidf_model.loc[doc_id].to_numpy()
                score = cosine_similarity(tokenized_query.reshape(1,-1),doc.reshape(1,-1))
                scores.append(score[0][0])
            
            o_doc_ids = np.array(self.tfidf_model.index)[np.argsort(np.array(scores))[::-1]]
            doc_IDs_ordered.append(list(o_doc_ids))
        
        return doc_IDs_ordered


#if __name__== "__main__":
#    ir1 = InformationRetrieval()
#    ir1.buildIndex([['Herbivores are typically plant eaters and not meat eaters'], ['Carnivores are typically meat eaters and not plant eaters'],['Deers eat grass and leaves']], ['S1','S2','S3'])
#    print(ir1.rank([["Herbivores plant eaters"]]))


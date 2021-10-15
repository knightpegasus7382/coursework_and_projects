from util import *

# Add your import statements here
from collections import defaultdict, Counter


class InformationRetrieval():

    def __init__(self):
        self.index = None
        
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
        index = defaultdict(set)
        for document_id, document in zip(docIDs,docs):
            for sentence in document:
                for token in sentence: #.split(" ") : 
                    index[token].add(document_id)
        self.index = index

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
            relevant_doc_ids = []
            for sentence in query:
                for token in sentence:#.split(" ") : 
                    relevant_doc_ids.extend(list(self.index[token]))
            
            counter = Counter(relevant_doc_ids)
            ordered_doc_ids = sorted(counter, key=counter.get, reverse=True)
            doc_IDs_ordered.append(ordered_doc_ids)
        
        return doc_IDs_ordered


#if __name__== "__main__":
#    ir1 = InformationRetrieval()
#    ir1.buildIndex([['Herbivores are typically plant eaters and not meat eaters'], ['Carnivores are typically meat eaters and not plant eaters'],['Deers eat grass and leaves']], ['S1','S2','S3'])
#    print(ir1.rank([["Herbivores plant eaters"]]))


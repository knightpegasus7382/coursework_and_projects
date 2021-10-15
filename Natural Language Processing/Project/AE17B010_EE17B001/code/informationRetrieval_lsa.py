from util import *

# Add your import statements here
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities
from operator import itemgetter
from gensim.models import Word2Vec


class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.corpus = None
        self.tfidfvectorizer = None
        self.tfidf_model = None
        self.lsa_vectorizer = None
        self.lsa_model = None
        self.n_components = None
        self.model_type = None
        self.lda_model = None
        self.dictionary = None
        self.loaded_corpus = None
        self.doc_embeddings = None
        self.docIDs = None
        self.w2v_model = None
        
    def preprocess_sentence(self, sentence, lemmatizer, stop_words):
        # NOT USED
        tokenized_sentence = word_tokenize(sentence.lower())
        lemmatized_sentence = [lemmatizer.lemmatize(
            token) for token in tokenized_sentence]
        final_sentence = filter(
            lambda i: i not in stop_words, lemmatized_sentence)
        return final_sentence

    def buildIndex(self, docs, docIDs, n_components, model_type):
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
        arg3 : int 
            An integer denoting the number of components to be used for LSA (-1 for only tfidf)
        arg4 : str 
            A string denoting the model type to be trained and used
        Returns
        -------
        None
        """

        index = None
        # Fill in code here
        self.model_type = model_type
        self.docIDs = docIDs
        # Generating inverted index by going through all tokens
        corpus = []
        index = defaultdict(set)
        for document_id, document in zip(docIDs, docs):
            all_document_sentences = ""
            for sentence in document:
                document_sentence = " ".join(sentence)
                all_document_sentences += document_sentence + " "
                for token in sentence:
                    index[token].add(document_id)
            corpus.append(all_document_sentences)
        self.index = index
        self.corpus = corpus

        # Training the TF-IDF Vectorizer Model using the corpus
        self.n_components = n_components
        tfidfvectorizer = TfidfVectorizer()
        tfidf_wm = tfidfvectorizer.fit_transform(corpus)
        tfidf_tokens = tfidfvectorizer.get_feature_names()
        df_tfidfvect = pd.DataFrame(
            data=tfidf_wm.toarray(), index=docIDs, columns=tfidf_tokens)
                
        self.tfidfvectorizer = tfidfvectorizer
        self.tfidf_model = df_tfidfvect
        
        if self.n_components > 0:
                if self.model_type == 'lsa':
                        lsa_model = TruncatedSVD(n_components=self.n_components, n_iter=100, random_state=42)
                        lsa_transformed = lsa_model.fit_transform(df_tfidfvect)
                        #print(lsa_transformed.shape)
                        df_lsa = pd.DataFrame(data = lsa_transformed, index=docIDs, columns=list(range(self.n_components)))
                        #print(df_lsa.head())
                        self.lsa_model = df_lsa
                        self.lsa_vectorizer = lsa_model
                        
                elif self.model_type == 'lda' :
                        corpus_words = [doc.split(" ") for doc in corpus]
                        dictionary = Dictionary(corpus_words)
                        #dictionary.save('./code/output/corpus_dictionary.dict')
                        self.dictionary = dictionary
                        
                        loaded_corpus = [dictionary.doc2bow(doc) for doc in corpus_words]
                        #corpora.MmCorpus.serialize('./code/output/loaded_corpus.mm', loaded_corpus)
                        self.loaded_corpus = loaded_corpus
                        
                        #lda_model = models.LdaModel(corpora.MmCorpus('./code/output/corpus_corpora.mm'), num_topics = self.n_components)
                        print(self.n_components)
                        lda_model = models.LdaModel(loaded_corpus, num_topics = self.n_components, iterations=100)
                        self.lda_model = lda_model
                        
                        docs_transformed = []
                        for doc in loaded_corpus:
                            #df_lda.append(docs_transformed[self.dictionary.doc2bow(doc)])
                            docs_transformed.append(lda_model.inference([doc])[0][0]) #lda_model.inference([dictionary.doc2bow(query)])
                        
                        self.df_lda = pd.DataFrame(data = docs_transformed, index=docIDs) #, columns=list(range(self.n_components)))
                        print(self.df_lda.shape)
                        
                        
                elif self.model_type == 'w2v':
                        corpus_words = [doc.split(" ") for doc in corpus]
                        w2v_model = Word2Vec(corpus_words, vector_size=self.n_components, min_count=2,window=5, sg=1,workers=4)
                        doc_embeddings = []
                        for doc_tokens in corpus_words:
                            embeddings = []
                            if len(doc_tokens)<1:
                                doc_embeddings.append(np.zeros(self.n_components))
                            else:
                                for tok in doc_tokens:
                                    if tok in w2v_model.wv.key_to_index:
                                        embeddings.append(w2v_model.wv.get_vector(tok))
                                    else:
                                        embeddings.append(np.random.rand(self.n_components))
                                doc_embeddings.append(np.mean(embeddings, axis=0))
                        self.doc_embeddings = doc_embeddings
                        self.w2v_model = w2v_model


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

        # Fill in code here

        # Generating the tf-idf vector for each query and using cosine similarity to obtain ranked document results
        for query in queries:
            scores = []
            q = ""
            for sentence in query:
                q = " ".join(sentence) + " "

            if self.model_type == 'lda' :
                
                query = self.dictionary.doc2bow(q.split(" "))
                model = self.lda_model
                loaded_corpus = self.loaded_corpus
                dictionary = self.dictionary
                tokenized_query = np.array(model.inference([query])[0][0])
                #print(tokenized_query)
                """
                index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))
                query_weight = model[query]
                sim = index[query_weight]
                doc_IDs_ordered.append(sim)
                o_doc_ids = []
                ranking = sorted(enumerate(sim), key = itemgetter(1), reverse=True)
                for rank in ranking:
                    o_doc_ids.append(rank[0])
                doc_IDs_ordered.append(o_doc_ids)
                """
                for doc_id in self.df_lda.index : 
                    doc = self.df_lda.loc[doc_id].to_numpy()
                
                    score = cosine_similarity(tokenized_query.reshape(1, -1), doc.reshape(1, -1))
                    scores.append(score[0][0])
                    
                o_doc_ids = np.array(self.df_lda.index)[np.argsort(np.array(scores))[::-1]]
                doc_IDs_ordered.append(list(o_doc_ids))
                
            elif self.model_type == 'w2v' : 
                w2v_model = self.w2v_model
                query = q.split(" ")
                query_embeddings = []
                for tok in query:
                    if tok in w2v_model.wv.key_to_index:
                        query_embeddings.append(w2v_model.wv.get_vector(tok))
                    else:
                        query_embeddings.append(np.random.rand(self.n_components))
                query_embeddings = np.mean(query_embeddings, axis=0)
                
                scores = []
                for doc in self.doc_embeddings : 
                    score = cosine_similarity(query_embeddings.reshape(1, -1), doc.reshape(1, -1))
                    scores.append(score[0][0])
                # print(scores,self.docIDs)
                o_doc_ids = np.array(self.docIDs)[np.argsort(np.array(scores))[::-1]]
                doc_IDs_ordered.append(list(o_doc_ids))
    
                
            else : 
                tokenized_query = self.tfidfvectorizer.transform([q])
                if self.n_components > 0 : tokenized_query = self.lsa_vectorizer.transform(tokenized_query.reshape(1,-1))

                for doc_id in self.tfidf_model.index:
                    if self.n_components > 0 : doc = self.lsa_model.loc[doc_id].to_numpy()
                    else: doc = self.tfidf_model.loc[doc_id].to_numpy()
                
                    #score = cosine_similarity(reduced_query.reshape(1, -1), doc.reshape(1, -1))
                    score = cosine_similarity(tokenized_query.reshape(1, -1), doc.reshape(1, -1))
                    scores.append(score[0][0])

                # Using descending order of cosine similarity scores to rank the documents
                o_doc_ids = np.array(self.tfidf_model.index)[np.argsort(np.array(scores))[::-1]]
                doc_IDs_ordered.append(list(o_doc_ids))
                

        return doc_IDs_ordered


# if __name__== "__main__":
#    ir1 = InformationRetrieval()
#    ir1.buildIndex([['Herbivores are typically plant eaters and not meat eaters'], ['Carnivores are typically meat eaters and not plant eaters'],['Deers eat grass and leaves']], ['S1','S2','S3'])
#    print(ir1.rank([["Herbivores plant eaters"]]))

"""
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
with open('corpus.txt','r') as inf : corpus = inf.readlines()
corpus_words = [corp.split(" ") for corp in corpus]
corpus_dict = Dictionary(corpus_words)
corpus_corpus = [corpus_dict.doc2bow(text) for text in corpus_words]
lda = LdaModel(corpus_corpus,10)

# query_bow = Dictionary.doc2bow(query.split())
query_bow = corpus_dict.doc2bow(query.split())
query_lda = lda(query_bow)
corpus_corpus = [corpus_dict.doc2bow(text) for text in corpus_words]
lda = LdaModel(corpus_corpus,10)

"""
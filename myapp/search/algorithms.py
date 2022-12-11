import collections
from collections import defaultdict

from numpy import linalg as la
import numpy as np
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re


def search_in_corpus(new_query, index, idf, tf, df, t_corpus, top=10):
    def build_terms(line):
        stemmer = PorterStemmer()

        stop_words = set(stopwords.words("english"))
        line = line.lower()  # Convert to lowercase
        line = line.split()  # Tokenize the text to get a list of terms
        line = [x for x in line if x not in stop_words]  # remove the stopwords
        line = [x for x in line if x.startswith(("@", "https://", "$", '#')) is not True]  # remove mentions
        line = [re.sub('[^a-z]+', '', x) for x in line]
        line = [stemmer.stem(word) for word in line]  # perform stemming
        return line

    def search_tf_idf(query, idx, term_freq, doc_freq, tw_corpus, inv_freq):
        """
        output is the list of documents that contain any of the query terms.
        So, we will get the list of documents for each query term, and take the union of them.
        """
        query = build_terms(query)
        docs = set()
        for term in query:
            try:
                # store in term_docs the ids of the docs that contain "term"
                term_docs = [posting[0] for posting in index[term]]

                # docs = docs Union term_docs
                docs = docs.union(term_docs)
            except:
                # term is not in index
                pass
        docs = list(docs)
        rank_docs = rank_documents(query, docs, idx, inv_freq, term_freq, doc_freq, tw_corpus)
        return rank_docs

    def rank_documents(terms, docs, idx, inv_freq, term_freq, doc_freq, tweets_corpus):
        """
        Perform the ranking of the results of a search based on the tf-idf weights

        Argument:
        terms -- list of query terms
        tweets -- list of tweets, to rank, matching the query
        index -- inverted index data structure
        idf -- inverted document frequencies
        tf -- term frequencies

        Returns:
        Print the list of ranked documents
        """

        doc_vectors = defaultdict(lambda: [0] * len(terms))
        query_vector = [0] * len(terms)

        # compute the norm for the query tf
        query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.

        query_norm = la.norm(list(query_terms_count.values()))

        for termIndex, term in enumerate(terms):  # termIndex is the index of the term in the query
            if term not in index:
                continue

            # Compute tf*idf(normalize TF as done with documents)
            query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

            # Generate doc_vectors for matching docs
            for tweet_index, (tweet, postings) in enumerate(index[term]):
                if tweet in docs:
                    doc_vectors[tweet][termIndex] = tf[term][tweet_index] * idf[term]

        # Calculate the score of each doc
        # compute the cosine similarity between queryVector and each docVector

        doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
        doc_scores.sort(reverse=True)
        result_docs = [x[1] for x in doc_scores]

        if len(result_docs) == 0:
            print("No results found, try again")
        return result_docs

    ranked_docs = search_tf_idf(new_query, index, t_corpus, tf, df, idf)

    return ranked_docs[0:top]

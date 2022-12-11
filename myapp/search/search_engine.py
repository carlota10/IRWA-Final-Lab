import random

from myapp.search.objects import ResultItem, Document
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from array import array
import re
import math


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


# 1. create create_tfidf_index
def create_index_tfidf(tweet_corpus):
    """
    Implement the inverted index and compute tf, df and idf

    Argument:
    lines -- collection of Wikipedia articles
    num_documents -- total number of documents

    Returns:
    index - the inverted index containing terms as keys and the corresponding
    list of document these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents
    df = defaultdict(int)  # document frequencies of terms in the corpus
    idf = defaultdict(float)

    for tweet in tweet_corpus.keys():
        page_id = tweet_corpus[tweet].id
        terms = build_terms(tweet_corpus[tweet].description)

        current_page_index = {}

        for position, term in enumerate(terms):  # terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corresponding list
                current_page_index[term][1].append(position)
            except KeyError:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term] = [page_id, array('I', [position])]

        # normalize term frequencies
        norm = 0
        for term, posting in current_page_index.items():
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            tf[term].append(np.round(len(posting[1]) / norm, 4))
            df[term] += 1

        # merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        # Compute IDF following the formula (3) above.
        for term in df:
            idf[term] = np.round(np.log(float(len(tweet_corpus) / df[term])), 4)

    return index, tf, df, idf


def build_demo_results(corpus: dict, search_id):
    """Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking"""

    res = []
    size = len(corpus)
    ll = list(corpus.values())
    for index in range(random.randint(0, 40)):
        item: Document = ll[random.randint(0, size)]
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                              "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.random()))

    # for index, item in enumerate(corpus['Id']):
    #     # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
    #     res.append(DocumentInfo(item.Id, item.Tweet, item.Tweet, item.Date,
    #                             "doc_details?id={}&search_id={}&param2=2".format(item.Id, search_id), random.random()))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    index = []
    tf = []
    df = []
    idf = []

    def create_index(self, corpus):
        self.index, self.tf, self.df, self.idf = create_index_tfidf(corpus)

    def search(self, search_query, search_id, tweets):
        print("Search query:", search_query)

        result_tweets = []

        results = search_in_corpus(search_query, self.index, self.idf, self.tf, self.df, tweets, top=10)

        for result in results:
            result_tweets.append(tweets[result])

        return result_tweets

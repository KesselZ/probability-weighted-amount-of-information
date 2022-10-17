from nltk.probability import FreqDist
import math
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups

def get_doc_scores(data, input_text):
    processed_input = []
    for i in input_text:
        str = ''
        for each in i:
            str = str + each + ' '
        processed_input.append(str)

    data = data + processed_input
    topic_num = len(processed_input)

    vectorizer = CountVectorizer(stop_words='english')
    corpus = pd.Series(data)
    tf_idf_matrix = vectorizer.fit_transform(corpus)
    data_array = tf_idf_matrix.toarray()

    l = len(data_array)

    topic_list = []
    for i in range(topic_num):
        topic_list.append(data_array[l - 1 - i])
    topics = np.array(topic_list, np.int64)
    topics = normalize(topics, norm='l2')

    final_result = []
    for doc in data_array:
        ranks = np.inner(doc, topics)
        indexes = np.flip(np.argsort(ranks)[-20:])
        scores = np.array([ranks[res] for res in indexes])
        final_result.append(scores)

    return final_result


def default_tokenizer(doc):
    # This part was copied from Top2Vec tokenizer, if you are using a specific tokenizer you should not use the default one when computing the measure
    """Tokenize documents for training and remove too long/short words"""
    return simple_preprocess(strip_tags(doc), deacc=True)


def PWI(docs, list, num_words=3):
    """
    :param model: top2vec model
    :param docs: list of strings
    :param num_topics: number of topics to use in the computation
    :param num_words: number of words to use
    :return: PWI value
    """

    # This is used to tokenize the data and strip tags (as done in top2vec)
    tokenized_data = [default_tokenizer(doc) for doc in docs]
    # Computing all the word frequencies
    # First I concatenate all the documents and use FreqDist to compute the frequency of each word
    word_frequencies = FreqDist(np.concatenate(tokenized_data))

    # Computing the frequency of words per document
    # Remember to change the tokenizer if you are using a different one to train the model
    dict_docs_freqs = {}
    for i, doc in enumerate(docs):
        counter_dict = FreqDist(default_tokenizer(doc))
        if i not in dict_docs_freqs:
            dict_docs_freqs[i] = counter_dict

    PWI = 0.0
    p_d = 1 / len(docs)

    docs_scores = get_doc_scores(docs, list)

    for i, doc in enumerate(docs):

        topic_words = []
        for j in list:
            topic_words.append(np.array(j))

        topic_scores = docs_scores[i]

        for words, t_score in zip(topic_words, topic_scores):
            for word in words[:num_words]:
                if word not in dict_docs_freqs[i]:
                    # This is added just for some specific cases when we are using different collection to test
                    continue
                # P(d,w) = P(d|w) * p(w)
                p_d_given_w = dict_docs_freqs[i].freq(word)
                p_w = word_frequencies.freq(word)
                p_d_and_w = p_d_given_w * p_w
                left_part = p_d_given_w * t_score
                if (p_d_and_w != 0):
                    log = math.log(p_d_and_w / (p_w * p_d))
                else:
                    log = 0
                PWI += left_part * log
    return PWI


if __name__ == '__main__':

    #example for usage

    # Get documents
    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = dataset['data']

    # We assume you already have topics extracted from above documents
    # Put these topics in form of list
    list=[['card', 'monitor', 'video'], ['game', 'hockey', 'team'], ['fbi', 'batf', 'koresh'], ['medical', 'health', 'cancer']]

    #num_words limit the number of keywords in each topic
    print("PWI:", PWI(data, list, num_words=3))

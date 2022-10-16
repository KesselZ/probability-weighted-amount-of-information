# probability-weighted-amount-of-information

# What is PWI?

This paper presents a mathematical definition of the “probability-weighted amount of information” (PWI), a measure of the specificity of terms in documents that is based on an information-theoretic view of retrieval events.

# What is it for?
It is used as a metric for topic modeling problems. It was introduced in the paper "An information-theoretic perspective of tf–idf measures" (https://www.sciencedirect.com/science/article/abs/pii/S0306457302000213).

And it was used in a topic modeling algorithm "top2vec" (https://github.com/ddangelov/Top2Vec). 

For more discussion, you can find them here: https://github.com/ddangelov/Top2Vec/issues/158

# How to use it?
Since this is a metric for topic modeling. We assume user already get a result (topics) from their own topic modeling algorithm(such as LDA). 

With the documents, and topics (extracted from documents), please run the following code:

import PWI
print("PWI:", PWI.PWI(data, list, num_words=3))

You could also find a example in the PWI.py, in its main function.

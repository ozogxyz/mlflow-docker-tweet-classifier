from collections import OrderedDict, Counter, defaultdict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
import heapq
import itertools
import math

class BoW(TransformerMixin):
    """
    Bag of words transformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurrences (the highest first)
        # store most frequent tokens in self.bow

        dict_1 = dict(Counter([token for text in X for token in text.split()]))
        self.bow = sorted(dict_1, key=dict_1.get, reverse=True)[:self.k]
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        dict_1 = {key:text.split().count(key) for key in self.bow}
        result = [v for k, v in dict_1.items()]

        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf transformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        
        if self.k is None:
            self.k = len(set(' '.join(X).split()))

        term_freq = dict(Counter([token for text in X for token in text.split()]))
        self.idf = {term: math.log10(len(X) / count) for term, count in term_freq.items()}
        self.idf = dict(sorted(self.idf.items(), key=lambda x:x[1],reverse=True))
        self.idf = dict(itertools.islice(self.idf.items(), self.k))  

        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """

        word_freq = dict(Counter([token for token in text.split()]))
        tfidf = {key: 0 for key in self.idf}
        for key in word_freq:
            if key in tfidf.keys():
                tfidf[key] = self.idf[key] * word_freq[key] / len(text.split())
        result = [v for k, v in tfidf.items()]        

        eps = 1e-5
        if self.normalize is True:            
            result = result / (np.linalg.norm(result) + eps)
        
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.idf





import numpy
from langdetect import detect_langs
import nltk.tokenize
from numpy.ma import where, ceil

import skipthoughts
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib.pyplot as plot

"""
Author: William Blackie

Methods for the summation of news articles via Skip-Thought vectors sentence embeddings.

"""


class ArticleSummation:
    def __init__(self, article, error_dict, compression_rate):
        self.article = article
        self.error_dict = error_dict
        self.compression_rate = compression_rate

    def calculate_words(self):
        """
        :return: Integer of how many words in self.article.
        """
        return len(self.article[0].split())

    def summarise(self):
        """
        Performs summation of articles;
        Implemented using Kmeans for clustering.
        :return: List of Strings for summary
        """

        avg = []
        total_words = self.calculate_words()
        print('Splitting into sentences...'),
        self.sentence_tokenize()
        print('done')
        if self.is_language_english():
            print('Encode sentences...'),
            encoded_article = self.encode_article()
            print('done')

            print("Clustering"),
            number_of_clusters = int(ceil(
                len(encoded_article[0]) ** self.compression_rate))  # Get user defined number of clusters using compression rate.

            k_means_clusters = KMeans(n_clusters=number_of_clusters, random_state=0).fit(
                encoded_article[0])  # Get the clusters of the article.

            for i in range(number_of_clusters):
                index = where(k_means_clusters.labels_ == i)[0]
                avg.append(numpy.mean(index))  # Find centroid from mean of points.

            closest, temp = pairwise_distances_argmin_min(k_means_clusters.cluster_centers_,
                                                          encoded_article[0])  # Assign labels based on closest center.

            #self.plot_article_clusters(k_means_clusters.cluster_centers_, closest)  # Plotting article clusters for testing.

            ordering = sorted(range(number_of_clusters),
                              key=lambda cluster: avg[cluster])  # sort clusters by their average value.
            print ("done")

            self.article[0] = ' '.join([self.article[0][closest[index]] for index in ordering])  # Join selected clusters in order.

            [x.encode('utf-8') for x in self.article]  # encode article in utf-8 (removes ascii tokens) for formatting.

            new_total_words = self.calculate_words()

            return x, total_words, abs(new_total_words-total_words), self.error_dict
        else:
            return "Article summation failed", 0, 0, self.error_dict

    @staticmethod
    def plot_article_clusters(k_means_cluster_centers, labels):
        """
        Basic method to plot clusters produced by kmeans.
        :param k_means_cluster_centers:  k_means_clusters.cluster_centers_
        :param labels: labels from pairwise_distances
        :return: None
        """
        plot.scatter(k_means_cluster_centers[:, 0], k_means_cluster_centers[:, 1], c=labels, s=200, alpha=0.5)
        plot.show()

    def sentence_tokenize(self):
        """
        Method to tokenize an article; removes empty strings.
        """
        sentences = nltk.tokenize.sent_tokenize(self.article[0])
        for index in reversed(range(len(sentences))):
            sent = sentences[index]
            sentences[index] = sent.strip()
            if sent == '':
                sentences.pop(index)
            self.article[0] = sentences

    def encode_article(self):
        """
        Method to embed sentences into sentence vectors.
        :return: List of encoded sentences
        """
        sentences_to_encode = []
        encoded_article = [None]
        sentence_list = [0]
        sent_count = 0

        for sentence in self.article:
            sent_count += len(sentence)
            sentence_list.append(sent_count)
            for sent in sentence:
                sentences_to_encode.append(sent)  # Split into sentences

        print('Loading models...'),
        model = skipthoughts.LoadModel().load_model()
        encoder = skipthoughts.Encoder(model)
        print('Encoding sentences...'),
        encoded_sentences = encoder.encode(sentences_to_encode, verbose=False)
        print('Done')

        for i in range(len(self.article)):
            begin = sentence_list[i]
            end = sentence_list[i + 1]
            encoded_article[i] = encoded_sentences[begin:end]
        return encoded_article

    def is_language_english(self):
        """
        Detect if all sentences in article are english;
        this method is not perfect as it will reject any other language even if single words are non-english.
        :return: Boolean True if all sentences are English only, False if otherwise.
        """
        for string in self.article:
            article_language = detect_langs(str(string))
            for item in article_language:
                if item.lang == "en":
                    continue
                else:
                    print string
                    self.error_dict["language_error"] = "Non-English text detected; language detected: " + str(item.lang)
                    return False
            return True

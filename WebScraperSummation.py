import threading

from Utils import WebScraperUtils
from ArticleSummation import ArticleSummation

"""
Author: William Blackie

Class for creating summaries from any given URL, best results are gotten from bb.co.uk and thegaurdian.com.
"""

article = [None]


class WebScraperSummation:
    """
    Class for taking any URL and creating a summarised article.
    """

    def __init__(self):
        pass

    @staticmethod
    def scrape(url, include_lists, compression_rate, error_dict):
        """
        Method for taking any URL and returning a Skip-Thought summarised article;
        The webscraper used is tailored for BBC.co.uk and thegaurdian.com articles but any can be used.
        :param error_dict: Dict containing error strings.
        :param url: String containing any URL with an article.
        :param include_lists: Boolean if <li> elements should be included in summation;
        For best results include_lists = False.
        :param compression_rate: Float (0.1-1.0) of the rate of compression for summation.
        :return: List[0] where 0 is a String or String, total_words, total_words_removed and an error_dict.
        """
        web_scraper = WebScraperUtils.WebScraperUtils()
        text = web_scraper.get_url(url, include_lists)
        if text is None:
            error_dict["no_text"] = "No text found on website or website URL not found."
            return "Article summation failed", 0, 0, error_dict  # Return with error dict
        print "loaded article"
        article_summariser = ArticleSummation(article=text, error_dict=error_dict, compression_rate=compression_rate)
        return article_summariser.summarise()
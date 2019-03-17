from ArticleSummation import ArticleSummation
"""
Author: William Blackie

Class for creating summaries from .txt files loaded from a Django webservice.
"""


class TextFileSummation:
    """
    Class to handle text.txt file upload to Django webservice, currently only handles UTF-8 encoding.
    """

    def __init__(self):
        pass

    def __str__(self):
        return self.name.encode('utf8')  # Force strings to be encoded as UTF-8, can reduce cases of UnicodeDecodeError.

    @staticmethod
    def upload_txt(request, errors):
        """
        Method to handle text from a UTF-8 encoded text file; creates a concatenated string in place in a list.
        :param errors:
        :param request:
        :return: List containing a concatenated string at List[0]
        """
        sanitised_article = [""]
        article = []

        uploaded_file = request.FILES["file"]
        for line in uploaded_file:
            line = line.rstrip()  # Remove "\n" and "\r"
            if len(line) > 1:  # check if empty string i.e. char "\n" becomes ""
                try:
                    if isinstance(line, str):
                        line = line.decode("utf-8")  # Decode for summariser to handle the text
                    article.append(line)
                except UnicodeDecodeError:
                    errors["NON-UTF-8-ENCODING"] = "Text submitted is non UTF-8 encoded"

        sanitised_article[0] = [' '.join(article)]
        return sanitised_article[0], errors

    def summarise_text_file(self, request, compression_rate, error_dict):
        """
        Method to pass a Django request to article_summation.py and summarise it.
        :param error_dict:
        :param request: Django request (HttpRequest objects);
        more info: https://docs.djangoproject.com/en/2.1/ref/request-response/
        :param compression_rate: Float (0.1-1.0) for rate of compression.
        :return: List[0] where 0 is a String or String, total_words, total_words_removed and an error_dict.
        """
        text, error_dict = self.upload_txt(request, error_dict)
        if len(error_dict) > 0:
            if text is None:
                error_dict["No_text"] = "No text found in submitted article"
            return "Article summation failed", 0, 0, error_dict  # Return with error dict
        print "loaded article"
        article_summariser = ArticleSummation(article=text, error_dict=error_dict, compression_rate=compression_rate)
        return article_summariser.summarise()


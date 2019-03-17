import json
import os
import sys
from bs4 import BeautifulSoup

reload(sys)
sys.setdefaultencoding('utf8')  # Set encoding for faster parsing


"""
Author: William Blackie

Class to create corpus from WikiExtractor xlm output.

"""


class DataUtils:
    def __init__(self):
        pass

    @staticmethod
    def load_json(json_path):
        """
        Method to return a Json object from a json file.
        :param json_path: String containing path to Json file.
        :return data: data Json object.
        """
        with open(json_path, 'w') as json_file:
            data = json.load(json_file)

        return data

    @staticmethod
    def unique_tokens(text, dictionary_path):
        """
        Method to find all tokens within a text file.
        :param text: String of text.
        :param dictionary_path: path to saved dictionary to append new words, saves storing variables all in ram.
        """
        words = text.split()
        dict = open(dictionary_path, 'w')

        for word in words:
            if word not in dict:
                dict.write(str(word) + "\n")
        dict.close()

    @staticmethod
    def get_data_directory(directory):
        """
        Method to find all files within a directory, used for indexing all the created sanitised corpses.
        :param directory: Relative path to data dir root
        :return file_list: list of relative directories
        :return root_directory: String containing the root directory
        """
        file_list = []
        for root_directory, dirs, files in os.walk(directory):
            for name in files:
                file_list.append(os.path.join(root_directory, name))
        return file_list, root_directory

    def create_corpus(self, root_dir):
        """
        Method to create batches of 250MB sanitised .text files
        :param root_dir: Path of root directory to json files from WikiExtractor
        """
        directories, root = self.get_data_directory(root_dir)
        counter = 0
        wiki_article = ""

        for directory in directories:
            print(directory)
            with open(directory, 'r') as dict_file:
                for line in dict_file:
                    temp = json.loads(line)
                    wiki_article += str(temp['text'])
                clean_text = BeautifulSoup(wiki_article, 'lxml', from_encoding="utf-8").text
                clean_text = clean_text.replace('\n', '').replace('>', '')
                print(clean_text)
                wiki_article = ""

                file_name = r'D:\Projects\skip-thoughts\text\corpus\corpus' + str(counter) + ".txt"
                with open(file_name, 'a') as corpus:
                    corpus.writelines(clean_text)
                    if int(os.path.getsize(file_name)) > 262144000:  # 0.25GB per file
                        counter += 1
                        print(os.path.getsize(file_name))




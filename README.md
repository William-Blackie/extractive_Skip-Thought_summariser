# Extractive_Skip-Thought_summariser
A final year project for the University of the West of England. This module allows users to summarise a URL or a testfile using pre-trained Skip-Thought vectors for more details of training please see below.

This summariser builds upon the original work of Ryan Kiros and others to build an extractive summariser and is tailored for deployment in a Django webserver although others may work.

The summariser itself was trained using a sample of an english Wikipedia dump and then after roughly two weeks of training implemented vocuablary expansion using Facebook's Fast-Text vectors increasing the vocabulary greatly.

The front-end Django web-server I made can be found [here.](https://github.com/William-Blackie/Extractive_Skip-thought_Summeriser) This solution offers more of a complete interaction with this module as text file summation is coupled with using an WSGI request to invoke the methods, although article summation via web scraping will still work.

## Getting started
To get started you will need:

### Hardware
For Summation:
* 8GB of ram(untested), more is better.
* A recent CPU
* Around 5GB of storage.

For Training:
* Minimum 16GB of ram, more is better.
* A recent CPU
* Around 5GB of storage + however big your corpus and vectors if you are doing the vocabulary expansion step.

### Software

[git bash](https://gitforwindows.org/) - to clone the repo.

[Anaconda Python 2.7](https://www.anaconda.com/distribution/) - to be able to run the server.

[CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive)  - to be able to run Theano (used for training and summation)

[CUDNN 7.42](https://developer.nvidia.com/rdp/cudnn-archive) - CUDA updates used in Theano, the files need to be placed in the installation directory of CUDA.

## Cloning the repo
Using git bash:
```
    git clone https://github.com/William-Blackie/extractive_Skip-Thought_summariser
```

## Models
My trained models can be found here:
```
https://1drv.ms/f/s!AobPYFxChsscgoFVh3Xnz5b4Hk33vw
```
Password 
```
Skippy
```
place them into the models folder.
## Installation
### Theano
Theano requires environment variables to be set, which can be done programmatically but for ease of please save the following into your home directory as theanorc.txt:

Please make sure the CUDA and dnn paths match your own installation.

Note if you are planning on training device=cuda and a recent Nividia compute card or graphics card is needed, Minimum 6gb ram but more will allow for increased batch size.
```
[global]
floatX = float32
device = cpu
force_device = True
allow_gc = True
allow_pre_alloc = False

[theano]
exception_verbosity=high

[cuda]
root = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
CUDA_LAUNCH_BLOCKING=1 

[nvcc]
fastmath = True

[dnn] 
library_path = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
include_path = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include
enabled = True

[mkl]
MKL_THREADING_LAYER=GNU
```

## Anaconda
Open the terminal and follow the bellow instructions.

Navigate to the root of the cloned project and run to create a conda environment for the project:
```
  conda env create -n skippy_env -f package-list.txt
```

Activate your environment:
```
  conda activate skippy_env
```

## Basic usage

### Summarise a URL
Create a python file in the root directory and add the following:

```
import WebScraperSummation 

"""
Author: William Blackie
Example method for the summation of english URL.
"""

# Some variables to get you started:
url = r"https://www.bbc.co.uk/news/business-47287386"
remove_lists = True
compression_rate = 0.7 # (0.1-1.0)
errors = {} # Dict for error checking; non-english text for example.


def web_skip_thought_summeriser:
    scraper = WebScraperSummation.WebScraperSummation()
    summary_text, total_words, total_words_removed, errors = scraper.scrape(
                        url, remove_lists,
                        compression_rate, errors)
     print summary_text
```

Note that summation will take some time depending article length and a higher compression rate, having a recent CPU will help but this project will benefit greatly from being properly deployed.

## Training
Refactor the setup.py script to the correct directory to your WikiExtractor output and run, this will create 250MB corpus files and a dictionary from them, after which the current  corpus will be used for training.
The Uni-Skip vectors have been trained for two weeks similar to the ones used in the original research implementation but using a non-fiction corpus.

### WikiExtractor
WikiExtractor can be found here which includes details for usage. [Github](https://github.com/attardi/wikiextractor)

### VectorExpansion
In vectorExpansion.py set the path variable to your chosen FastText vectors which will load the FastText vectors into your embeddings, vastly increasing the accuracy of summation.
## Authors

* **William Blackie** - [Github](https://github.com/William-Blackie) - Email: contact@williamblackie.com


## License

This project is licensed under the apache 2.0 license - see the [here](http://www.apache.org/licenses/LICENSE-2.0) file for details.

If you choose to extend this project please give credit to the training material sources as well as the original authors and a link to this project.

## Acknowledgments

* This project uses a research implementation by [Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja](https://github.com/ryankiros/skip-thoughts) of Skip-Thought to create and train my vectors and the encoding of sentences into vectors. For more information please read their paper [here.](https://arxiv.org/abs/1506.06726)
* This project implemented FaceBook's [FastText vectors](https://fasttext.cc/docs/en/crawl-vectors.html) used in the vocabulary expansion step after training.
* The corpus used in training of my vectors was a full dump of [english Wikipedia](https://dumps.wikimedia.org/), of which a sample was taken.

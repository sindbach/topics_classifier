
FROM ubuntu:14.04

RUN apt-get update && apt-get install -y \
    git \
    nano \
    python-pip \
    python-numpy \
    python-scipy \
    python-matplotlib \
    python-sympy

RUN pip install pymongo
RUN pip install nltk
RUN pip install gensim

RUN useradd --create-home --shell /bin/bash ubuntu
USER ubuntu
WORKDIR /home/ubuntu

RUN python -m nltk.downloader -d ~/nltk_data stopwords

RUN git clone --depth 1 https://github.com/sindbach/topics_classifier.git /home/ubuntu/topics_classifier

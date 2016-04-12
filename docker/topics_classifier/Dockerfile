
FROM ubuntu:14.04

RUN apt-get update && apt-get install -y \
    git \
    nano \
    python-pip \
    python-dev \
    libblas-dev \
    liblapack-dev \
    gfortran

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --shell /bin/bash ubuntu
USER ubuntu
WORKDIR /home/ubuntu 

RUN python -m nltk.downloader -d ~/nltk_data stopwords

RUN git clone --depth 1 https://github.com/sindbach/topics_classifier.git /home/ubuntu/topics_classifier

ENTRYPOINT ["/bin/bash"]

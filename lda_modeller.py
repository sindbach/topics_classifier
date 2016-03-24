#!/bin/env python 

import argparse
import logging

from gensim import corpora
from gensim import models

from reader import MongoReader

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

class CustomCorpus(object):
    ''' Custom corpus iterator
    '''
    def __init__(self, reader, wordsmatrix):
        ''' init
            :param reader: source Reader object
            :param wordsmatrix: matrix of words
            :type wordsmatrix: id2word corpus
        '''
        self.reader = reader
        self.matrix = wordsmatrix
        self.titles = list()

    def __iter__(self):
        for title, tokens in self.reader.iterate():
            self.titles.append(title)
            yield self.matrix.doc2bow(tokens)


class BuildLDAModel(object):
    ''' Build LDA model file. 
    '''
    def __init__(self, fileoutput, num_topics=20, num_passes=10, 
                 num_min_docs=5, num_min_pct=0.5, num_topic_words=15):
        ''' init
            :param fileoutput: output model file 
            :param num_topics: number of topics to be generated
            :param num_passes: number of passes iteration to generate the model
            :param num_min_docs: number of documents to ignore
            :param num_min_pct: number of percentage of documents to ignore
            :param num_words: number of words to show per topic
        '''
        self.fileoutput = fileoutput
        self.num_topics = num_topics
        self.num_passes = num_passes
        self.num_min_docs = num_min_docs
        self.num_min_pct = num_min_pct
        self.num_topic_words = num_topic_words
        self.lda_model = None

    def build(self, reader):
        ''' Build model
            :param reader: source Reader object
        '''
        doc_stream = (text for _, text in reader.iterate())
        wordsmatrix = corpora.Dictionary(doc_stream)
        # ignore words that appear in less than 5 documents or more than 5% documents
        wordsmatrix.filter_extremes(no_below=self.num_min_docs, no_above=self.num_min_pct)

        corpus = CustomCorpus(reader=reader, wordsmatrix=wordsmatrix)

        self.lda_model = models.LdaModel(corpus, num_topics=self.num_topics, 
                                         id2word=wordsmatrix, passes=self.num_passes)
        self.lda_model.save(self.fileoutput)        
        

    def dump_topics(self, topics_file, model=None):
        ''' Dump topics into a file
            :param topics_file: output topics file
            :param model: lda model file. Optional. 
        '''
        if model:
            lda_model = models.LdaModel.load(model)
            self.lda_model = lda_model
        if not self.lda_model:
            raise Exception("There is no model set to output topics. Run build() first.")

        with open(topics_file, 'w') as tfile:
            for topic in self.lda_model.show_topics(num_topics=self.num_topics, 
                                                    num_words=self.num_topic_words):
                tid,stats = topic
                tfile.write("TopicId [%s]: ?? \n" % tid)
                stats = stats.split(" + ")
                for stat in stats:
                    tfile.write("\t%s\n" % stat)
                tfile.write("\n")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LDA model file.")
    parser.add_argument('--model', help="Specify output model file. default: lda.model", default="./lda.model")
    parser.add_argument('--db', help="Specify MongoDB db name. default:bow", default='bow')
    parser.add_argument('--coll', help="Specify MongoDB collection name. default:traning", default='training')
    parser.add_argument('--mongoURI', help="Specify MongoDB URI for different server/ports", default="mongodb://localhost:27017")
    parser.add_argument('--query', help="Specify a query to filter MongoDB. default:all", default={})
    parser.add_argument('--limit', help="Specify the limit of records to read from source. default: None", default=None)
    parser.add_argument('--dumptopics', help="Specify a file to print topics to")
    parser.add_argument('--nobuild', help="Flag to skip building a model", action="store_true", default=False)
    args = parser.parse_args()

    if (not args.model) or (args.dumptopics and not args.model):
        parser.print_help()
        sys.exit(0)

    builder = BuildLDAModel(fileoutput=args.model)
    if not args.nobuild:
        reader = MongoReader(query=args.query, mongoURI=args.mongoURI, 
                             dbName=args.db, collName=args.coll, limit=args.limit)
        builder.build(reader)
    
    if args.dumptopics:
        builder.dump_topics(topics_file=args.dumptopics, model=args.model)



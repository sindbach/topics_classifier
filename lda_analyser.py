#!/usr/bin/env python 

import argparse
import logging
import json 

from reader import MongoReader

from gensim import corpora
from gensim import models

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
_logger = logging.getLogger(__name__)


class LDAAnalyser(object):
    def __init__(self, model, reader, topic_file=None):
        ''' init 
            :param model: LDA model
            :param reader: source reader object
            :param topicfile: lookup topic file.
        '''
        lda_model = models.LdaModel.load(model)
        self.lda_model = lda_model
        self.topic_file = topic_file
        self.reader = reader
        # Only analyse the question
        self.reader.return_fields = ['question']

    def analyse(self):
        ''' analyse '''
        wordsmatrix = corpora.Dictionary()
        wordsmatrix.merge_with(self.lda_model.id2word) 
        
        lookup = None
        if self.topic_file: 
            with open(self.topic_file) as tfile:    
                lookup = json.load(tfile)

        for doc in self.reader.iterate():
            # doc[1] is the texts, while doc[1] is the title.
            content = wordsmatrix.doc2bow(doc[1])
            stats = list(sorted(self.lda_model[content], key=lambda x:x[1]))
            most_related = stats[-1]
            _logger.info("Most related %s", most_related)
            key = str(most_related[0])
            if lookup:
                _logger.info("Topic title is: %s", lookup[key].get('topic'))
        return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given a model file classify a topic.")
    parser.add_argument('--model', help="Specify input model file. default: lda.model", default="./lda.model")
    parser.add_argument('--db', help="Specify MongoDB db name. default:topicsDB", default='topicsDB')
    parser.add_argument('--coll', help="Specify MongoDB collection name. default:data", default='data')
    parser.add_argument('--mongoURI', help="Specify MongoDB URI for different server/ports", default="mongodb://localhost:27017")
    parser.add_argument('--query', help="Specify a query to filter MongoDB. default:all", default={})
    parser.add_argument('--limit', help="Specify a number of documents to classify. default:5", type=int, default=5)
    parser.add_argument('--topics', help="Specify a lookup topic file.")
    args = parser.parse_args()

    if not args.model:
        parser.print_help()
        sys.exit(0)

    reader = MongoReader(query=args.query, mongoURI=args.mongoURI, 
                         dbName=args.db, collName=args.coll, limit=args.limit)
    analyser = LDAAnalyser(model=args.model, reader=reader, topic_file=args.topics)
    analyser.analyse()




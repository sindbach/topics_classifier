#!/usr/bin/env python

import pymongo
import logging

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
_logger = logging.getLogger(__name__)

class Reader(object):
    ''' Source reader object feeds other objects to iterate through a source. '''
    def __init__(self):
        ''' init '''
        exclude_stops = set(('.', '(', ')'))
        self.stop = set(stopwords.words('english')) - exclude_stops
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.p_stemmer = SnowballStemmer('english')
        self.wn_lemmatizer = WordNetLemmatizer()

    def prepare_words(self, text):
        ''' Prepare text 
        '''
        # lower cased all text
        text = text.lower()
        # tokenize
        texts = self.tokenizer.tokenize(text)
        # remove numbers
        texts = [t for t in texts if not t.isdigit()]
        # remove stopped words
        texts = [i for i in texts if not i in self.stop]
        # lemmatize 
        texts = [self.wn_lemmatizer.lemmatize(i) for i in texts]
        # stemmed
        texts = [self.p_stemmer.stem(i) for i in texts]  
        return texts

    def iterate(self):
        ''' virtual method '''
        pass

class MongoReader(Reader):
    def __init__(self, query={}, mongoURI="mongodb://localhost:27017", 
                 dbName='bow', collName='training', limit=None):
        ''' init
            :param query: MongoDB query
            :param mongoURI: mongoDB URI. default: localhost
            :param dbName: MongoDB database name. default: bow
            :param collName: MongoDB Collection name. default: training
        '''
        Reader.__init__(self)
        self.conn = pymongo.MongoClient(mongoURI)[dbName][collName]
        self.query = query
        self.limit = limit
        self.fields = ['title', 'components', 'question', 'answers']
        self.key_field = 'components'
        self.return_fields = ['title', 'question', 'answers']

    def get_value(self, value):
        if isinstance(value, list):
            return ' '.join(value)
        else:
            return str(value)

    def iterate(self):
        ''' Iterate through the source reader '''
        cursor = self.conn.find(self.query, self.fields)
        if self.limit: 
            cursor = cursor.limit(self.limit)
        for doc in cursor:
            try:
                content = ""
                for f in self.return_fields:
                    content +=" %s" % (self.get_value(doc.get(f)))
                texts = self.prepare_words(content)   
                yield doc.get(self.key_field), texts
            except Exception, ex: 
                raise Exception("Failed to prepare words: %s" % ex)


if __name__ == "__main__":
    pass
#!/bin/env python

import pymongo

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

class Reader(object):
    ''' Source reader object feeds other objects to iterate through a source. '''
    def __init__(self):
        ''' init '''
        self.stop = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.p_stemmer = PorterStemmer()

    def prepare_words(self, text):
        ''' Prepare text 
        '''
        # lower cased all text
        text = text.lower()
        # tokenize into words
        words = self.tokenizer.tokenize(text)
        # remove stopped words
        stopped_tokens = [i for i in words if not i in self.stop]
        # remove stemmed 
        texts = [self.p_stemmer.stem(i) for i in stopped_tokens]       
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
        self.fields = ['components', 'question', 'answers']
        self.key_field = 'components'
        self.return_fields = ['question', 'answers']

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
Topics Classifier 
----

Python scripts to classify topics/questions using statistics model [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation). 

Python libraries used:

* gensim
* nltk
* pymongo


**Data**

There is a class called `Reader` which reads and prepares the data for the modelling. By default, it will use a class called `MongoReader` which inherits from `Reader` and reads data from [MongoDB](http://mongodb.com). 

The document structure in MongoDB is structured like below:

```json
{
    "title" : "foo bar baz",
    "question": "foo baz ?",
    "components": [ "foo", "bar", .. ],
    "questions": [ "foo", "bar", .. ]
}
```

**Topics Modelling**

In order to build a topics model, you can execute the `lda_modeller.py` file. 

Example usage: 

```sh
lda_modeller.py --db topicsDB --coll training --topics ./topics.json --model ./example.model
```
The command above will read training data in MongoDB (running locally on `localhost:27017`) in database 'topicsDB' and collection 'training'. It will also output two files, an LDA model file called 'example.model' and a topics file called 'topics.json'

**Classifiy Topics** 

Once you have a topics file i.e. topics.json. You can open it up and assign topics title appropriately within the file. This would be the lookup file for the classifier. 

**Analyse Topics**

You  can execute `lda_analyser.py` to test the model file generated above. 

Example usage: 

```sh
lda_analyser.py --db topicsDB --coll data --limit 1 --topics ./topics.json --model ./example.model
```

The command above will read 1 document in MongoDB (running locally on `localhost:27017`) in database 'topicsDB' and collection 'data'. It will use the model file `example.model` to classify words and use the lookup file `topics.json` to translate the topics id to a useful title. 

**Docker Environment** 

The easiest way to start the development environment is by using docker-compose. Go to the docker directory, and type: 

```
docker-compose run topics
```

Which should run a docker environment with the code in `/home/ubuntu`. You can access MongoDB 3.2 from hostname `mongodb` port 27107. 



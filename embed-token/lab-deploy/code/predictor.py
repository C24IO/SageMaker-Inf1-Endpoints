# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import sys
import signal
import traceback
import torch
import torch_neuron
import transformers
from transformers import BertTokenizer
from transformers import BertModel
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import logging
import numpy as np
import io
import urllib.request
import pickle


try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

prefix = '/opt/program/'
tokenizer_path = os.path.join(prefix, 'bert-base-uncased-saved')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    tokenizer = None            # Where we keep the tokenizer when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""        

        if cls.model == None:
            cls.model = torch.jit.load(os.path.join(model_path, 'neuron_compiled_model.pt'))
            print('model loaded')
            #print(cls.model)          
            
        return cls.model
    
    @classmethod
    def get_tokenizer(cls):
        """Get the tokenizer object for this instance, loading it if it's not already loaded."""        

        if cls.tokenizer == None:
            cls.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print('tokenizer loaded')
            
        return cls.tokenizer

    @classmethod
    def predict(cls, input_sentences):
        
        print('predict')
        clf = cls.get_model()
        tokenizer = cls.get_tokenizer()
            
        encoded_sentence = tokenizer.encode_plus(input_sentences['sentence1'], input_sentences['sentence2'], max_length=128, pad_to_max_length=True, return_tensors="pt", truncation=True)
        encoded_sentence_tuple = encoded_sentence['input_ids'], encoded_sentence['attention_mask'], encoded_sentence['token_type_ids'] 

        #print(encoded_sentence_tuple)
        outputs = clf(*encoded_sentence_tuple)                
        #print(outputs)
        
        return outputs

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here
    #print(health)
    #health = True 

    status = 200 if health else 404
    
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    data = None

    #print(flask.request.content_type) #'application/json'
    #print(flask.request.data)
    
    incoming_data_structure = json.loads(flask.request.data)
    incoming_data_structure = json.loads(incoming_data_structure)
    
    #print(incoming_data_structure.keys())
    #print(type(incoming_data_structure))
    #print('\n~~~~~~~~~~~~~~``\n')    
    #input_data = {}
    #input_data['sentence1'] = sentence1
    #input_data['sentence2'] = sentence3
    #json_data = json.dumps(input_data)
    
    embedding = ScoringService.predict(incoming_data_structure)
    
    #print('\n~~~~~~~~~~~~~~``\n')
    #print(embedding)
    #print('\n~~~~~~~~~~~~~~``\n')        
    print('Scored')
    
    #print(type(embedding))
    
    raw_bytes_embedding = pickle.dumps(embedding)    
    result = raw_bytes_embedding    
    
    return flask.Response(response=result, status=200, mimetype='application/binary')

# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
import pickle
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

try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""        

        if cls.model == None:
            cls.model = torch.jit.load(os.path.join(model_path, 'neuron_compiled_model.pt'))
            print('model loaded')
            print(cls.model)
            
        return cls.model

    @classmethod
    def predict(cls, *input):
        
        print('predict')
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        
        #print(type(*input))
        print(*input)
        return clf(*input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here
    print(health)
    #health = True 

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    
    print(flask.request.content_type)
        
    pickled_bytes = flask.request.data
    encoded_sentence_tuple = pickle.loads(pickled_bytes)
    print(type(encoded_sentence_tuple))
    #input_statement = encoded_sentence['input_ids'], encoded_sentence['attention_mask'], encoded_sentence['token_type_ids']  
    print(encoded_sentence_tuple)
    embedding = ScoringService.predict(*encoded_sentence_tuple)
    print(type(embedding))
    print(embedding)
    raw_bytes_embedding = pickle.dumps(embedding)
    print('Scored')
         
    result = raw_bytes_embedding
    return flask.Response(response=result, status=200, mimetype='application/binary')

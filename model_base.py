#######
# Copyright 2020 Jian Zhang, All rights reserved
##
import logging
from abc import ABC, abstractmethod

class ModelBase(ABC):
    @abstractmethod
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        if load_trained:
            logging.info('load model from file ...')

    @abstractmethod
    def train(self, x):
        '''
        Train the model on one batch of data
        :param x: train data. For composer training, a single torch tensor will be given.
                  For your model training, you may use a sequence length of your choice.
                  But make sure this function can be called with input x containing sequences of length 100.
        :return: (mean) loss of the model on the batch
        '''
        pass

class ComposerBase(ModelBase):
    '''
    Class wrapper for a model that can be trained to generate music sequences.
    '''
    @abstractmethod
    def compose(self):
        '''
        Generate a music sequence which can play for at least 20 seconds
        :return: the generated sequence
        '''
        pass

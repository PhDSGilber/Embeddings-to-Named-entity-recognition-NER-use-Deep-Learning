import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tf2crf import CRF as crf6
import keras as k
from mwrapper import ModelWithCRFLoss, ModelWithCRFLossDSCLoss
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras import Sequential, Model, Input
from tf2crf import CRF, ModelWithCRFLoss
import keras as k
from mwrapper import ModelWithCRFLoss, ModelWithCRFLossDSCLoss
from gensim.models import KeyedVectors
from tensorflow_addons.optimizers import AdamW 
import numpy as np
import tensorflow_addons as tfa
from tensorflow_addons.text.crf_wrapper import CRFModelWrapper


class Model2NER:
  
  def __init__(self,vocab_size,embedding_dim,word2index,max_length):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.word2index = word2index
    self.max_length = max_length
  
  def create_embedding(self,type_embbeding,path_embedding):
       
       allowed_types = ["fasttext", "word2vec", "default"] 

       if type_embbeding not in allowed_types:
        raise ValueError(f"Invalid type_embbeding. Allowed types are: {allowed_types}")

       if (type_embbeding == "fasttext") or (type_embbeding == "word2vec"):

          self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
          model_embedding = KeyedVectors.load_word2vec_format(path_embedding)

          for word, i in self.word2index.items():
            if word in model_embedding:
              embedding_vector = model_embedding[word]
              if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
          
          self.embedding_matrix = [self.embedding_matrix]
        
       else:          
          self.embedding_matrix = None


  def create(self,bilstm_units,lstm_units,optimizer,metrics,summary: bool):

        input_l = Input(shape=(self.max_length,))

        model = Embedding(input_dim=self.vocab_size,
                  output_dim=self.embedding_dim,
                  input_length=self.max_length,
                  weights=self.embedding_matrix,  
                  mask_zero=False)(input_l)

        model = Bidirectional(LSTM(units=bilstm_units,
                     return_sequences=True,
                     dropout=0.5,
                     recurrent_dropout=0))(model)

        if lstm_units is not None:
          model = LSTM(units=lstm_units,
                       return_sequences=True, 
                       dropout=0.5,recurrent_dropout=0)(model)

        model  = Dense(units=11, activation='elu')(model)

        NER_model = Model(inputs=input_l, outputs=model)

        if summary:
          NER_model.summary()
          
        NER_model = CRFModelWrapper(NER_model, 11)
        NER_model.compile(optimizer=optimizer, metrics=metrics)

        # NER_model.compile(optimizer=optimizer, metrics=metrics) 
        # if summary:
        #     NER_model.summary()                

        # crf_layer = CRF(units=11)
        # output_l = crf_layer(model)

        # NER_model= Model(inputs=input_l, outputs=output_l)
        # if summary:
        #   NER_model.summary()
        # model_crf = ModelWithCRFLoss(NER_model, sparse_target=True)
        # model_crf.build(input_shape=(None, self.max_length))

        # if summary:
        #   model_crf.summary()

        # model_crf.compile(optimizer=optimizer, metrics=metrics)

        return NER_model

  def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'word2index': self.word2index,
            'max_length': self.max_length,
            'embedding_matrix': self.embedding_matrix,
            # Cualquier otro parámetro relevante
        }

  @classmethod
  def from_config(cls, config):
      return cls(**config)
#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy
import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
# from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
# from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *

config = tf.ConfigProto()
config.gpu_options.allow_growth= True
session = tf.Session(config=config)

KTF.set_session(session)

from Hypers import *
from Utils import *
from Preprocessing import *
from Generator import *
from Models import *

# In[5]:

data_root_path = ''
embedding_path = ''
KG_root_path = ''

news,news_index,category_dict,subcategory_dict,word_dict,content_dict,entity_dict = read_news(data_root_path)
news_title,news_vert,news_subvert,news_entity,news_content=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict,content_dict,entity_dict)

title_word_embedding_matrix, have_word = load_matrix(embedding_path, word_dict)
content_word_embedding_matrix, have_word = load_matrix(embedding_path, content_dict)

graph, EntityId2Index, EntityIndex2Id, entity_embedding = load_entity_metadata(KG_root_path)
news_entity_KG = load_news_entity(news,EntityId2Index,data_root_path)

news_entity_index = parse_zero_hop_entity(EntityId2Index,news_entity_KG,news_index,max_entity_num)
one_hop_entity = parse_one_hop_entity(EntityId2Index,EntityIndex2Id,news_entity_index,graph,news_index,max_entity_num)

train_session = read_train_clickhistory(news_index, data_root_path, 'MINDsmall_train/behaviors.tsv')
train_user = parse_user(news_index, train_session)
train_sess, train_user_id, train_label = get_train_input(news_index, train_session)

news_fetcher = NewsFetcher(news_title, news_content, news_vert, news_subvert, news_entity)  # 传递，设为共有

test_session = read_test_clickhistory_noclk(news_index, data_root_path, 'MINDsmall_dev/behaviors.tsv')
test_user = parse_user(news_index, test_session)
test_docids, test_userids, test_labels, test_bound = get_test_input(news_index, test_session)

train_generator = get_hir_train_generator(news_fetcher, news_entity_index, one_hop_entity, entity_embedding,
                                          train_user['click'], train_user_id, train_sess, train_label, 32)

test_generator = get_test_generator(test_docids,test_userids,news_fetcher,news_entity_index,one_hop_entity,entity_embedding,test_user['click'],64)

model, inter_model = create_model_new(title_word_embedding_matrix, content_word_embedding_matrix, entity_dict, category_dict, subcategory_dict)

print('model:', model)

model.fit_generator(train_generator, epochs=7, verbose=1)

predicted_label = inter_model.predict_generator(test_generator, verbose=1)
AUC, MRR, nDCG5, nDCG10 = evaluate(predicted_label, test_labels, test_bound)
print('AUC:', AUC, ' MRR:', MRR, ' nDCG5:', nDCG5, ' nDCG10:', nDCG10)

# model.summary()
# news_encoder Model:input:shape=(?,87) output=(0,400)
# news_scoring = news_encoder.predict_generator(news_generator, verbose=2)
# print('news_scoring:', news_scoring)
# test_user_generator = get_hir_user_generator(news_fetcher, news_entity_index, one_hop_entity, entity_embedding,
#                                           train_user['click'], train_user_id, train_sess, train_label, 16)
# test_user_scoring = user_encoder.predict_generator(test_user_generator, verbose=1)
# print('test_user_scoring:', test_user_scoring)
# AUC, MRR, nDCG5, nDCG10 = evaluate(test_impressions,news_scoring,test_user_scoring)
# print('AUC:', AUC, ' MRR:', MRR, ' nDCG5:', nDCG5, ' nDCG10:', nDCG10)
# dump_result(test_impressions, news_scoring, test_user_scoring)
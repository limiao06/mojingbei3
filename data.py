# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#z

import os
import numpy as np
import pandas as pd
import torch

"""
def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}

    dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path,
                                                 'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip() for line in
                                 open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in
                                 open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test
"""

def get_embeddings(embeddings_path):
    embeddings = {}
    with open(embeddings_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            embeddings[word] = np.array(list(map(float, vec.split())))
    print('Found {0} words with embeddingss'.format(
                len(embeddings)))
    return embeddings

def get_batch(questions_dict, batch, embeddings, random_flip=True):
    # batch is a np.array of [label, q1, q2]
    batch_size = len(batch)
    # random flip q1 and q2
    if random_flip:
        random_flag = np.random.randn(batch_size)
        batch = np.array([[l, q1, q1] if f>0 else [l, q2, q1] for (l, q1, q2), f in zip(batch, random_flag)], dtype=object)

    label_batch = batch[:,0]
    q1_keys_batch = batch[:,1]
    q2_keys_batch = batch[:,2]

    q1_sents = _get_sents(q1_keys_batch, questions_dict)
    q2_sents = _get_sents(q2_keys_batch, questions_dict)

    q1_batch, q1_len = _get_sents_embed(q1_sents, embeddings)
    q2_batch, q2_len = _get_sents_embed(q2_sents, embeddings)

    return label_batch, q1_batch, q1_len, q2_batch, q2_len

def _get_sents(sent_keys, questions_dict, feature="words"):
    sents = []
    for k in sent_keys:
        sents.append(questions_dict[k][feature].split(" "))
    return sents


def _get_sents_embed(sents, embeddings):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in sents])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(sents), 300))

    for i in range(len(sents)):
        for j in range(len(sents[i])):
            embed[j, i, :] = embeddings[sents[i][j]]

    return torch.from_numpy(embed).float(), lengths

def get_data(data_path):
    questions = pd.read_csv(os.path.join(data_path, "question.csv"))
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))

    dev_size = 9000
    dev = train.iloc[-dev_size:]
    dev = dev.reset_index(drop=True)
    train = train.iloc[0:-dev_size]
    train = train.reset_index(drop=True)

    questions_dict = {}
    for q, words, chars in list(questions.values):
        questions_dict[q] = {"words": words, "chars": chars}

    return questions_dict, train, dev, test
    

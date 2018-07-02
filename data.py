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

    label_batch = batch[:,0].astype(np.float32).reshape((-1,1))
    q1_keys_batch = batch[:,1]
    q2_keys_batch = batch[:,2]

    q1_sents = _get_sents(q1_keys_batch, questions_dict)
    q2_sents = _get_sents(q2_keys_batch, questions_dict)

    q1_batch, q1_len = _get_sents_embed(q1_sents, embeddings)
    q2_batch, q2_len = _get_sents_embed(q2_sents, embeddings)

    return label_batch, q1_batch, q1_len, q2_batch, q2_len

def get_test_batch(questions_dict, batch, embeddings, random_flip=False):
    # for test set
    # batch is a np.array of [label, q1, q2]
    batch_size = len(batch)
    # random flip q1 and q2
    if random_flip:
        random_flag = np.random.randn(batch_size)
        batch = np.array([[q1, q1] if f>0 else [q2, q1] for (q1, q2), f in zip(batch, random_flag)], dtype=object)

    q1_keys_batch = batch[:,0]
    q2_keys_batch = batch[:,1]

    q1_sents = _get_sents(q1_keys_batch, questions_dict)
    q2_sents = _get_sents(q2_keys_batch, questions_dict)

    q1_batch, q1_len = _get_sents_embed(q1_sents, embeddings)
    q2_batch, q2_len = _get_sents_embed(q2_sents, embeddings)

    return q1_batch, q1_len, q2_batch, q2_len


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

def get_word_vec(embeddings_path):
    vocab = []
    weight = []
    with open(embeddings_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            vocab.append(word)
            weight.append(np.array(list(map(float, vec.split()))))
    print('Found {0} words with embeddingss'.format(
                len(vocab)))
    vocab = dict([(w,i) for (i,w) in enumerate(vocab)])
    weight = np.array(weight)

    embed_dim = weight.shape[1]
    # add a padding symbol
    vocab['<p>'] = len(vocab)
    weight = np.append(weight, np.zeros((1, embed_dim)), axis=0)
    return vocab, weight

def get_batch_new(questions_dict, batch, vocab, random_flip=True):
    # batch is a np.array of [label, q1, q2]
    # return seq ids, not embeddings
    # not finished
    batch_size = len(batch)
    # random flip q1 and q2
    if random_flip:
        random_flag = np.random.randn(batch_size)
        batch = np.array([[l, q1, q1] if f>0 else [l, q2, q1] for (l, q1, q2), f in zip(batch, random_flag)], dtype=object)

    label_batch = batch[:,0].astype(np.float32).reshape((-1,1))
    q1_keys_batch = batch[:,1]
    q2_keys_batch = batch[:,2]

    q1_batch, q1_len = _get_sents_ids(q1_keys_batch, questions_dict, vocab)
    q2_batch, q2_len = _get_sents_ids(q2_keys_batch, questions_dict, vocab)

    return label_batch, q1_batch, q1_len, q2_batch, q2_len

def _get_sents_ids(sent_keys, questions_dict, vocab, feature="words"):
    sents = []
    sents_len = []
    for k in sent_keys:
        sent = questions_dict[k][feature].split(" ")
        sents.append(sent)
        sents_len.append(len(sent))

    sents_len = np.array(sents_len)
    padding_id = len(vocab) - 1
    max_len = np.max(sents_len)
    
    sent_ids = []

    for i, sent in enumerate(sents):
        sent_id = [vocab.get(w) for w in sent]
        sent_id += [padding_id] * (max_len - sents_len[i])
        sent_ids.append(sent_id)
    sent_ids = np.transpose(np.array(sent_ids, dtype=np.int64))
    return torch.LongTensor(sent_ids), sents_len

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


def expand_data(train):

    sim_sets = []
    reverse_dict = {}
    already_known_num = 0

    # get sim_sets
    for label, q1, q2 in tqdm(train):
        if label == 1:
            if q1 in reverse_dict and q2 not in reverse_dict:
                sim_set_id = reverse_dict[q1]
                sim_sets[sim_set_id].append(q2)
                reverse_dict[q2] = sim_set_id
            elif q1 not in reverse_dict and q2 in reverse_dict:
                sim_set_id = reverse_dict[q2]
                sim_sets[sim_set_id].append(q1)
                reverse_dict[q1] = sim_set_id
            elif q1 in reverse_dict and q2 in reverse_dict:
                sim_set_id1 = reverse_dict[q1]
                sim_set_id2 = reverse_dict[q2]
                if sim_set_id1 == sim_set_id2:
                    #print "%s, %s: already know they are same!" %(q1, q2)
                    already_known_num += 1
                    continue
                elif sim_set_id1 < sim_set_id2: # remove sim_set_id2
                    for qid in sim_sets[sim_set_id2]:
                        reverse_dict[qid] = sim_set_id1
                    sim_sets[sim_set_id1].extend(sim_sets[sim_set_id2])
                    sim_sets[sim_set_id2] = []
                else: # remove sim_set_id1
                    for qid in sim_sets[sim_set_id1]:
                        reverse_dict[qid] = sim_set_id2
                    sim_sets[sim_set_id2].extend(sim_sets[sim_set_id1])
                    sim_sets[sim_set_id1] = []   
                    
            else: # q1 q2 both not in reverse_dict
                sim_set_id = len(sim_sets)
                reverse_dict[q1] = sim_set_id
                reverse_dict[q2] = sim_set_id
                sim_sets.append([q1,q2])

    not_sim_set_dict = {}
    already_known_notsim_num = 0

    for label, q1, q2 in tqdm(train):
        if label == 0:
            if q1 not in reverse_dict:
                reverse_dict[q1] = len(sim_sets)
                sim_sets.append([q1])
            
            if q2 not in reverse_dict:
                reverse_dict[q2] = len(sim_sets)
                sim_sets.append([q2])
                
            sim_set_id1 = reverse_dict[q1]
            sim_set_id2 = reverse_dict[q2]
            
            if sim_set_id1 < sim_set_id2:
                tuple_key = (sim_set_id1, sim_set_id2)
            else:
                tuple_key = (sim_set_id2, sim_set_id1)
                
            if tuple_key not in not_sim_set_dict:
                not_sim_set_dict[tuple_key] = 1
            else:
                #print "%s, %s: already know they are not same!" %(q1, q2)
                already_known_notsim_num += 1


    # get same data
    same_data = []
    # deal with same tuples
    for sim_set in tqdm(sim_sets):
        set_size = len(sim_set)
        if set_size == 0: 
            continue
        for i in range(set_size):
            for j in range(i+1, set_size):
                same_data.append([1, sim_set[i], sim_set[j]] if sim_set[i] < sim_set[j]\
                           else [1, sim_set[j], sim_set[i]])

    # get not same data
    not_same_data = []
    # deal with not same tuples
    for sim_set_id1, sim_set_id2 in tqdm(not_sim_set_dict.keys()):
        for q1 in sim_sets[sim_set_id1]:
            for q2 in sim_sets[sim_set_id2]:
                not_same_data.append([0, q1, q2] if q1 < q2\
                           else [0, q2, q1])

    return (np.array(same_data, dtype=object), np.array(not_same_data, dtype=object))


def get_embeddings(embeddings_path):
    embeddings = {}
    with open(embeddings_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            embeddings[word] = np.array(list(map(float, vec.split())))
    print('Found {0} words with embeddingss'.format(
                len(embeddings)))
    return embeddings

def get_batch(questions_dict, batch, embeddings, random_flip=True, feature="words", batch_first=False):
    # batch is a np.array of [label, q1, q2]
    batch_size = len(batch)
    # random flip q1 and q2
    if random_flip:
        random_flag = np.random.randn(batch_size)
        batch = np.array([[l, q1, q1] if f>0 else [l, q2, q1] for (l, q1, q2), f in zip(batch, random_flag)], dtype=object)

    label_batch = batch[:,0].astype(np.float32).reshape((-1,1))
    q1_keys_batch = batch[:,1]
    q2_keys_batch = batch[:,2]

    q1_sents = _get_sents(q1_keys_batch, questions_dict, feature)
    q2_sents = _get_sents(q2_keys_batch, questions_dict, feature)

    q1_batch, q1_len = _get_sents_embed(q1_sents, embeddings, batch_first)
    q2_batch, q2_len = _get_sents_embed(q2_sents, embeddings, batch_first)

    return label_batch, q1_batch, q1_len, q2_batch, q2_len

def get_test_batch(questions_dict, batch, embeddings, random_flip=False, feature="words", batch_first=False):
    # for test set
    # batch is a np.array of [label, q1, q2]
    batch_size = len(batch)
    # random flip q1 and q2
    if random_flip:
        random_flag = np.random.randn(batch_size)
        batch = np.array([[q1, q1] if f>0 else [q2, q1] for (q1, q2), f in zip(batch, random_flag)], dtype=object)

    q1_keys_batch = batch[:,0]
    q2_keys_batch = batch[:,1]

    q1_sents = _get_sents(q1_keys_batch, questions_dict, feature)
    q2_sents = _get_sents(q2_keys_batch, questions_dict, feature)

    q1_batch, q1_len = _get_sents_embed(q1_sents, embeddings, batch_first)
    q2_batch, q2_len = _get_sents_embed(q2_sents, embeddings, batch_first)

    return q1_batch, q1_len, q2_batch, q2_len


def _get_sents(sent_keys, questions_dict, feature="words"):
    sents = []
    for k in sent_keys:
        sents.append(questions_dict[k][feature].split(" "))
    return sents


def _get_sents_embed(sents, embeddings, batch_first=False):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in sents])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(sents), 300))

    for i in range(len(sents)):
        for j in range(len(sents[i])):
            embed[j, i, :] = embeddings[sents[i][j]]
    if batch_first:
        embed = np.transpose(embed, (1,0,2))
    #print embed.shape
    return torch.from_numpy(embed).float(), lengths

def get_data_bk(data_path):
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

def get_data(data_path):
    questions = pd.read_csv(os.path.join(data_path, "question.csv"))
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))

    questions_dict = {}
    for q, words, chars in list(questions.values):
        questions_dict[q] = {"words": words, "chars": chars}

    return questions_dict, train, test

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

def get_batch_new(questions_dict, batch, vocab, random_flip=True, batch_first=False):
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

    q1_batch, q1_len = _get_sents_ids(q1_keys_batch, questions_dict, vocab, batch_first=batch_first)
    q2_batch, q2_len = _get_sents_ids(q2_keys_batch, questions_dict, vocab, batch_first=batch_first)

    return label_batch, q1_batch, q1_len, q2_batch, q2_len

def _get_sents_ids(sent_keys, questions_dict, vocab, feature="words", batch_first=False):
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
    if not batch_first:
        sent_ids = np.transpose(np.array(sent_ids, dtype=np.int64))
    else:
        sent_ids = np.array(sent_ids, dtype=np.int64)
    return torch.LongTensor(sent_ids), sents_len

def get_test_batch_new(questions_dict, batch, vocab, random_flip=False, batch_first=False):
    # for test set
    # batch is a np.array of [label, q1, q2]
    batch_size = len(batch)
    # random flip q1 and q2
    if random_flip:
        random_flag = np.random.randn(batch_size)
        batch = np.array([[q1, q1] if f>0 else [q2, q1] for (q1, q2), f in zip(batch, random_flag)], dtype=object)

    q1_keys_batch = batch[:,0]
    q2_keys_batch = batch[:,1]

    q1_batch, q1_len = _get_sents_ids(q1_keys_batch, questions_dict, vocab, batch_first=batch_first)
    q2_batch, q2_len = _get_sents_ids(q2_keys_batch, questions_dict, vocab, batch_first=batch_first)

    return q1_batch, q1_len, q2_batch, q2_len

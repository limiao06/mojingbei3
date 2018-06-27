# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_data, get_test_batch, get_embeddings
from mutils import get_optimizer
from models import MoJingNet
from tqdm import tqdm

def make_submission(predict_prob, output):
    with open(output, 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()


WORD_EMBEDDING_PATH = "mojing/word_embed.txt"


parser = argparse.ArgumentParser(description='Mojing inference')
# paths
parser.add_argument("--datapath", type=str, default='mojing/', help="mojing data path")
parser.add_argument("--modelpath", type=str, default='savedir/model.pickle', help="inference model path")
parser.add_argument("--output", type=str, default='output')
parser.add_argument("--batch_size", type=int, default=512)

"""
# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=1024, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=1, help="same or not")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
"""

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")


params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
questions_dict, train, dev, test = get_data(params.datapath)
word_vec = get_embeddings(WORD_EMBEDDING_PATH)

mojing_net = torch.load(params.modelpath)
print(mojing_net)

# cuda by default
mojing_net.cuda()

def inference():
    mojing_net.eval()
    results = []

    for i in tqdm(range(0, len(test), params.batch_size)):
        # prepare batch

        q1_batch, q1_len, q2_batch, q2_len = get_test_batch(questions_dict, 
            test[i:i + params.batch_size], word_vec)

        q1_batch, q2_batch = Variable(q1_batch).cuda(), Variable(q2_batch).cuda()

        # model forward
        prob = mojing_net.predict_prob((q1_batch, q1_len), (q2_batch, q2_len))
        results.extend(list(prob.reshape((-1))))
    return results

results = inference()
make_submission(results, params.output)


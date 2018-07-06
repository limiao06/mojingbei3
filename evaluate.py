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

from data import get_data, get_batch, get_embeddings
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


def main():
    parser = argparse.ArgumentParser(description='Mojing inference')
    # paths
    parser.add_argument("--datapath", type=str, default='mojing/', help="mojing data path")
    parser.add_argument("--modelpath", type=str, default='savedir/model.pickle', help="inference model path")
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--feature", type=str, default='words', help="words or chars")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")


    params, _ = parser.parse_known_args()

    # set gpu device
    torch.cuda.set_device(params.gpu_id)

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

    dev = dev.values

    mojing_net = torch.load(params.modelpath)
    #print(mojing_net)

    # cuda by default
    mojing_net.cuda()

    mojing_net.eval()
    correct = 0.

    for i in range(0, len(dev), params.batch_size):
        # prepare batch

        label_batch, q1_batch, q1_len, q2_batch, q2_len = get_batch(questions_dict, 
            dev[i:i + params.batch_size], word_vec, random_flip=False, feature=params.feature)

        q1_batch, q2_batch = Variable(q1_batch).cuda(), Variable(q2_batch).cuda()
        tgt_batch = Variable(torch.FloatTensor(label_batch)).cuda()

        # model forward
        output = mojing_net((q1_batch, q1_len), (q2_batch, q2_len))

        pred = output.data > 0
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().numpy()

    # save model
    eval_acc = round(100 * correct / len(dev), 4)
    print eval_acc

if __name__ == '__main__':
    main()

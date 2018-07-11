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


def main():
    parser = argparse.ArgumentParser(description='Mojing inference ensemble')
    # paths
    parser.add_argument("--datapath", type=str, default='mojing/', help="mojing data path")
    parser.add_argument("--modeldir", type=str, default='', help="dir contains inference models")
    parser.add_argument("--output", type=str, default='output')
    parser.add_argument("--batch_size", type=int, default=1024)
    
    parser.add_argument("--feature", type=str, default='words', help="words or chars")

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
    if params.feature == "words":
        WORD_EMBEDDING_PATH = "mojing/word_embed.txt"
    elif params.feature == "chars":
        WORD_EMBEDDING_PATH = "mojing/char_embed.txt"
    else:
        raise Exception("Unknown feature: %s" %(params.feature))

    questions_dict, train, test = get_data(params.datapath)
    word_vec = get_embeddings(WORD_EMBEDDING_PATH)

    test = test.values

    def inference(model):
        model.eval()
        results = []
        for i in tqdm(range(0, len(test), params.batch_size)):
            # prepare batch

            q1_batch, q1_len, q2_batch, q2_len = get_test_batch(questions_dict, 
                test[i:i + params.batch_size], word_vec, feature=params.feature)

            q1_batch, q2_batch = Variable(q1_batch).cuda(), Variable(q2_batch).cuda()

            # model forward
            probs = model.predict_prob((q1_batch, q1_len), (q2_batch, q2_len))
            probs = probs.data.cpu().numpy().reshape((-1))
            results.extend(list(probs))
        return results

    model_files = os.listdir(params.modeldir)

    results = []
    for f in model_files:
        mojing_net = torch.load(os.path.join(params.modeldir, f))
        results.append(inference(mojing_net))

    results = np.mean(results, axis=0)
    make_submission(results, params.output)

if __name__ == '__main__':
    main()

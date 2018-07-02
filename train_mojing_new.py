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

from data import get_data, get_batch_new, get_word_vec
from mutils import get_optimizer
from models import MoJingNet_e as MoJingNet


WORD_EMBEDDING_PATH = "mojing/word_embed.txt"


parser = argparse.ArgumentParser(description='Mojing training')
# paths
parser.add_argument("--datapath", type=str, default='mojing/', help="mojing data path")
parser.add_argument("--save_path", type=str, default='savedir/model.pickle')

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
parser.add_argument("--freeze_embed", type=float, default=0, help="freeze embedding layer or not")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=1024, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=1, help="same or not")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

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

vocab, weight = get_word_vec(WORD_EMBEDDING_PATH)

dev = dev.values

params.word_emb_dim = 300


"""
MODEL
"""
# model config
config_mojing_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    #'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
    'freeze_embed'   :  params.freeze_embed   ,
    'weight'         :  weight                ,
}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
mojing_net = MoJingNet(config_mojing_model)
print(mojing_net)

# loss
#weight = torch.FloatTensor(params.n_classes).fill_(1)
#loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn = nn.BCEWithLogitsLoss()
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(mojing_net.parameters(), **optim_params)

# cuda by default
mojing_net.cuda()
loss_fn.cuda()


"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    mojing_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    # need to do here
    permutation = np.random.permutation(len(train))

    train_perm = train.iloc[permutation].values

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(train), params.batch_size):
        # prepare batch
        """
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        """

        label_batch, q1_batch, q1_len, q2_batch, q2_len = get_batch_new(questions_dict, 
            train_perm[stidx:stidx + params.batch_size], vocab)

        q1_batch, q2_batch = Variable(q1_batch).cuda(), Variable(q2_batch).cuda()
        tgt_batch = Variable(torch.FloatTensor(label_batch)).cuda()


        k = q1_batch.size(1)  # actual batch size

        # model forward
        output = mojing_net((q1_batch, q1_len), (q2_batch, q2_len))

        pred = output.data > 0
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().numpy()
        assert len(pred) == len(train_perm[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        #all_costs.append(loss.data[0])
        all_costs.append(loss.data)
        words_count += (q1_batch.nelement() + q2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in mojing_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 4),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 4)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(train), 4)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, final_eval=False):
    mojing_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    print('\nVALIDATION : Epoch {0}'.format(epoch))

    for i in range(0, len(dev), params.batch_size):
        # prepare batch

        label_batch, q1_batch, q1_len, q2_batch, q2_len = get_batch_new(questions_dict, 
            dev[i:i + params.batch_size], vocab, random_flip=False)

        q1_batch, q2_batch = Variable(q1_batch).cuda(), Variable(q2_batch).cuda()
        tgt_batch = Variable(torch.FloatTensor(label_batch)).cuda()

        # model forward
        output = mojing_net((q1_batch, q1_len), (q2_batch, q2_len))

        pred = output.data > 0
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().numpy()

    # save model
    eval_acc = round(100 * correct / len(dev), 4)
    if final_eval:
        print('finalgrep : accuracy: {0}'.format(eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy:\
              {1}'.format(epoch, eval_acc))

    if epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            torch.save(mojing_net, params.save_path)
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, False)
    epoch += 1

# Run best model on test set.
del mojing_net
mojing_net = torch.load(params.save_path)

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, True)

## Save encoder instead of full model
#torch.save(mojing_net.encoder,
#           os.path.join(params.outputdir, params.outputmodelname + '.encoder'))

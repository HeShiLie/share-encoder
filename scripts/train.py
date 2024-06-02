# train也分两种mode
# 一个是pretrain 阶段

# import os
# print("当前工作目录:", os.getcwd())
import _init_paths
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse

import model.algorithms as algorithms
import dataset.datasets as datasets
import model.utils as utils

from dataset.loaders import InfiniteDataLoader, FastDataLoader, split_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model', type=str, default='erm', help='model name')
    parser.add_argument('--root', type=str, default='data', help='root directory of the dataset')
    parser.add_argument('--dataset', type=str, default='PACSDataset', help='dataset name')
    # parser.add_argument('--device', type=str, default='cuda', help='device to train on')
    parser.add_argument('--test_domains', type=list, default=[3], help='domains to test on')
    parser.add_argument('--max_pretraining_steps', type=int, default=2000, help='maximum number of pretraining steps')
    parser.add_argument('--chk_frq', type=int, default=None, help='checkpoint frequency')
    parser.add_argument('--holdout_fraction', type=float, default=0.1, help='fraction of the training data to hold out for validation')
    parser.add_argument('--pretraining_acc_threshold', type=float, default=0.9, help='pretraining accuracy threshold,\
                        if the accuracy is higher than this threshold, we will stop pretraining and begin the aligning')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    for key, value in vars(args).items():
        print(key,':', value)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # generate the hparams
    hparams = utils.get_hparams(args.model, args.dataset)
    for key, value in hparams.items():
        print(key,':', value)



    # load data
    root = args.root
    no_of_test_domains = args.test_domains

    dataset_load = getattr(datasets, args.dataset)(root, no_of_test_domains, hparams)
    #   turn into dataloaders
    #       split the dataset into train and test

    # load model (need to fulfil)
    algorithm = getattr(algorithms, args.model)
    
    model = algorithm(dataset_load.input_shape, dataset_load.num_classes, hparams)
    model.to(device)

    train_split = []
    test_split = []

    print('len of dataset_load:', len(dataset_load))
    for idx,domain in enumerate(dataset_load):
        domain_dataset = dataset_load[idx]
        train, test = split_dataset(domain_dataset, int(len(domain_dataset) * (1 - args.holdout_fraction)), seed=args.seed) 
        train_split.append(train)
        test_split.append(test)
    #       form the dataloaders
    train_loaders = [InfiniteDataLoader(train,
                                         None,
                                           hparams['batch_size'],
                                             1) 
                                             for i, train in enumerate(train_split) if i not in args.test_domains]
    eval_loaders = [FastDataLoader(test,
                                      32,
                                        1) 
                                        for test in (train_split+test_split)]
    
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(train_split))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(test_split))]
    
    print('already load the data')

    
    # pretraining steps
    max_pretraining_steps = args.max_pretraining_steps
    pre_steps = min(max_pretraining_steps,dataset_load.STEPS)
    print('pre_steps:', pre_steps)

    current_total_accuracy = 0

    # form the iterators
    checkpoint_freq = args.chk_frq or dataset_load.CHECKPOINT_FREQ
    train_minibatches_iterator = zip(*train_loaders)

    # pretrain the model without regarding the domains, that is to trains the model on all the source domains
    for epoch in range(pre_steps):
        # load minibatches
        minibatches_device = [(x.to(device), y.to(device))
        for x,y in next(train_minibatches_iterator)]

        # update the model
        loss_val = model.update(minibatches_device)

        # check the accuracy of the model
        if (epoch % checkpoint_freq == 0) or (epoch == pre_steps - 1):
            # total_acc function needs to be fulfilled
            current_total_accuracy = utils.total_acc(model, eval_loaders, args.test_domains)

            if current_total_accuracy > args.pretraining_acc_threshold:
                print('Pretraining accuracy is higher than the threshold, begin aligning')
                break
            
        print('Epoch:', epoch, 'Loss:', loss_val, 'Total Accuracy:', current_total_accuracy)

    # train the model through an aligning way
            




import numpy as np
import torch

def get_hparams(model_name, dataset_name):
    '''
    model_name: str
    dataset_name: str
    '''
    if model_name == 'erm':
        hparams = {
            'resnet18': True,
            'resnet_dropout': False,
            'lr': 1e-3,
            'lambda': 0,
            'aug_transform': False
        }
    if dataset_name == 'PACSDataset':
        hparams['aug_transform'] = True
        hparams['batch_size'] = 16
    
    return hparams

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

def total_acc(model, eval_loaders, test_domains, weights = None, device = 'cuda'):
    '''
    model: nn.Module
    eval_loaders: list of torch.utils.data.DataLoader
    test_domains: list of int, indicating the domains to test on, and the total accuracy will be calculated on only the source domains
    '''
    # delete the test_domains in eval_loaders
    domain_num = len(eval_loaders)//2

    del_idx_train, del_idx_test = [], []
    del_idx_train = test_domains
    del_idx_test = [i+domain_num for i in test_domains]

    eval_loaders_new = [eval_loaders[i] for i in range(len(eval_loaders)) if i not in del_idx_train+del_idx_test]

    # calculate the total accuracy
    # calculate the number of samples in each loader
    num_samples_list = [len(loader) for loader in eval_loaders_new]
    # evaluate the model on each loader
    acc_list = []
    for loader in eval_loaders_new:
        acc = accuracy(model, loader, weights, device)
        acc_list.append(acc)

    # calculate the total accuracy only on the test loaders(the latter half of the eval_loaders)
    num_train_domains = len(eval_loaders_new)//2
    total_num_samples = sum(num_samples_list[num_train_domains:])
    total_correct = sum([acc*num_samples for acc, num_samples in zip(acc_list[num_train_domains:], num_samples_list[num_train_domains:])])
    total_acc = total_correct/total_num_samples

    return total_acc






    
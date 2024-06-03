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
            'lambda': 0.1,
            'aug_transform': False
        }
    if dataset_name == 'PACSDataset':
        hparams['aug_transform'] = True
        hparams['batch_size'] = 32
    
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

# defirm the function to get the mu and sigma from the domains
def get_mu_sigma(model, source_loader, device):
    '''
    model: nn.Module
    source_loader: torch.utils.data.DataLoader
    '''
    # calculate the mu and sigma
    model.eval()
    feature_list = []
    total_num = 0
    with torch.no_grad():
        for x, y in source_loader:
            batch_weights = torch.ones(len(x))
            batch_weights = batch_weights.to(device)
            x = x.to(device)
            feature = model.featurelizer(x)
            feature_list.append(feature)
            total_num += batch_weights.sum().item()

    feature_list = torch.cat(feature_list, dim=0)
    mu = feature_list.mean(dim=0)
    sigma2 = feature_list.var(dim=0)
    model.train()

    return mu, sigma2, total_num

def get_common_mu_sigma(list_mu_sigma_num):
    '''
    list_mu_sigma_num: list of tuple, each tuple contains the mu, sigma and the number of samples in the domain

    where mu and sigma are torch.Tensor of shape (n_features,)

    '''
    # total_N
    total_N = sum([num for mu, sigma, num in list_mu_sigma_num])

    # calculate the common mu and sigma
    # common_mu (mu_0) = (N_1*mu_1 + N_2*mu_2 + ... + N_n*mu_n)/total_N
    common_mu = sum([num*mu for mu, sigma, num in list_mu_sigma_num])/total_N

    # common_sigma = [[(N_1-1)*sigma_1**2 + N_1(mu_0 - mu_1)**2] + [(N_2-1)*sigma_2**2 + N_2(mu_0 - mu_2)**2] + ... + [(N_n-1)*sigma_n**2 + N_n(mu_0 - mu_n)**2]]/(total_N-1)
    common_sigma2 = sum([(num-1)*sigma**2 + num*(mu - common_mu)**2 for mu, sigma, num in list_mu_sigma_num])/(total_N-1)

    return common_mu, common_sigma2





    
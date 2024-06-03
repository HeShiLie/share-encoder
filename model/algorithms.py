import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .networks import ResNet

# hparams require 'resnet18', 'resnet_dropout', 'lr', 'lambda', 'aug_transform'
#erm
class erm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams):
        super(erm, self).__init__()
        self.hparams = hparams
        self.featurelizer = ResNet(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurelizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurelizer, self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.hparams['lr'])

    def forward(self, x):
        return self.network(x)

    def update(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        loss = F.cross_entropy(self.forward(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    
    def get_feature(self, x):
        return self.featurelizer(x)
    
    def update_align(self, minibatches, shift_list, scale_list, hparams):
        all_x = [x for x, y in minibatches]
        all_y = [y for x, y in minibatches]

        all_feature = [self.get_feature(x) for x in all_x]
        all_feature_affined = [(feature-shift)*scale for feature, shift, scale in zip(all_feature, shift_list, scale_list)]

        mse_loss = F.mse_loss(torch.cat(all_feature_affined), torch.cat(all_feature))

        loss = F.cross_entropy(self.forward(torch.cat(all_x)), torch.cat(all_y)) + hparams['lambda']*mse_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

# aligning
class shares_erm(erm):
    def __init__(self, input_shape, num_classes, hparams):
        super(shares_erm, self).__init__()

        self.final_bn = nn.BatchNorm1d(self.featurelizer.n_outputs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = hparams['lr'])
        
    def train_forward(self, x, tempfile = 'tempfile.pth',device = 'cuda'):
        feature = self.featurelizer(x)
        y = self.classifier(feature)
        # save feature to at the end of the tempfile

        torch.save(torch.cat([torch.load(tempfile).to(device),feature],dim=1), tempfile)

        return y
    
    def test_forward(self, x):
        return self.network(x)

    def forward(self, x, mode = 'train'):
        if mode == 'train':
            return self.train_forward(self, x)
        elif mode == 'test':
            return self.test_forward(self, x)
        else:
            raise ValueError('mode must be either train or test')

    def update(self, minibatches, common_mu_sigma, specific_mu_sigma):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        feature = self.featurelizer(all_x)
        revised_feature = (feature-specific_mu_sigma['mu']+common_mu_sigma['mu'])*common_mu_sigma['sigma']/specific_mu_sigma['sigma']

        y_hat = self.classifier(feature)
        mse_loss = F.mse_loss(revised_feature, feature)
        classification_loss = F.cross_entropy(y_hat, all_y)
        loss = self.hparams['lambda']*mse_loss + classification_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'mse_loss': mse_loss.item(), 'classification_loss': classification_loss.item()}
        
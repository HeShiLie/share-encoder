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
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr = self.hparams['lr'])
        self.optimizer_featurelizer = torch.optim.Adam(self.featurelizer.parameters(), lr = self.hparams['lr'])

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

        mse_loss = hparams['lambda']*F.mse_loss(torch.cat(all_feature_affined), torch.cat(all_feature))

        ce_loss = F.cross_entropy(self.forward(torch.cat(all_x)), torch.cat(all_y))

        loss = mse_loss + ce_loss

        self.optimizer_featurelizer.zero_grad()
        self.optimizer_classifier.zero_grad()

        ce_loss.backward(retain_graph=True)
        self.optimizer_classifier.step()

        mse_loss.backward()
        self.optimizer_featurelizer.step()

        # self.optimizer.zero_grad()
        # self.optimizer_featurelizer.zero_grad()

        return {'ce_loss': ce_loss.item(), 'mse_loss': mse_loss.item()}

# aligning while frozening the resnet architecture
class share_erm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams):
        super(share_erm, self).__init__()
        self.hparams = hparams
        self.featurelizer = ResNet(input_shape, self.hparams)
        self.middle_fc_layers = nn.Sequential(nn.Linear(self.featurelizer.n_outputs, self.featurelizer.n_outputs), 
                                              nn.ReLU(), 
                                              nn.Linear(self.featurelizer.n_outputs, self.featurelizer.n_outputs), 
                                              nn.ReLU(),
                                              nn.Linear(self.featurelizer.n_outputs, self.featurelizer.n_outputs),
                                              nn.ReLU())
        self.classifier = nn.Linear(self.featurelizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurelizer, self.middle_fc_layers,self.classifier)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.hparams['lr'])
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr = self.hparams['lr'])
        self.optimizer_midlayers = torch.optim.Adam(self.middle_fc_layers.parameters(), lr = self.hparams['lr'])

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
    
    def get_feature_after_midlayers(self, x):
        return self.middle_fc_layers(self.featurelizer(x))
    
    def update_align(self, minibatches, shift_list, scale_list, hparams):
        all_x = [x for x, y in minibatches]
        all_y = [y for x, y in minibatches]

        all_feature = [self.get_feature_after_midlayers(x) for x in all_x]
        all_feature_affined = [(feature-shift)*scale for feature, shift, scale in zip(all_feature, shift_list, scale_list)]

        mse_loss = hparams['lambda']*F.mse_loss(torch.cat(all_feature_affined), torch.cat(all_feature))

        ce_loss = F.cross_entropy(self.forward(torch.cat(all_x)), torch.cat(all_y))

        # loss = mse_loss + ce_loss

        self.middle_fc_layers.zero_grad()
        self.optimizer_classifier.zero_grad()

        ce_loss.backward(retain_graph=True)
        self.optimizer_classifier.step()

        mse_loss.backward()
        self.middle_fc_layers.step()

        # self.optimizer.zero_grad()
        # self.optimizer_featurelizer.zero_grad()

        return {'ce_loss': ce_loss.item(), 'mse_loss': mse_loss.item()}
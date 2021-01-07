import os
import click
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import utils
from global_constants import income_const
from model import IncomeClassifier, IncomeClassifierConstants
from dataset import FeatDataset
from focal_loss import FocalLoss

def predict(model_const, dataset, exp_const):
    model = IncomeClassifier(model_const)
    loaded_object = torch.load(model_const.model_path)
    model.load_state_dict(loaded_object['State'])

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=exp_const.num_workers)
    test_value = next(iter(dataloader))
    
    model.eval()
    test_x = test_value['feat']
    test_y = test_value['label']
    logits, probs = model(test_x)
    pred_label = probs[:,1] > 0.5

    print("test_x: {}\npred_y: {}\ntest_y: {}".format(test_x, pred_label, test_y.bool()))
    

@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    show_default=True,
    type=str,
    help='Name of the experiment')
@click.option(
    '--num_hidden_blocks',
    default=2,
    show_default=True,
    type=int,
    help='Number of hidden blocks in the classifier')
def main(**kwargs):
    exp_const = utils.Constants()
    exp_const.exp_dir = os.path.join(income_const['exp_dir'],kwargs['exp_name'])
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 256
    exp_const.num_workers = 1

    model_const = IncomeClassifierConstants()
    model_const.num_hidden_blocks = kwargs['num_hidden_blocks']
    model_const.model_path = os.path.join(exp_const.model_dir,'best_model')

    dataset = FeatDataset('test')

    predict(model_const,dataset,exp_const)


if __name__=='__main__':
    main()
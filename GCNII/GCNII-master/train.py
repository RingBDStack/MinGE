from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid
import time
start = time.time()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--split-seed', type=int, default=42, help='seed for data splits (train/test/val)')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.08, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.02, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=16, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=100, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='airport', help='dateset')
parser.add_argument('--task', default='lp', help='tast nc or lp')
parser.add_argument('--test-prop', type=float, default=0.3, help='alpha_l')
parser.add_argument('--val-prop', type=float, default=0.1, help='proportion of test edges for link prediction')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.3, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--r', type=int, default=2, help='fermi-dirac decoder parameter for lp')
parser.add_argument('--t', type=int, default=1, help='fermi-dirac decoder parameter for lp')
parser.add_argument('--manifold',default='Euclidean', help='which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]')
parser.add_argument('--c', type=float, default=1.0, help='hyperbolic radius, set to None for trainable curvature.')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed) 
# Load data
if args.task == 'nc':
    adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.split_seed,args.data)
elif args.task == 'lp':
    adj, features, labels,idx_train,idx_val,idx_test,lpdata = load_citation_lp(args,args.data)
    args.nb_false_edges = len(lpdata['train_edges_false'])
    args.nb_edges = len(lpdata['train_edges'])
# print(lpdata.keys())
print('##############################')
print(idx_train.shape)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

if args.task == 'lp':
    model = LPGCNII(nfeat=features.shape[1],
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant,
                args=args).to(device)
elif args.task == 'nc':             
    model = GCNII(nfeat=features.shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    variant=args.variant).to(device)

optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    if args.task == 'lp':
         train_metrics = model.compute_metrics(output, lpdata, 'train')
         acc_train = train_metrics['roc']
         loss_train = train_metrics['loss']
    else:
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        if args.task == 'lp':
            val_metrics = model.compute_metrics(output, lpdata, 'val')
            acc_val = val_metrics['roc']
            loss_val = val_metrics['loss']
        else:
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        if args.task == 'lp':
            test_metrics = model.compute_metrics(output, lpdata, 'test')
            loss_test = test_metrics['loss']
            acc_test = test_metrics['roc']
        else:
            loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
            acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()
    
t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra,acc_tra = train()
    loss_val,acc_val = validate()
    if(epoch+1)%1 == 0: 
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            'acc:{:.2f}'.format(acc_tra*100),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100))
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]
end = time.time()
print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
print('running time %s Second'%(end -start))
    






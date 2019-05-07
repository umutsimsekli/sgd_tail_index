# some useful functions

import math
import torch
from torchvision import datasets, transforms
import numpy as np


def get_layerWise_norms(net):
    w = []
    g = []
    for p in net.parameters():    
        if p.requires_grad:
            w.append(p.view(-1).norm())
            g.append(p.grad.view(-1).norm())
    return w, g

def linear_hinge_loss(output, target):
    binary_target = output.new_empty(*output.size()).fill_(-1)
    for i in range(len(target)):
        binary_target[i, target[i]] = 1
    delta = 1 - binary_target * output
    delta[delta <= 0] = 0
    return delta.mean()

def get_grads(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.grad.view(-1))
    grad_flat = torch.cat(res)
    return grad_flat

# Corollary 2.4 in Mohammadi 2014
def alpha_estimator(m, X):
    # X is N by d matrix
    N = len(X)
    n = int(N/m) # must be an integer
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff

# Corollary 2.2 in Mohammadi 2014
def alpha_estimator2(m, k, X):
    # X is N by d matrix
    N = len(X)
    n = int(N/m) # must be an integer
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps)
    X_log_norm = torch.log(X.norm(dim=1) + eps)

    # This can be implemented more efficiently by using 
    # the np.partition function, which currently doesn't 
    # exist in pytorch: may consider passing the tensor to np
    
    Yk = torch.sort(Y_log_norm)[0][k-1]
    Xk = torch.sort(X_log_norm)[0][m*k-1]
    diff = (Yk - Xk) / math.log(m)
    return 1 / diff

def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)

def get_data(args):

    # mean/std stats
    if args.dataset == 'cifar10':
        data_class = 'CIFAR10'
        num_classes = 10
        stats = {
            'mean': [0.491, 0.482, 0.447], 
            'std': [0.247, 0.243, 0.262]
            } 
    elif args.dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        stats = {
            'mean': [0.5071, 0.4867, 0.4408] , 
            'std': [0.2675, 0.2565, 0.2761]
            } 
    elif args.dataset == 'mnist':
        data_class = 'MNIST'
        num_classes = 10
        stats = {
            'mean': [0.1307], 
            'std': [0.3081]
            }
    else:
        raise ValueError("unknown dataset")

    # input transformation w/o preprocessing for now

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
        ]
        
    # get tr and te data with the same normalization
    tr_data = getattr(datasets, data_class)(
        root=args.path, 
        train=True, 
        download=True,
        transform=transforms.Compose(trans)
        )

    te_data = getattr(datasets, data_class)(
        root=args.path, 
        train=False, 
        download=True,
        transform=transforms.Compose(trans)
        )

    # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train, 
        shuffle=False,
        )

    train_loader_eval = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_eval, 
        shuffle=False,
        )

    test_loader_eval = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval, 
        shuffle=False,
        )

    return train_loader, test_loader_eval, train_loader_eval, num_classes

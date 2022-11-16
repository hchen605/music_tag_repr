# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
import pandas as pd
import soundfile as sound
import librosa
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate
import pycuda
from pycuda import compiler
import pycuda.driver as drv
from utilities import *

drv.init()
torch.cuda.empty_cache()
print("%d device(s) found." % drv.Device.count())
print('== check GPU ==')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print('current device: ', torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
#print(torch.cuda.get_device_name(1))
#torch.cuda.set_device(1)
#print(torch.cuda.current_device())
#print(torch.cuda.device_count() )

#print(torch.cuda.memory_summary())



print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "openmic", "mtg"])

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

args = parser.parse_args()

DATA_ROOT = '/home/hc605/dataset/MTT/'



print('loading data ..,')
#x_train = np.load(DATA_ROOT + 'x_train_1.npy')
#x_train_2 = np.load(DATA_ROOT + 'x_train_2.npy')

x_train_1 = np.load(DATA_ROOT + 'x_train_10s.npy')
x_train_2 = np.load(DATA_ROOT + 'x_train_10s_middle.npy')
x_train = np.vstack((x_train_1,x_train_2))
#x_train = x_train[:1905]
print('Train: ', x_train.shape)
x_test = np.load(DATA_ROOT + 'x_test_10s.npy')
#x_test = np.load(DATA_ROOT + 'x_test.npy')
print('Test: ', x_test.shape)
print('----')
print('loading label ..,')

y_train = np.load(DATA_ROOT + 'y_train.npy')
y_train = np.vstack((y_train,y_train))
y_test = np.load(DATA_ROOT + 'y_test.npy')

y_mask_train = np.full((30488,50), True)
#y_mask_train = np.full((1905,50), True)
#y_mask_train = np.full((19898,56), True)#aug
y_mask_test = np.full((4332,50), True)


# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')
    # dataset spectrogram mean and std, used to normalize the input
    norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526], 'openmic': [0,0], 'mtg': [0,0],}
    target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128, 'openmic':1024, 'mtg':1024}
    # if add noise for data augmentation, only use for speech commands
    noise = {'audioset': False, 'esc50': False, 'speechcommands':True, 'openmic':False, 'mtg':False}

    audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1],
                  'noise':noise[args.dataset], 'skip_norm': True}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False, 'skip_norm': True}

    
    print('data loading ...')
    train_loader = torch.utils.data.DataLoader(
        dataloader.MTGDataset(x_train, y_train, y_mask_train, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.MTGDataset(x_test, y_test, y_mask_test, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128, input_tdim=target_length[args.dataset], imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')


for para in audio_model.parameters():
    para.requires_grad = False
    #print(para)    
params = audio_model.state_dict()
print(params.keys())

print('Trainable parameters')
for name, param in audio_model.named_parameters():
    if not param.requires_grad and 'mlp_head' in name:
        param.requires_grad = True
        print(name)
    if not param.requires_grad and 're_cnn' in name:
        param.requires_grad = True
        print(name)
    if not param.requires_grad and 'repr' in name:
        param.requires_grad = True
        print(name)
    if not param.requires_grad and 'mean' in name:
        param.requires_grad = True
        print(name)
    if not param.requires_grad and 'std' in name:
        param.requires_grad = True
        print(name)
    if not param.requires_grad and 'delta' in name:
        param.requires_grad = True  
        print(name)
    if not param.requires_grad and 'unet' in name:
        param.requires_grad = True  
        print(name)

    
print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
#print(torch.cuda.memory_summary())
train(audio_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # best model on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])


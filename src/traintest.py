# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import json

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    
    print('Total parameter number is : {:.5f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.5f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    #optim = optim.Adam(itertools.chain(model.parameters(), gamma), lr=1e-3)
    
    # dataset specific settings
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    if args.dataset == 'audioset':
        if len(train_loader.dataset) > 2e5:
            print('scheduler for full audioset is used')
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1)
        else:
            print('scheduler for balanced audioset is used')
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5, last_epoch=-1)
        main_metrics = 'mAP'
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = True
    elif args.dataset == 'esc50':
        print('scheduler for esc-50 is used')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
        main_metrics = 'acc'
        loss_fn = nn.CrossEntropyLoss()
        warmup = False
    elif args.dataset == 'speechcommands':
        print('scheduler for speech commands is used')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
        main_metrics = 'acc'
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = False
    elif args.dataset == 'openmic': 
        print('scheduler for openmic is used')
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5, last_epoch=-1)
        main_metrics = 'acc'
        #loss_fn = nn.CrossEntropyLoss()
        #loss_fn = nn.BCELoss()
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = False 
    elif args.dataset == 'mtg': 
        print('scheduler for mtg is used')
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(2,20)), gamma=0.85)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5, last_epoch=-1)
        main_metrics = 'acc'
        #loss_fn = nn.CrossEntropyLoss()
        #loss_fn = nn.BCELoss()
        pos_weight = torch.tensor([33.61148649,37.68560606,27.4077135,24.62623762,51.5492228, 37.26217228, 65.45394737,  27.03532609,  52.08900524,  58.52352941, 28.50716332,  16.04677419,  23.19114219,  29.3480826,   39.32411067, 51.02051282,  19.97791165,  19.93787575,  11.60910152,  16.55407654, 160.46774194,  13.33646113,  45.22272727,  91.27522936,  58.86982249, 127.55128205,  10.73247033, 101.52040816, 101.52040816,  72.62043796, 17.73440285,  20.18052738,  26.60160428,  43.63596491,  16.63712375, 26.74462366,  48.06280193,  72.62043796,  78.33858268,  26.46010638, 93.85849057,  13.03931848,  71.57553957,  27.4077135,   26.53066667, 168.62711864,  33.72542373,  47.83173077,  23.97349398,  27.48342541, 49.49751244,  38.11877395,  62.57232704, 111.78651685, 46.06018519, 22.15812918]).to(device, non_blocking=True)
                                   #33.61148649,37.68560606,27.4077135,24.62623762,51.5492228, 37.26217228, 65.45394737,  27.03532609,  52.08900524,  58.52352941, 28.50716332,  16.04677419,  23.19114219,  29.3480826,   39.32411067, 51.02051282,  19.97791165,  19.93787575,  11.60910152,  16.55407654, 160.46774194,  13.33646113,  45.22272727,  91.27522936,  58.86982249, 127.55128205,  10.73247033, 101.52040816, 101.52040816,  72.62043796, 17.73440285,  20.18052738,  26.60160428,  43.63596491,  16.63712375, 26.74462366,  48.06280193,  72.62043796,  78.33858268,  26.46010638, 93.85849057,  13.03931848,  71.57553957,  27.4077135,   26.53066667, 168.62711864,  33.72542373,  47.83173077,  23.97349398,  27.48342541, 49.49751244,  38.11877395,  62.57232704, 111.78651685, 46.06018519, 22.15812918]).to(device, non_blocking=True)
        #loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = False 
    else:
        raise ValueError('unknown dataset, dataset should be in [audioset, speechcommands, esc50]')
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    args.loss_fn = loss_fn

    epoch += 1
    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels, mask) in enumerate(train_loader):

            
            
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            #print('labels.shape: ', labels.size)
            mask = mask.to(device, non_blocking=True)
            

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                audio_output = audio_model(audio_input)
                #print('audio_output.shape: ', audio_output.size)
                #print(audio_output[mask])
                #print(labels[mask])
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output[mask], torch.argmax(labels[mask].long(), axis=1))
                else:
                    loss = loss_fn(audio_output[mask], labels[mask])

            # optimization if amp is not used
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)

        # ensemble results
        #cum_stats = validate_ensemble(args, epoch)
        #cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        #cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        #cum_acc = cum_stats[0]['acc']

        #mAP = np.mean([stat['AP'] for stat in stats])
        #mAUC = np.mean([stat['auc'] for stat in stats])
        #acc = stats[0]['acc']
        acc = stats['F1_macro']
        precision = stats['precision']
        pr_auc = stats['pr_auc']
        roc_auc = stats['auroc']
        
        #print(stats)
        print('average F1: ', np.mean(acc))
        print('average precision: ', np.mean(precision))
        print('PR AUC: ', np.mean(pr_auc))
        print('ROC AUC: ', np.mean(roc_auc))
        
        #print(acc)

        #middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        #middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        #average_precision = np.mean(middle_ps)
        #average_recall = np.mean(middle_rs)
        '''
        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        '''
        #print("AUC: {:.6f}".format(mAUC))
        #print("Avg Precision: {:.6f}".format(average_precision))
        #print("Avg Recall: {:.6f}".format(average_recall))
        #print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        '''
        if main_metrics == 'mAP':
            result[epoch-1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_mAP, cum_mAUC, optimizer.param_groups[0]['lr']]
        else:
            result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_acc, cum_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        '''
        print('validation finished')
        
        '''
        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch
        '''
        '''       
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP
        '''
        
        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    if args.dataset == 'audioset':
        if len(train_loader.dataset) > 2e5:
            stats=validate_wa(audio_model, test_loader, args, 1, 5)
        else:
            stats=validate_wa(audio_model, test_loader, args, 6, 25)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)
        wa_result = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC)]
        print('---------------Training Finished---------------')
        print('weighted averaged model results')
        print("mAP: {:.6f}".format(mAP))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        np.savetxt(exp_dir + '/wa_result.csv', wa_result)

def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    A_mask = []
    n_classes = 50
    all_y_true = torch.Tensor(0, n_classes)
    all_y_mask = torch.BoolTensor(0, n_classes)
    all_predictions = torch.Tensor(0, n_classes)
    all_t_predictions = torch.Tensor(0, n_classes)
    
    with torch.no_grad():
        for i, (audio_input, labels, mask) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()
            
            
            A_predictions.append(predictions)
            A_targets.append(labels)
            A_mask.append(mask)
            
            all_y_true = torch.cat((all_y_true, labels))
            all_y_mask = torch.cat((all_y_mask, mask))
            all_predictions = torch.cat((all_predictions, predictions))

            # compute the loss
            labels = labels.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                #loss = args.loss_fn(audio_output, labels)
                loss = loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        #stats = calculate_stats(audio_output, target)
        #np.save('./log/TL/pred_TL_pure_ep'+str(epoch)+'.npy', all_predictions)
        #np.save('./log/TL/Y_true_ep'+str(epoch)+'.npy', all_y_true)
        metrics = metric_fn_openmic(all_y_true, all_y_mask, all_predictions)
        # jsonify metrics and write to json as well for manual inspection
        js = {}
        for key, val in metrics.items():
            if not np.ndim(val) == 0:
                js[key] = val.tolist()
            else:
                js[key] = val
        js['F1 macro avg'] = np.mean(metrics['F1_macro'])
        #js['Best Epoch'] = best_epoch
        #json.dump(js, open(f'metrics.json', 'w'))
        
        # save the prediction here
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return metrics, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats

def validate_wa(audio_model, val_loader, args, start_epoch, end_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)

    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
        if args.save_model == False:
            os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    audio_model.load_state_dict(sdA)

    torch.save(audio_model.state_dict(), exp_dir + '/models/audio_model_wa.pth')

    stats, loss = validate(audio_model, val_loader, args, 'wa')
    return stats
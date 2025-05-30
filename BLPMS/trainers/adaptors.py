from args import args
import torch
from torch import optim
import math

import numpy as np
import pathlib

from models import module_util
from utils.utils import kth_elt
from functools import partial
from utils import my_utils
from torch import nn
import trainers

def adapt_test( model, data_loader, split, alphas=None, ):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if args.text_exp:
                data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                output = model(data, mask)["preds"]
            else:
                data, target = batch[0].to(args.device), batch[1].to(args.device)
                output = model(data)["preds"]
            pred = output.argmax( dim=1, keepdim=True )
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += pred.numel()

        acc = float(correct) / float(total)
    print(f"{split} Accuracy: {acc:.6f}\tCorrect: {correct}\tTotal: {total}")  
    return acc

# gt means ground truth task -- corresponds to GG
def gt( model, writer, data_loader, num_tasks_learned, task, split ):
    model.zero_grad()
    model.train()

    # changed requires_grad to False.
    alphas = ( torch.zeros( [args.num_tasks, 1, 1, 1, 1], device=args.device, requires_grad=True ) )
    alphas.data[task] = 1

    model.apply(lambda m: setattr(m, "alphas", alphas))
    model.apply(lambda m: setattr(m, "task", task))

    acc = adapt_test( model, data_loader, split, alphas)
    print( f"\nLearned Task:{num_tasks_learned-1}\tTesting Task:{task}\tAccuracy:({acc:.4f}%)" )
    model.apply(lambda m: setattr(m, "alphas", None))
    return acc


def epoch_evaluate( model, data_loader, in_task, out_task, split):
    if in_task is not None:
        model.apply(lambda m: setattr(m, "task", in_task))
        #该函数使用model.apply()方法对模型model进行遍历，对每个子模块m使用setattr()函数设置属性task为in_task。
        #是MultitaskMaskConv类中的forward()函数中会根据task属性处理子网络
    
    if split.lower() in ['val', 'validation']:
        loaders = data_loader.val_loaders
    elif split.lower() in ['train_test']:
        loaders = data_loader.test_loaders
        loaders2= data_loader.val_loaders
    elif split.lower() in ['only_test']:
        loaders = data_loader.test_loaders
    else:
        raise NotImplementedError(f'{split} not implemented')
    
    #TODO: 测试前用val训一训
    optimizer, scheduler, train_epochs = my_utils.get_optim_and_scheduler(model, optim_type=args.weight_opt, idx=-1)
    train, batch_evaluate = my_utils.get_train_test_function('weights')
    criterion = nn.CrossEntropyLoss().to(args.device)
    get_editable_weights_mask_dict = getattr(trainers, "weights").get_editable_weights_mask_dict
    weight_mask_dict, curr_act_dict = get_editable_weights_mask_dict(model, type=args.weight_mask_type)
    
    if out_task == 'all':
        task_list = range(args.num_tasks)
    elif type(out_task) == int:
        task_list = [out_task]
    else:
        raise NotImplementedError(f'{out_task} not implemented')

    # with torch.no_grad():
    acc = np.zeros(len(task_list))
    for i, idx in enumerate(task_list):
        correct, total = 0, 0
        if split.lower() in ['train_test']:
            model.train()
            for p in model.module.parameters():
                p.requires_grad = False
            for p in model.module.cnn_model.linear.parameters():
                p.requires_grad = True
            train(model, None, loaders2[idx], optimizer, criterion, 1, in_task, weight_mask_dict, curr_act_dict)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loaders[idx]):
                if args.debug and batch_idx > 10: break
                if args.text_exp:
                    token_type_ids = None
                    if args.superglue:
                        data, mask, token_type_ids, target = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),\
                                batch['token_type_ids'].to(args.device), batch['labels'].to(args.device)
                    else:
                        data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                        # print(data.shape)
                        # print(mask.shape)
                        # print(target.shape)
                    output = model(data, mask, token_type_ids=token_type_ids)["preds"]
                    # print(output.size())
                else:
                    data, target = batch[0].to(args.device), batch[1].to(args.device)
                    output = model(data)["preds"]
                """
                1.pred = output.argmax(dim=1, keepdim=True)：这一行代码的作用是将output张量在第1维（即每行）上取最大值，
                并将结果作为一个新的张量pred。keepdim=True表示在结果张量中保留原来的维度大小。
                2.correct += pred.eq(target.view_as(pred)).sum().item()：这一行代码的作用是计算预测正确的样本数。
                首先，target.view_as(pred)将target张量的形状改变为与pred相同，然后使用eq()函数比较pred和target张量的每个元素是否相等，
                返回一个布尔型的张量。接着，使用sum()函数对该布尔型张量进行求和操作，得到预测正确的样本数，
                最后使用.item()方法将结果转换为Python的整数类型，并累加到correct变量中。
                3.total += pred.numel()：这一行代码的作用是计算总样本数。
                pred.numel()返回pred张量中元素的个数，将其累加到total变量中。
                """
                pred = output.argmax( dim=1, keepdim=True )
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += pred.numel()
        acc[i] = float(correct) / float(total)
        print(f"In Task: {in_task}\tOut Task: {idx}\t{split} Accuracy: {acc[i]:.6f} ")
    if len(acc) == 1:
        return acc[0]
    return acc


def adapt_all( model, test_loaders):
    
    model.eval()
    
    with torch.no_grad():
        test_acc = np.zeros(args.num_tasks)
        for idx in range(args.num_tasks):
            correct = 0
            total = 0
            for batch_idx, batch in enumerate(test_loaders[idx]):
                if args.text_exp:
                    token_type_ids = None
                    if args.superglue:
                        data, mask, token_type_ids, target = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device),\
                             batch['token_type_ids'].to(args.device), batch['labels'].to(args.device)
                    else:
                        data, mask, target = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                    output = model(data, mask, token_type_ids=token_type_ids)["preds"]
                else:
                    data, target = batch[0].to(args.device), batch[1].to(args.device)
                    output = model(data)["preds"]
                pred = output.argmax( dim=1, keepdim=True )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += pred.numel()

            test_acc[idx] = float(correct) / float(total)
            print(f"Task{idx} Val Accuracy: {test_acc[idx]:.6f} ")
        
    return test_acc
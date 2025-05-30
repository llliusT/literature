import os, sys, platform
import pathlib, os
# import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from args import args
import trainers.adaptors as adaptors
import data
import trainers
from utils import utils
from utils.metrics import get_forgetting_metric, get_forward_transfer
from utils import my_utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tasks_all = ["ag", "yelp", "amazon", "yahoo", "dbpedia"]
# tasks_all = ["yelp", "amazon"]
def main():
    before = time.time()
    if args.seed is None:
        args.seed = np.random.randint(10000)
    print(f"SETTING SEED TO {args.seed}")
    my_utils.seed_everything(args.seed)
    
    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    # i是控制文件名的
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}-weights-of-{str(i)}th")
        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"-{platform.node().split('.')[0]}-weights={i}"
            break
        i += 1
    print(f"run_base_dir:{run_base_dir}")
    args.project_name = args.log_dir.split('/')[-1]
    f = open(f'{run_base_dir}/logs.txt', 'w') # 记录日志文件
    sys.stdout = my_utils.Tee(sys.stdout, f)

    (run_base_dir / "settings.txt").write_text(str(args)) # 将配置参数写入文件
    args.run_base_dir = run_base_dir
    print(f"=> Saving data in {run_base_dir}")
    print(args)
    
    #TODO:获取任务的dataloader列表
    loader_path="/home/gch/project/biye_final/BLPMS/data/loader_cache/text_loader.pth"
    print(loader_path)
    if os.path.exists(loader_path)==False:
        data_loader = getattr(data, args.set)()
        torch.save(data_loader, loader_path)
        print(data_loader.tasks)
    else:
        data_loader = torch.load(loader_path)
        print(data_loader.tasks)
    # data_loader = getattr(data, args.set)()
    #args.set='TextCLDataset' 用哪个dataset，只有TextDataset应该是因为在data的init文件中将其他几个图片数据集的dataset注释掉了
    #这个data是从data文件夹的__init__.py文件导入
    #getattr方法是获取对象属性的，在这里目前推断是获取data中TextCLDataset这个类
    #getattr(data, args.set)()是调用这个类，并返回一个对象
    #TextDataset声明对象后，里面有train_loaders，val_loaders，test_loaders列表分别存多个task的dataloader

    #TODO:定义评估函数
    epoch_evaluate = getattr(adaptors, 'epoch_evaluate')

    # Track accuracy on all tasks.
    best_mask_acc = [0.0 for _ in range(args.num_tasks)]
    best_weight_acc = [0.0 for _ in range(args.num_tasks)]
    last_acc = [0.0 for _ in range(args.num_tasks)]
    all_val_acc = np.zeros((args.num_tasks,args.num_tasks))
    all_test_acc = np.zeros((args.num_tasks,args.num_tasks))

    #TODO:实例化模型.
    model = utils.get_model(args.text_exp, data_loader.max_classes)
    #这个args.text_exp=True,在get_model函数中并没有调用
    
    # If necessary, set the sparsity of the model of the model using the ER sparsity budget (see paper).
    if args.er_sparsity is not None:
        for n, m in model.named_modules():
            if hasattr(m, "sparsity"):
                if args.er_sparsity == 'normal':
                    sp = args.sparsity
                elif args.er_sparsity == 'er':
                    raise NotImplementedError(f"{args.er_sparsity} is not implemented!")
                m.sparsity = sp #只有那几个卷积层带sparsity属性，相当于将几个卷积层的sparsity属性赋值为args.sparsity，数值为0.1
                if args.verbose: print(f"Set sparsity of {n} to {m.sparsity}")
    
    # Put the model on the GPU, Optionally resume from a checkpoint.
    model = utils.set_gpu(model)
    # print(model)
    criterion = nn.CrossEntropyLoss().to(args.device)
    writer = SummaryWriter(log_dir=run_base_dir)
    num_tasks_learned = 0

    ## TODO:计算每个任务的accuracy用于前向迁移
    # Getting Random accuracies to compute Forward Transfer ##
    if not args.debug:
        rand_accs = np.zeros(args.num_tasks)
        for ti in range(args.num_tasks):
            #-----
            for p in model.parameters():
                p.requires_grad = True
            #-----
            rand_accs[ti] = epoch_evaluate(model, data_loader, in_task=ti, out_task=ti, split='only_test')
    
    #TODO:遍历所有任务开始训练
    # Iterate through all tasks for training.
    for curr_idx in range(args.num_tasks or 0):#arg.num_tasks存在就使用args.num_tasks，否则使用0
        print("\nTRAINING FOR MASKS\n")
        if curr_idx > 0:
            # 如果当前任务id大于0就加载上一个任务的最佳模型
            # Load best Checkpoint of the Last Task.
            model = my_utils.load_best_training(model, run_base_dir / "local_best.pt")
        
        ## Update Dataloader and Task ##
        print(f"Task {args.set}: {curr_idx}")
        model.apply(lambda m: setattr(m, "task", curr_idx))
        assert hasattr(data_loader, "update_task"), "[ERROR] Need to implement update task method for use with multitask experiments"
        data_loader.update_task(curr_idx)#设置当前的curr_task,train_loader,val_loader,test_loader,num_classes
        
        # TODO:训练掩码Train for masks.
        if args.epochs > 0:
            train, batch_evaluate = my_utils.get_train_test_function("default")
            if curr_idx != 0 and curr_idx != args.num_tasks and args.sim_init != "":
                if "knn" in args.sim_init:
                    print('Performing KNN classification to find similar tasks!')
                    task_accs = my_utils.findSimilarTasks(model, data_loader, num_tasks_learned, type=args.sim_init, num_topk=args.num_topk)
                    print(f"task accs: {task_accs}")
                    best_indices = np.array([])
                    if args.sim_init == "knn_best":
                        if task_accs.max() > 1/data_loader.task_classes[curr_idx]:
                            best_indices = np.array([task_accs.argmax()])
                    elif args.sim_init == "knn_always":
                        best_indices = np.array([task_accs.argmax()])
                    elif args.sim_init == "knn_all":
                        best_indices = np.where(task_accs > 1/data_loader.task_classes[curr_idx])[0]
                else:
                    raise NotImplementedError(f"{args.sim_init} not implemented!")

                if best_indices.size == 0:
                    print(f'No Good tasks found.')
                else:
                    print(f"Type: {type}\tBest Index: {best_indices}")
                    my_utils.score_transfer(model, best_indices, curr_idx)
                model.apply(lambda m: setattr(m, "task", curr_idx))
                data_loader.update_task(curr_idx)
            else:
                print("No Similarity based initialization.")
            
            optimizer, scheduler, train_epochs = my_utils.get_optim_and_scheduler(model, optim_type=args.mask_opt, idx=curr_idx)

            # TODO：为当前任务训练得分Train on the scores for current task.
            for epoch in range(1, train_epochs + 1):
                print('\n')
                train(model, writer, data_loader, optimizer, criterion, epoch, curr_idx,)
                utils.cache_weights(model, num_tasks_learned + 1)
                last_acc[curr_idx] = batch_evaluate( model, writer, criterion, data_loader, epoch, curr_idx, split='Val')
                if last_acc[curr_idx] > best_mask_acc[curr_idx]:
                    best_mask_acc[curr_idx] = last_acc[curr_idx]
                    torch.save( { "epoch": args.epochs, "arch": args.model, "state_dict": model.state_dict(), "best_mask_acc": best_mask_acc,
                        "last_acc": last_acc, "args": args, }, run_base_dir / "local_best.pt",)

                if scheduler:
                    scheduler[1].step()
                    scheduler[0].step(last_acc[curr_idx])
                if ( args.iter_lim > 0 and len(data_loader.train_loader) * epoch > args.iter_lim ):
                    break
            # caching masks and deleting optimizer and schedulers
            utils.cache_masks(model)
            del optimizer, scheduler
        print(model)
        #TODO:为当前任务训练权重Train on the weights for current task.
        if args.weight_epochs > 0:
            
            print("\nTRAINING FOR WEIGHTS\n")
            optimizer, scheduler, train_epochs = my_utils.get_optim_and_scheduler(model, optim_type=args.weight_opt, idx=-1)
            train, batch_evaluate = my_utils.get_train_test_function('weights')
            get_editable_weights_mask_dict = getattr(trainers, "weights").get_editable_weights_mask_dict
            weight_mask_dict, curr_act_dict = get_editable_weights_mask_dict(model, type=args.weight_mask_type)
                
            for weight_epoch in range(1, args.weight_epochs+1):
                train(model, writer, data_loader.train_loader, optimizer, criterion, weight_epoch, curr_idx, weight_mask_dict, curr_act_dict)
                print('\n')
                last_acc[curr_idx] = batch_evaluate( model, writer, criterion, data_loader, weight_epoch, curr_idx, split='Val' )
                if last_acc[curr_idx] > best_mask_acc[curr_idx]:
                    best_mask_acc[curr_idx] = last_acc[curr_idx]
                    torch.save( { "epoch": args.epochs, "arch": args.model, "state_dict": model.state_dict(), "best_mask_acc": best_mask_acc,
                     "last_acc": last_acc, "args": args, }, run_base_dir / "local_best.pt",)
                if scheduler:
                    scheduler[1].step()
                    scheduler[0].step(last_acc[curr_idx])

            del optimizer, scheduler

        num_tasks_learned += 1
        model.apply(lambda m: setattr(m, "num_tasks_learned", num_tasks_learned))

        # EVALUTATION ON ALL TASKS!
        print('EPOCH END， EVALUATION')
        if num_tasks_learned in args.eval_ckpts or num_tasks_learned == args.num_tasks or args.eval_all:
            # Evaluate until current task + 1 if not the last task
            eval_tasks = (num_tasks_learned + 1) if curr_idx < args.num_tasks-1 else num_tasks_learned
                
            for test_idx in range(eval_tasks):
                #------
                # for p in model.parameters(): p.requires_grad = True
                # for b in model.buffers(): b.requires_grad = True
                all_test_acc[curr_idx, test_idx] = epoch_evaluate(model, data_loader, in_task=test_idx, out_task=test_idx, split='train_test')
                writer.add_scalar(f"task_test/acc_{test_idx}", all_test_acc[curr_idx, test_idx], curr_idx)

            print(f"Adapt Test Accuracies: {all_test_acc[curr_idx,:]}")
            print(f"Average Test Accuracy: {all_test_acc[curr_idx, :num_tasks_learned].mean():.4f}")
            utils.clear_masks(model)
            torch.cuda.empty_cache()
        torch.save(model, run_base_dir / "model.pt")
        torch.save(model.module.cnn_model.linear.state_dict(), run_base_dir / f"task:{tasks_all[curr_idx]}-linear.pt")


    # printing stuff to console
    if args.num_tasks > 1:
        # val_forgetting = get_forgetting_metric(all_val_acc, bwt=True)
        test_forgetting = get_forgetting_metric(all_test_acc, bwt=True)

        print(f'Test Forgetting: {test_forgetting.tolist()}')
        print(f'Average Test Forgetting: {test_forgetting.mean():.4f}')
    
        overlap_obj = my_utils.getOverlap(model, args.num_tasks)
        print(f"Sparse Overalp: {overlap_obj.mask_sparse_overlap.tolist()}")
        print(f"Total Overalp: {overlap_obj.mask_total_overlap.tolist()}")
        print(f"Avg. Sparse Overalp: {overlap_obj.avg_sparse_overlap:.8f}")
        print(f"Avg. Total Overalp: {overlap_obj.avg_sparse_overlap:.8f}")
    
    # saving the last model.
    if args.save:
        torch.save( { "epoch": args.epochs, "arch": args.model, "state_dict": model.state_dict(), "best_mask_acc": best_mask_acc,
                    "last_acc": last_acc, "args": args, }, run_base_dir / "final.pt",)
    print( f"Finished experiment in {str((time.time() - before) / 60.0)} minutes." )

    #return all_val_acc


if __name__ == "__main__":
    main()
    pass

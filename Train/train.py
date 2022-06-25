import os
import oyaml as yaml
import math
import time
import cv2
import torch
import shutil
import numpy as np
import random
import tqdm
import argparse
import torch.distributed as dist

from vfi import get_loader, get_loss, get_model, get_optimizer
from vfi.solver import Solver

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cfg,local_rank):

    ## setup environment
    run_id = random.randint(1, 100000)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    log_path = os.path.join("runs", os.path.basename(cfg['exp_name'])[:-4], str(run_id))
        
    if local_rank <= 0:
        writer = SummaryWriter(log_path + '/train')
        writer_val = SummaryWriter(log_path + '/validate')
        shutil.copy(cfg['exp_name'], log_path)
    else:
        writer, writer_val = None, None


    ## setup device

    torch.cuda.set_device(local_rank)
    #world_size = dist.get_world_size()
    dist.init_process_group(backend="nccl", init_method='env://')

    step = 0
    val_psnr = 0.0

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])

    dataset_train = data_loader(dataset_split='train', dataset_path=cfg["data"]["path"])
    sampler = DistributedSampler(dataset_train, shuffle=True)
    train_data = DataLoader(dataset_train, batch_size=cfg["training"]["batch_size"]//torch.cuda.device_count(), num_workers=cfg["training"]["n_workers"], pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()

    dataset_val = data_loader(dataset_split='validation', dataset_path=cfg["data"]["path"])
    val_data = DataLoader(dataset_val, batch_size=1, pin_memory=True, num_workers=cfg["training"]["n_workers"], drop_last=True)
    
    #Setup Loss
    loss = get_loss(cfg['training']['loss'])

    #Setup Model
    model = get_model(cfg['model'])

    #Setup optimizer
    optimizer = get_optimizer(cfg['training']['optimizer'],model)

    solver = Solver(model, optimizer, loss, local_rank)

    #solver.load_model(cfg['model']['path'])
    #################
    time_stamp = time.time()
    #evaluate(solver, val_data, step, local_rank, writer_val)

    while step <= cfg["training"]["train_iters"]:
        for img0, img1, gt, t_loc in train_data:
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            img0 = img0.to(device, non_blocking=True) / 255.
            img1 = img1.to(device, non_blocking=True) / 255.
            gt = gt.to(device, non_blocking=True) / 255

            ###################################################
            output = solver.update(img0, img1, gt, step, cfg["training"]["train_iters"], training=True, fltTimes=[t_loc])
            train_time_interval = time.time() - time_stamp

            pred = output['pred']
            output.pop('pred')

            time_stamp = time.time()
            if step % 100 == 1 and local_rank == 0:
                for intGroup, objGroup in enumerate(solver.optimG.optim.param_groups):
                    writer.add_scalar('lr-'+str(intGroup), objGroup['lr'], step)
                #end

                for (los_n,los_v) in output.items():
                    writer.add_scalar(los_n, los_v, step)
                #end
            #end
       
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')

                for i in range(2):
                    imgs = np.concatenate((pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                #end

                writer.flush()
            #end

            if step % 500 == 1 and local_rank == 0:
                print('Iter:{}/{} time:{:.2f}+{:.2f} total_loss:{:.4e}'.format(step, cfg["training"]["train_iters"], data_time_interval, train_time_interval, output['lossttl']))
            #end
            
            if step%cfg['training']['val_interval']==1:
                bestflag = False             
                if local_rank==0:
                    psnr_ = evaluate(solver, val_data, step, local_rank, writer_val)
                    if psnr_ > val_psnr:
                        val_psnr = psnr_
                        bestflag = True
                    #end
                    solver.save_model(log_path, local_rank, bestflag)    
                #end
                #dist.barrier()
            #emd

            step += 1
        #end
    #end
#end

def evaluate(solver, val_data, step, local_rank, writer_val):
    loss_tt_list = []
    psnr_list = []
    time_stamp = time.time()
    for i, (img0, img1, gt, t_loc) in enumerate(val_data):

        img0 = img0.to(device, non_blocking=True) / 255.
        img1 = img1.to(device, non_blocking=True) / 255.
        gt = gt.to(device, non_blocking=True) / 255.

        with torch.no_grad():
            output = solver.update(img0, img1, gt, training=False, fltTimes=[t_loc])
        #end

        pred = output['pred']

        pred = pred.cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
        pred = np.round(255*pred).astype('uint8') / 255.
        pred = torch.FloatTensor(pred.transpose(2, 0, 1)[None, :, :, :]).cuda()

        loss_tt_list.append(output['lossttl'].cpu().numpy())

        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
        #end

        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')

        if i == 0 and local_rank == 0:
            for j in range(2):
                imgs = np.concatenate((pred[i], gt[i]), 1)[:, :, ::-1]
                writer_val.add_image(str(i) + '/img', imgs.copy(), step, dataformats='HWC')
            #end
        #end
    #end
    
    eval_time_interval = time.time() - time_stamp
    if local_rank == 0:
        print('eval time: {}    psnr: {}'.format(eval_time_interval,np.array(psnr_list).mean())) 
        writer_val.add_scalar('loss_ltotal', np.array(loss_tt_list).mean(), step)
        writer_val.add_scalar('psnr', np.array(psnr_list).mean(), step)
    #end

    return np.array(psnr_list).mean()
#end

        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='vfi')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/VFI_NewLoss.yml",
        help="Configuration file to use",
    )

    parser.add_argument('--local_rank', type = int, default = 0)

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    cfg['exp_name'] = args.config

    train(cfg,local_rank = args.local_rank)

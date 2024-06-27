import datetime
import os
import sys
import argparse
import logging
import json
from collections import deque
import numpy as np
import cv2

import torch
import torch.utils.data
import torch.optim as optim


from FYP_RG.models import get_network
from hardware.device import get_device
from torchsummary import summary

# import tensorboardX

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Grasp Generator')

    # Network
    parser.add_argument('--network', type=str, default='resnet', help='Network Name in .models')
    parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str,default="cornell", help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str,default="/home/zzl/Pictures/cornell" ,help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.80, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=48, help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=200, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=350, help='Validation Batches')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--scheduler', type=int, default=1,
                        help='use Scheduler')
    parser.add_argument('--update-threshold', type=int, default=1,
                        help='use Scheduler')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='logs/', help='Log directory')
    parser.add_argument('--vis', default=False, help='Visualise the training process')
    parser.add_argument('--gdrive', type=int, default=1, help='Save to drive')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()
    return args

def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
            for x, y, didx, rot, zoom_factor in val_data:
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   threshold=iou_threshold,
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1
                    

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False,scheduler=None, iou_threshold = 0.25):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    index=0
    count=1
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            # print("shape:",x.shape)
            batch_idx += 1
            if batch_idx >= batches_per_epoch and batches_per_epoch != 1:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']
            loss = loss * (1/iou_threshold)

            
            if batch_idx % 100 == 0 or batches_per_epoch == 1:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
              scheduler.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%d%m%y_%H%M')

    run_name = args.dataset[0] + "_rgb"
    run_name += "d" if args.use_depth else ""
    net_desc = '{}_{}'.format(dt, run_name)

    save_folder = os.path.join(args.logdir, net_desc)
    model_folder = os.path.join(save_folder, args.outdir)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    #tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    dataset = Dataset(args.dataset_path,
                            output_size=args.input_size,
                            ds_rotate=args.ds_rotate,
                            random_rotate=True,
                            random_zoom=True,
                            include_depth=args.use_depth,
                            include_rgb=args.use_rgb,
                            )

    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler,
    )

    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    print("channels:",input_channels)

    # init model
    net=get_network(input_channels=input_channels,
              output_size=args.input_size,)
    
    device = get_device(args.force_cpu)
    net = net.to(device)

    # set optimizer
    if args.optim.lower() == 'adam':
        optimizer = optim.AdamW(net.parameters(),lr=1e-4)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # set scheduler
    mode='cosineAnnWarm'
    if args.scheduler:
      if mode=='cosineAnn':
          scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
      elif mode=='cosineAnnWarm':
          scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
    else:
      scheduler = None

    logging.info('Done')

    best_iou = 0.0

    avg_n = 5
    iou_stack = deque(maxlen=avg_n)
    iou_threshold = args.threshold
    
    model_count = 0

    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer,
                              args.batches_per_epoch,
                              vis=args.vis,
                              scheduler=scheduler,
                              iou_threshold = iou_threshold,
                              )

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        
        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])

        # add iou to fifo stack
        if len(iou_stack) < avg_n:
            iou_stack.appendleft(iou)
        else:
            iou_stack.pop()
            iou_stack.appendleft(iou)

        # calculate new iou based on threshold value
        real_iou = iou*iou_threshold
        logging.info('thresh %0.4f iou %0.4f >>> %0.4f' % (iou_threshold, iou, real_iou) )

        # update threshold
        if len(iou_stack) == avg_n and (sum(iou_stack)/avg_n > 0.85) and args.update_threshold:
              iou_threshold = iou_threshold*1.05
              logging.info(f'iou_threshold updated {iou_threshold} !!!')
              
        if real_iou > best_iou or epoch == 0 or (epoch % 10) == 0 or epoch == (args.epochs-1):
            model_name = str(model_count) + "_" + run_name + '_r_%02d_thresh_%02d_iou_%02d__epoch_%02d' % (real_iou*100,iou_threshold*100, iou*100,epoch)
            model_out_path = os.path.join(save_folder,args.outdir, model_name+".pt")
            torch.save(net, model_out_path, _use_new_zipfile_serialization=False)            
            logging.info(f'Model saved as {model_out_path}')
            model_count +=1

            if real_iou > best_iou and args.gdrive:
              model_out_path_d = os.path.join("/content/drive/MyDrive/bitirme/models/", run_name+".pt")
              torch.save(net, model_out_path_d, _use_new_zipfile_serialization=False)
              logging.info(f'Model saved to Drive')

            if real_iou >= best_iou:
              best_iou = real_iou
              logging.info('best_iou updated >>> %0.4f' % (best_iou) )


if __name__ == '__main__':
    run()

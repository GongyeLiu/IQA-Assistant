import argparse
import math
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from metrics.dataset import BaseSingleDataset
from net.FID import load_patched_inception_v3, extract_inception_features, calculate_fid


def calculate_fid_folder(args):
    device = torch.tensor(0.0).cuda().device

    # inception model
    inception = load_patched_inception_v3(device)
    # create dataset
    dataset = BaseSingleDataset(args.restored_folder)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        drop_last=False)
    args.num_sample = len(dataset)
    total_batch = math.ceil(args.num_sample / args.batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                out = data['lq'].cuda()
                out = (out - 0.5) / 0.5
                yield out

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:args.num_sample]
    print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load(args.fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    print(args.restored_folder)
    print('fid:', fid)


def calc_fid_gt(pred_fold, gt_fold, out_path=None, device='cpu'):
    # inception model
    inception = load_patched_inception_v3(device)

    mean1, cov1 = calc_fid_mean_conv_from_folder(inception, pred_fold)
    mean2, cov2 = calc_fid_mean_conv_from_folder(inception, gt_fold)
    fid = calculate_fid(mean1, cov1, mean2, cov2)
    print('fid:', fid)
    return fid


def calc_fid_mean_conv_from_folder(inception, folder, device='cpu'):
    # create dataset
    dataset = BaseSingleDataset(folder)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        sampler=None,
        drop_last=False)

    num_sample = len(dataset)
    total_batch = math.ceil(num_sample / 32)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                out = data['lq'].to(device)
                out = (out - 0.5) / 0.5
                yield out

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:num_sample]
    print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    return sample_mean, sample_cov


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', type=str, default="pred_folder/*.png")
    parser.add_argument('--gt_folder', type=str, default="gt_folder/*.png")
    parser.add_argument('--fid_stats', type=str, default="/home/lgy22/pretrained_weights/inception_FFHQ_512.pth")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default='4')
    parser.add_argument('--out_path', type=str, default=None)
    args = parser.parse_args()
    # calculate_fid_folder(args)
    device = torch.device('cuda:{}'.format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    calc_fid_gt(args.pred_folder, args.gt_folder, args.out_path, device)

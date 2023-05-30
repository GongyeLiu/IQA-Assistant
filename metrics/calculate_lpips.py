import os

import argparse
import cv2
from glob import glob
import numpy as np
import os.path as osp

import torch
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import lpips


def imread(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255
    img = img.transpose((2, 0, 1))
    return img


@torch.no_grad()
def img2tensor(img, device='cpu'):
    tensor = torch.from_numpy(img).float().to(device)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


@torch.no_grad()
def calc_lpips_from_folder(pred_fold, gt_fold, out_path=None, device='cpu'):
    # create model
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    # create dataset
    pred_path = sorted(glob(pred_fold))
    gt_path = sorted(glob(gt_fold))

    if out_path is not None:
        f = open(out_path + "/output_lpips.txt", 'w', encoding='utf-8')
    else:
        f = None

    lpips_list = []
    for i in range(len(pred_path)):
        pred = img2tensor(imread(pred_path[i]), device=device)
        gt = img2tensor(imread(gt_path[i]), device=device)
        name = pred_path[i].split('/')[-1]

        lpips_val = loss_fn_vgg(pred, gt)
        lpips_val = lpips_val.cpu().item()

        print("Image{} | LPIPS:{:.4f}".format(name, lpips_val))
        lpips_list.append(lpips_val)

    print("Avg LPIPS:", np.mean(lpips_list))
    print("min LPIPS:{:f} with file {}".format(np.min(lpips_list), pred_path[np.argmin(lpips_list)]))
    print("max LPIPS:{:f} with file {}".format(np.max(lpips_list), pred_path[np.argmax(lpips_list)]))
    return np.mean(lpips_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_folder', type=str, help='Path to the folder.', default="pred_folder/*.png")
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', default="gt_folder/*.png")
    parser.add_argument('--gpu_id', type=str, default='4')
    parser.add_argument('-out_path', type=str, default=None)
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    calc_lpips_from_folder(args.pred_folder, args.gt_folder, args.out_path, device)
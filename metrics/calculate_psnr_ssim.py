import cv2
import numpy as np

import argparse

from skimage.metrics import structural_similarity as ssim_calc
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from glob import glob


def imread(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
    return img


def calc_psnr_ssim_from_folder(pred_fold, gt_fold, out_path=None):
    # create dataset
    pred_path = sorted(glob(pred_fold))
    gt_path = sorted(glob(gt_fold))

    if out_path is not None:
        f = open(out_path + "/output_psnr_ssim.txt", 'w', encoding='utf-8')
    else:
        f = None

    psnr_list = []
    ssim_list = []
    for i in range(len(pred_path)):
        pred = imread(pred_path[i])
        gt = imread(gt_path[i])
        name = pred_path[i].split('/')[-1]
        psnr = psnr_calc(pred, gt, data_range=255.0)
        ssim = ssim_calc(pred, gt, data_range=255.0, multichannel=True)

        if f is not None or True:
            print("Image{} | PSNR:{:.3f} | SSIM:{:.4f}".format(name, psnr, ssim), file=f)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print("Avg PSNR:", np.mean(psnr_list))
    print("min PSNR:{:f} with file {}".format(np.min(psnr_list), pred_path[np.argmin(psnr_list)]))
    print("max PSNR:{:f} with file {}".format(np.max(psnr_list), pred_path[np.argmax(psnr_list)]))

    print("Avg SSIM:", np.mean(ssim_list))
    print("min SSIM:{:f} with file {}".format(np.min(ssim_list), pred_path[np.argmin(ssim_list)]))

    print("max SSIM:{:f} with file {}".format(np.max(ssim_list), pred_path[np.argmax(ssim_list)]))
    return np.mean(psnr_list), np.mean(ssim_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_folder', type=str, help='Path to the folder.', default="pred_folder/*.png")
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', default="gt_folder/*.png")
    parser.add_argument('-out_path', type=str, default=None)

    args = parser.parse_args()
    calc_psnr_ssim_from_folder(args.pred_folder, args.gt_folder, args.out_path)
    # calc_fid_gt(args)

import torch
import argparse

from metrics.calculate_psnr_ssim import calc_psnr_ssim_from_folder
from metrics.calculate_lpips import calc_lpips_from_folder
from metrics.calculate_fid_folder import calc_fid_gt


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('--pred_folder', type=str, default="pred_folder/*.png", help="Your predicted image folder.")
    parser.add_argument('--gt_folder', type=str, default="gt_folder/*.png", help="Your ground truth image folder.")
    parser.add_argument('--gpu_id', type=str, default='4')
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--metric', type=str, nargs='+', choices=['psnr', 'PSNR', 'lpips', 'LPIPS', 'fid', 'FID', 'all'],
                        default=['psnr', 'lpips', 'fid'])
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_parser()

    pred_folder = args.pred_folder
    gt_folder = args.gt_folder
    out_path = args.out_path
    device = torch.device('cuda:{}'.format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')

    print(pred_folder)
    # calculate psnr and ssim
    if 'psnr' in args.metric or 'PSNR' in args.metric or 'all' in args.metric:
        psnr, ssim = calc_psnr_ssim_from_folder(pred_folder, gt_folder, out_path)

    # calculate lpips
    if 'lpips' in args.metric or 'LPIPS' in args.metric or 'all' in args.metric:
        lpips = calc_lpips_from_folder(pred_folder, gt_folder, out_path)

    # calculate fid
    if 'fid' in args.metric or 'FID' in args.metric or 'all' in args.metric:
        fid = calc_fid_gt(pred_folder, gt_folder, out_path)

    print("=========================================================")
    if 'psnr' in args.metric or 'PSNR' in args.metric or 'all' in args.metric:
        print(f"psnr: {psnr}, ssim: {ssim}")
    if 'lpips' in args.metric or 'LPIPS' in args.metric or 'all' in args.metric:
        print(f"lpips: {lpips}")
    if 'fid' in args.metric or 'FID' in args.metric or 'all' in args.metric:
        print(f"fid: {fid}")


# IQA-Assistant
📈 A simple image metric calcuator for your computer vision project.

## Available
- [x] PSNR
- [x] SSIM
- [x] LPIPS
- [x] FID

## Usage
```
python calc_metric.py --pred_path <pred_path> --gt_path <gt_path> --gpu_id <gpu_id> --out_path <output_path> --metric 'psnr' 'fid' 'lpips'
```

with the following options:
- `pred_path`: path to predicted images, which ends with the suffix name of images. For example, 'pred/*.png'
- `gt_path`: path to ground truth images, which ends with the suffix name of images. For example, 'gt/*.png'
- `gpu_id`: gpu id. For example, '0'.
- `out_path`: path to output file.
- `metric`: metric to calculate. If all metrics are needed, just leave it blank or use 'all' as input.
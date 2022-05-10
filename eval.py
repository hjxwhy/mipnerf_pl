import argparse
import os
import torch
from tqdm import tqdm
from datasets import dataset_dict
from torch.utils.data import DataLoader
from datasets.datasets import Rays_keys, Rays
from utils.metrics import eval_errors, summarize_results
from utils.vis import save_images
from render_video import generate_video
from models.mip import rearrange_render_image
from models.nerf_system import MipNeRFSystem

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="Path to ckpt.")
parser.add_argument("--data", help="Path to data.")
parser.add_argument("--out_dir", help="Output directory.", type=str, required=True)
parser.add_argument("--chunk_size", help="Chunck size for render.", type=int, default=12288)
parser.add_argument("--white_bkgd", help="Train set image background color.", type=bool, default=True)
parser.add_argument('--save_image', help='whether save predicted image', action='store_true')
parser.add_argument('--summa_only', help='Only summarize results', action='store_true')
parser.add_argument('--scale', help='eval scale', type=int, required=True, choices=[1, 4])
parser.add_argument('--base_size', help='source image size', type=list, default=[800, 800])


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MipNeRFSystem.load_from_checkpoint(args.ckpt).to(device).eval()

    hparams = model.hparams
    exp_name = hparams['exp_name']
    if args.summa_only:
        return [exp_name]
    test_dataset = dataset_dict[model.hparams['dataset_name']](data_dir=args.data,
                                                               split='test',
                                                               white_bkgd=hparams['val.white_bkgd'],
                                                               batch_type=hparams['val.batch_type'],
                                                               # factor=2
                                                               )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    psnr_values = []
    ssim_values = []
    n = -1
    for i in range(args.scale):
        save_path = os.path.join(args.out_dir, 'test', exp_name, str(2 ** i))
        os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            if idx % args.scale == 0:
                n += 1
            rays, rgbs = batch
            rays = Rays(*[getattr(rays, name).to(device) for name in Rays_keys])
            rgbs = rgbs.to(device)
            _, height, width, _ = rgbs.shape  # N H W C
            single_image_rays, val_mask = rearrange_render_image(rays, args.chunk_size)
            coarse_rgb, fine_rgb = [], []
            distances, accs = [], []
            with torch.no_grad():
                for batch_rays in single_image_rays:
                    _, (f_rgb, distance, acc) = model(batch_rays, False, args.white_bkgd)
                    fine_rgb.append(f_rgb)
                    distances.append(distance)
                    accs.append(acc)

            fine_rgb = torch.cat(fine_rgb, dim=0)
            distances = torch.cat(distances, dim=0)
            accs = torch.cat(accs, dim=0)

            fine_rgb = fine_rgb.reshape(1, height, width, fine_rgb.shape[-1])  # N H W C
            distances = distances.reshape(height, width)  # H W
            accs = accs.reshape(height, width)  # H W
            psnr_val, ssim_val = eval_errors(fine_rgb, rgbs)
            psnr_values.append(psnr_val.cpu().item())
            ssim_values.append(ssim_val.cpu().item())
            out_path = os.path.join(args.out_dir, 'test', exp_name, str(int(args.base_size[0] / width)))
            if args.save_image:
                save_images(fine_rgb, distances, accs, out_path, n)
        with open(os.path.join(args.out_dir, 'test', exp_name, 'psnrs.txt'), 'w') as f:
            f.write(' '.join([str(v) for v in psnr_values]))
        with open(os.path.join(args.out_dir, 'test', exp_name, 'ssims.txt'), 'w') as f:
            f.write(' '.join([str(v) for v in ssim_values]))
        generate_video(os.path.join(args.out_dir, 'test', exp_name))
        return [exp_name]


if __name__ == '__main__':
    # blender_scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    args = parser.parse_args()
    blender_scenes = main(args)
    # I remove the LPIPS metric, if you want to eval it, you should modify eval code simply.
    print('PSNR | SSIM | Average')
    if args.scale == 1:
        print(summarize_results(args.out_dir, blender_scenes, 1))
    else:
        print(summarize_results(args.out_dir, blender_scenes, args.scale))

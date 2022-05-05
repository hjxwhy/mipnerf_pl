import argparse
import os
import torch
from datasets import dataset_dict
from torch.utils.data import DataLoader
from datasets.multi_blender import Rays_keys, Rays
from utils.metrics import eval_errors, summarize_results
from utils.vis import save_images
from models.mip import rearrange_render_image
from models.nerf_system import MipNeRFSystem

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="Path to ckpt.", required=True)
parser.add_argument("--chunk_size", help="Chunck size for render.", type=int, default=8192)
parser.add_argument("--white_bkgd", help="Train set image background color.", type=bool, default=True)
parser.add_argument("--out_dir", help="Output directory.", type=str, default='./')
parser.add_argument('--save_image', help='whether save predicted image', action='store_true')
parser.add_argument('--multi_scale', help='eval multi scale', action='store_true')


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MipNeRFSystem.load_from_checkpoint(args.ckpt).to(device).eval()

    hparams = model.hparams
    exp_name = hparams['exp_name']
    test_dataset = dataset_dict[model.hparams['dataset_name']](data_dir=hparams['data.path'],
                                                               split='test',
                                                               white_bkgd=hparams['val.white_bkgd'],
                                                               batch_type=hparams['val.batch_type'],
                                                               factor=2
                                                               )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
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
            save_path = os.path.join(args.out_dir, 'test', exp_name)
            os.makedirs(save_path, exist_ok=True)
            if args.save_image:
                save_images(fine_rgb, distances, accs, save_path, idx)
        with open(os.path.join(save_path, 'psnrs.txt'), 'w') as f:
            f.write(' '.join([str(v) for v in psnr_values]))
        with open(os.path.join(save_path, 'ssims.txt'), 'w') as f:
            f.write(' '.join([str(v) for v in ssim_values]))


if __name__ == '__main__':
    # blender_scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    blender_scenes = ['lego']
    args = parser.parse_args()
    main(args)
    # I remove the LPIPS metric, if you want to eval it, you should modify eval code simply.
    print('PSNR | SSIM | Average')
    if not args.multi_scale:
        print(summarize_results(args.out_dir, blender_scenes, 1))
    else:
        print(summarize_results(args.out_dir, blender_scenes, 4))

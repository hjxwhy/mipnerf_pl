import argparse
import os
import torch
from tqdm import tqdm
from datasets import dataset_dict
from torch.utils.data import DataLoader
from datasets.datasets import Rays_keys, Rays
from utils.metrics import eval_errors, summarize_results
from utils.vis import save_images, l2_normalize, visualize_normal
from render_video import generate_video
from models.mip import rearrange_render_image
from models.nerf_system import MipNeRFSystem

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="Path to ckpt.")
parser.add_argument("--data", help="Path to data.")
parser.add_argument("--out_dir", help="Output directory.", type=str, required=True)
parser.add_argument("--chunk_size", help="Chunck size for render.", type=int, default=2048)
parser.add_argument("--white_bkgd", help="Train set image background color.", type=bool, default=True)
parser.add_argument('--save_image', help='whether save predicted image', action='store_true')
parser.add_argument('--summa_only', help='Only summarize results', action='store_true')
parser.add_argument('--scale', help='eval scale', type=int, required=True, choices=[1, 4])
#parser.add_argument('--base_size', help='source image size', type=list, default=[400, 400])
parser.add_argument('--base_size', help='source image size', type=list, default=[624, 468])
parser.add_argument('--no_supervision', help='whether there is GT supervision or not', action='store_true', default=True)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MipNeRFSystem.load_from_checkpoint(args.ckpt).to(device).eval()

    hparams = model.hparams
    exp_name = hparams['exp_name']
    if args.summa_only:
        return [exp_name]
    test_dataset = dataset_dict[model.hparams['dataset_name']](data_dir=args.data,
                                                               split='cam',
                                                               white_bkgd=hparams['val.white_bkgd'],
                                                               batch_type=hparams['val.batch_type'],
                                                               # factor=2
                                                               )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
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
            #print(rays.depth.shape, rgbs.shape)
            rays = Rays(*[(getattr(rays, name)).to(device) for name in Rays_keys])
            rgbs = (rgbs).to(device)
            eval_batch = {'rgbs': rgbs, 'rays': rays}
            ret = model.render_image(eval_batch, mode='test', chunk_size = args.chunk_size)
            
            _, height, width, _ = rgbs.shape  # N H W C
            fine_rgb = ret['rgb_fine'].reshape(1, height, width, ret['rgb_fine'].shape[-1])
            distances = ret['depth'].reshape(height, width)
            
            mask = (~torch.any(rays.normal, dim=-1)) #(1,H,W)
            mask = mask.view(height*width) #(H,W)
            normals_pred = torch.sum(ret['weights'][...,None]*ret['normal'], dim=1) # (H*W, N_samples + Samples_fine (128), 3) --> (H*W, 3)
            normals_pred = l2_normalize(normals_pred)
            normals_pred[mask, ...] = 0
            normals_pred = visualize_normal(normals_pred.view(height, width, 3))  
            normals_pred = normals_pred.view(1, height, width, 3)

            accs = ret['weights'].sum(-1).reshape(height,width)
            rgbs = rgbs.view(1, height, width, rgbs.shape[-1])
            if not args.no_supervision:
                psnr_val, ssim_val = eval_errors(fine_rgb, rgbs)
                psnr_values.append(psnr_val.cpu().item())
                ssim_values.append(ssim_val.cpu().item())
            out_path = os.path.join(args.out_dir, 'test', exp_name, str(int(args.base_size[0] / width)))
            if args.save_image:
                save_images(fine_rgb, distances, normals_pred, out_path, n)
        if not args.no_supervision:
            with open(os.path.join(args.out_dir, 'test', exp_name, 'psnrs.txt'), 'w') as f:
                f.write(' '.join([str(v) for v in psnr_values]))
            with open(os.path.join(args.out_dir, 'test', exp_name, 'ssims.txt'), 'w') as f:
                f.write(' '.join([str(v) for v in ssim_values]))
        generate_video(os.path.join(args.out_dir, 'test', exp_name))
        return [exp_name]


if __name__ == '__main__':
    # blender_scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    args = parser.parse_args()
    print(f'evaluating {args.ckpt} and saving results at {args.out_dir}')
    blender_scenes = main(args)
    # I remove the LPIPS metric, if you want to eval it, you should modify eval code simply.
    print('PSNR | SSIM | Average')
    if not args.no_supervision:
        if args.scale == 1:
            print(summarize_results(args.out_dir, blender_scenes, 1))
        else:
            print(summarize_results(args.out_dir, blender_scenes, args.scale))

import numpy as np
from utils.vis import create_spheric_poses
from tqdm import tqdm
import os
import argparse
import collections
import torch
from torch.utils.data import Dataset, DataLoader
from utils.vis import save_images
from models.mip import rearrange_render_image
from models.nerf_system import MipNeRFSystem

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields


class RenderGen(Dataset):
    def __init__(self, base_focal, base_size, scales=4):
        super(RenderGen, self).__init__()
        self.base_focal = base_focal
        self.base_size = base_size
        self.scales = scales  # if multi else 1
        self.near = 2
        self.far = 6
        self._generate_rays()

    def _generate_rays(self):
        """Generating rays for all images."""
        cam2world = create_spheric_poses(4)
        width = np.ones(len(cam2world)) * self.base_size[0]
        height = np.ones(len(cam2world)) * self.base_size[1]
        focal = np.ones(len(cam2world)) * self.base_focal

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                indexing='xy')

        widths = []
        heights = []
        focals = []
        cam2worlds = []

        for i in range(self.scales):
            widths.append(width / 2 ** i)
            heights.append(height / 2 ** i)
            focals.append(focal / 2 ** i)
            cam2worlds.append(cam2world)
        widths = np.hstack(widths)
        heights = np.hstack(heights)
        focals = np.hstack(focals)
        cam2worlds = np.vstack(cam2worlds)
        self.n_sample = len(cam2worlds)
        fx = np.array(focals)
        fy = np.array(focals)
        cx = np.array(widths) * .5
        cy = np.array(heights) * .5
        arr0 = np.zeros_like(cx)
        arr1 = np.ones_like(cx)
        pix2cam = np.array([
            [arr1 / fx, arr0, -cx / fx],
            [arr0, -arr1 / fy, cy / fy],
            [arr0, arr0, -arr1],
        ])
        pix2cam = np.moveaxis(pix2cam, -1, 0)
        xy = [res2grid(w, h) for w, h in zip(widths, heights)]
        pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
        camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
        directions = [(v @ c2w[:3, :3].T).copy() for v, c2w in zip(camera_dirs, cam2worlds)]
        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(directions, cam2worlds)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return [
                x * np.ones_like(origins[i][..., :1])
                for i in range(len(origins))
            ]

        lossmult = broadcast_scalar_attribute(1).copy()
        near = broadcast_scalar_attribute(self.near).copy()
        far = broadcast_scalar_attribute(self.far).copy()
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmult,
            near=near,
            far=far)
        del origins, directions, viewdirs, radii, lossmult, near, far, xy, pixel_dirs, camera_dirs

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        return rays


def run_render(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MipNeRFSystem.load_from_checkpoint(args.ckpt).to(device).eval()

    hparams = model.hparams
    exp_name = hparams['exp_name']

    for i in range(args.scale):
        os.makedirs(os.path.join(args.out_dir, 'render_spheric', exp_name, str(2 ** i)), exist_ok=True)

    focal = .5 * args.base_size[0] / np.tan(.5 * args.camera_angle_x)
    render_dataset = RenderGen(focal, args.base_size, args.scale)
    nums = int(len(render_dataset) / args.scale)

    render_loader = DataLoader(render_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for idx, rays in enumerate(tqdm(render_loader)):
            rays = Rays(*[getattr(rays, name).float().to(device) for name in Rays_keys])
            _, height, width, _ = rays.origins.shape  # N H W C
            single_image_rays, _ = rearrange_render_image(rays, args.chunk_size)
            fine_rgb = []
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
            out_path = os.path.join(args.out_dir, 'render_spheric', exp_name, str(int(args.base_size[0] / width)))
            save_images(fine_rgb, distances, accs, out_path, idx%nums)
    generate_video(os.path.join(args.out_dir, 'render_spheric', exp_name))


def generate_video(image_path):
    import glob
    from PIL import Image
    import imageio
    scale_dir = os.listdir(image_path)
    tmp = []
    for s in scale_dir:
        if os.path.isdir(os.path.join(image_path, s)):
            tmp.append(s)
    scale_dir = tmp
    del tmp
    for i in range(len(scale_dir)):
        imgs = []
        images = glob.glob(os.path.join(image_path, str(2 ** i), '*_rgb.png'))
        if len(images) == 0:
            continue
        images.sort()
        for img_file in images:
            img = np.array(Image.open(img_file))
            imgs.append(img.astype(np.uint8))
        imgs += imgs[::-1]
        filename = 'video_{}.mov'.format(str(2 ** i))
        imageio.mimwrite(os.path.join(image_path, str(2 ** i), filename), imgs, fps=40, quality=10)
        print('generate video in {}'.format(os.path.join(image_path, str(2 ** i), filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="Path to ckpt.")
    parser.add_argument("--out_dir", help="Output directory.", type=str, required=True)
    parser.add_argument("--chunk_size", help="Chunck size for render.", type=int, default=12288)
    parser.add_argument("--white_bkgd", help="Train set image background color.", type=bool, default=True)
    parser.add_argument("--render_images_dir", help="already render image directory.", type=str, default=None)
    parser.add_argument('--scale', help='must specify nums of scale', type=int, required=True)
    parser.add_argument('--base_size', help='source image size', type=list, default=[800, 800])
    parser.add_argument('--camera_angle_x', help='camera_angle_x in source dataset',
                        type=float, default=0.6911112070083618)
    parser.add_argument('--gen_video_only', help='if you have generate image already, you can generate video '
                                                 'and do not need render again', action='store_true')
    args = parser.parse_args()
    if not args.gen_video_only:
        run_render(args)
    else:
        assert args.render_images_dir is not None, \
            'only generate video, you must give the different scale image base dir'
        generate_video(args.render_images_dir)

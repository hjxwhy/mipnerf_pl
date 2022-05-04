import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
from PIL import Image
import collections

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


class MultiScaleCam(Dataset):
    def __init__(self, data_dir, split, white_bkgd=True, batch_type='all_images'):
        super(MultiScaleCam, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.white_bkgd = white_bkgd
        self.batch_type = batch_type
        if not self.check_cache():
            self._load_renderings()
            self._generate_rays()
            if split == 'train':
                # all_images: [all_rays, dim]; single_image: [[image1_rays, dim], [image2_rays, dim], ...]
                self.images = self._flatten(self.images)
                self.rays = namedtuple_map(self._flatten, self.rays)
            else:
                # for val and test phase, keep the image shape
                assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            # self.cache_data()

    def check_cache(self):
        if self.white_bkgd:
            bkgd = 'white'
        else:
            bkgd = 'black'
        cache_path = os.path.join(self.data_dir, '_'.join(['cache', self.split, bkgd, self.batch_type]))
        if os.path.exists(cache_path):
            print('loading cached {} data'.format(self.split))
            if self.batch_type == 'single_image':
                self.images = np.load(os.path.join(cache_path, 'images.npy'), allow_pickle=True).tolist()
                self.rays = Rays(*[np.load(os.path.join(cache_path, name + '.npy'), allow_pickle=True).tolist()
                                   for name in Rays_keys])

            elif self.batch_type == 'all_images':
                self.images = np.load(os.path.join(cache_path, 'images.npy'))
                self.rays = Rays(*[np.load(os.path.join(cache_path, name + '.npy')) for name in Rays_keys])
            else:
                raise NotImplementedError
            return True
        else:
            print('cached {} data not found, regenerate cache data'.format(self.split))
            return False

    def cache_data(self):
        if self.white_bkgd:
            bkgd = 'white'
        else:
            bkgd = 'black'
        cache_path = os.path.join(self.data_dir, '_'.join(['cache', self.split, bkgd, self.batch_type]))
        assert not os.path.exists(cache_path)
        os.mkdir(cache_path)
        if self.batch_type == 'single_image':
            np.save(os.path.join(cache_path, 'images'), np.array(self.images, dtype=object))
            [np.save(os.path.join(cache_path, name), np.array(getattr(self.rays, name), dtype=object)) for name in
             Rays_keys]
        elif self.batch_type == 'all_images':
            np.save(os.path.join(cache_path, 'images'), self.images)
            [np.save(os.path.join(cache_path, name), getattr(self.rays, name)) for name in Rays_keys]
        else:
            raise NotImplementedError

    def _load_renderings(self):
        """Load images from disk."""
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as fp:
            self.meta = json.load(fp)[self.split]
        self.meta = {k: np.array(self.meta[k]) for k in self.meta}
        # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
        images = []
        for relative_path in self.meta['file_path']:
            image_path = os.path.join(self.data_dir, relative_path)
            with open(image_path, 'rb') as image_file:
                image = np.array(Image.open(image_file), dtype=np.float32) / 255.
            if self.white_bkgd:
                # image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
                # pixels with alpha between 0 and 1 has a weird color!
                mask = np.where(image[..., -1] > 1e-6, 1., 0.)[..., None].astype(np.float32)
                image = image[..., :3] * mask + (1. - mask)
            images.append(image[..., :3])
        self.images = images
        # self.n_examples = len(self.images)

    def _generate_rays(self):
        """Generating rays for all images."""
        pix2cam = self.meta['pix2cam'].astype(np.float32)
        cam2world = self.meta['cam2world'].astype(np.float32)
        width = self.meta['width'].astype(np.float32)
        height = self.meta['height'].astype(np.float32)

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                indexing='xy')

        xy = [res2grid(w, h) for w, h in zip(width, height)]
        pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
        camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
        directions = [v @ c2w[:3, :3].T for v, c2w in zip(camera_dirs, cam2world)]
        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape)
            for v, c2w in zip(directions, cam2world)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return [
                np.broadcast_to(x[i], origins[i][..., :1].shape).astype(np.float32)
                for i in range(len(self.images))
            ]

        lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
        near = broadcast_scalar_attribute(self.meta['near'])
        far = broadcast_scalar_attribute(self.meta['far'])

        # Distance from each unit-norm direction vector to its x-axis neighbor.
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

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            x = np.concatenate(x, axis=0)
        return x

    def __len__(self):
        if self.batch_type == 'all_images':
            return self.images.shape[0]
        elif self.batch_type == 'single_image':
            return len(self.images)
        else:
            raise NotImplementedError(f'{self.batch_type} batching strategy is not implemented.')

    def __getitem__(self, index):
        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        return rays, self.images[index]


if __name__ == '__main__':
    multicam = MultiScaleCam('/home/hjx/Documents/multi-blender/chair', 'train', batch_type='all_images')
    # from torch.utils.data import DataLoader
    # loader = DataLoader(multicam)
    # while True:
    #     for iteration, batch in enumerate(loader):
    #         print(iteration)

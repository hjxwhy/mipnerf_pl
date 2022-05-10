# This file is modified from official mipnerf
"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
from os import path
import cv2
import numpy as np
from PIL import Image
import collections
from torch.utils.data import Dataset

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


class BaseDataset(Dataset):
    """BaseDataset Base Class."""

    def __init__(self, data_dir, split, white_bkgd=True, batch_type='all_images'):
        super(BaseDataset, self).__init__()
        self.near = 2
        self.far = 6
        self.split = split
        self.data_dir = data_dir
        self.white_bkgd = white_bkgd
        self.batch_type = batch_type
        self.images = None
        self.rays = None
        self.it = -1
        self.n_examples = 1

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            x = np.concatenate(x, axis=0)
        return x

    def _train_init(self):
        """Initialize training."""

        self._load_renderings()
        self._generate_rays()

        if self.split == 'train':
            assert self.batch_type == 'all_images', 'The batch_type can only be all_images with flatten'
            # flatten the ray and image dimension together.
            self.images = self._flatten(self.images)
            self.rays = namedtuple_map(self._flatten, self.rays)

        else:
            assert self.batch_type == 'single_image', 'The batch_type can only be single_image without flatten'

    def _val_init(self):
        self._load_renderings()
        self._generate_rays()

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_renderings(self):
        raise ValueError('Implement in different dataset.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.split == 'val':
            index = (self.it + 1) % self.n_examples
            self.it += 1
        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        return rays, self.images[index]


class Multicam(BaseDataset):
    """Multicam Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images'):
        super(Multicam, self).__init__(data_dir, split, white_bkgd, batch_type)
        if split == 'train':
            self._train_init()
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()

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
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            images.append(image[..., :3])
        self.images = images
        del images
        self.n_examples = len(self.images)

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
        directions = [(v @ c2w[:3, :3].T).copy() for v, c2w in zip(camera_dirs, cam2world)]
        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
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

        lossmult = broadcast_scalar_attribute(self.meta['lossmult']).copy()
        near = broadcast_scalar_attribute(self.meta['near']).copy()
        far = broadcast_scalar_attribute(self.meta['far']).copy()

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
        del origins, directions, viewdirs, radii, lossmult, near, far, xy, pixel_dirs, camera_dirs


class Blender(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', factor=0):
        super(Blender, self).__init__(data_dir, split, white_bkgd, batch_type)
        self.factor = factor
        if split == 'train':
            self._train_init()
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()

    def _load_renderings(self):
        """Load images from disk."""
        with open(path.join(self.data_dir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
        images = []
        cams = []
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
                elif self.factor > 0:
                    raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                                     'set.'.format(self.factor))
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            if self.white_bkgd:
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            images.append(image[..., :3])

        self.images = images
        del images
        self.h, self.w = self.images[0].shape[:-1]
        self.camtoworlds = cams
        del cams
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.n_examples = len(self.images)

    def _generate_rays(self):
        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
            axis=-1)

        directions = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in self.camtoworlds]

        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(directions, self.camtoworlds)
        ]
        viewdirs = [
            v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return [
                x * np.ones_like(origins[i][..., :1])
                for i in range(len(self.images))
            ]

        lossmults = broadcast_scalar_attribute(1).copy()
        nears = broadcast_scalar_attribute(self.near).copy()
        fars = broadcast_scalar_attribute(self.far).copy()

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
            lossmult=lossmults,
            near=nears,
            far=fars)
        del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs
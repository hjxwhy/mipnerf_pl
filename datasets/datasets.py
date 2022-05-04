# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
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


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions


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
        # if split == 'train':
        #     self._train_init()
        # elif split == 'val':
        #     self._val_init()
        # else:
        #     raise ValueError(
        #         'the split argument should be either \'train\' or \'val\', set'
        #         'to {} here.'.format(split))

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
        self.it = 0

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_renderings(self):
        raise ValueError('Implement in different dataset.')

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        return rays, self.images[index]


class Multicam(BaseDataset):
    """Multicam Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images'):
        super(Multicam, self).__init__(data_dir, split, white_bkgd, batch_type)
        if not self.check_cache():
            if split == 'train':
                self._train_init()
            else:
                # for val and test phase, keep the image shape
                assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
                self._val_init()
            self.cache_data()

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
                # pixels with alpha between 0 and 1 has a weird color!
                # mask = np.where(image[..., -1] > 1e-6, 1., 0.)[..., None].astype(np.float32)
                # image = image[..., :3] * mask + (1. - mask)
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
        del origins, directions, viewdirs, radii, lossmult, near, far, xy, pixel_dirs, camera_dirs


class Blender(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', factor=0):
        super(Blender, self).__init__(data_dir, split, white_bkgd, batch_type)
        self.factor = factor
        if not self.check_cache():
            if split == 'train':
                self._train_init()
            else:
                # for val and test phase, keep the image shape
                assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
                self._val_init()
            self.cache_data()

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
        self.resolution = self.h * self.w
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


class LLFF(BaseDataset):
    """LLFF Dataset."""

    def _load_renderings(self):
        """Load images from disk."""
        # Load images.
        imgdir_suffix = ''
        if config.factor > 0:
            imgdir_suffix = '_{}'.format(config.factor)
            factor = config.factor
        else:
            factor = 1
        imgdir = path.join(self.data_dir, 'images' + imgdir_suffix)
        if not utils.file_exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in sorted(utils.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                images.append(image)
        images = np.stack(images, axis=-1)

        # Load poses and bds.
        with utils.open_file(path.join(self.data_dir, 'poses_bounds.npy'),
                             'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1]:
            raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
                images.shape[-1], poses.shape[-1]))

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / factor

        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale according to a default bd factor.
        scale = 1. / (bds.min() * .75)
        poses[:, :3, 3] *= scale
        bds *= scale

        # Recenter poses.
        poses = self._recenter_poses(poses)

        # Generate a spiral/spherical ray path for rendering videos.
        if config.spherify:
            poses = self._generate_spherical_poses(poses, bds)
            self.spherify = True
        else:
            self.spherify = False
        if not config.spherify and self.split == 'test':
            self._generate_spiral_poses(poses, bds)

        # Select the split.
        i_test = np.arange(images.shape[0])[::config.llffhold]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test])
        if self.split == 'train':
            indices = i_train
        else:
            indices = i_test
        images = images[indices]
        poses = poses[indices]

        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        if config.render_path:
            self.n_examples = self.render_poses.shape[0]
        else:
            self.n_examples = images.shape[0]

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        if self.split == 'test':
            n_render_poses = self.render_poses.shape[0]
            self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                              axis=0)

        super()._generate_rays()

        if not self.spherify:
            ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                         self.rays.directions,
                                                         self.focal, self.w, self.h)

            mat = ndc_origins
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
            dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

            dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
            dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

            ones = np.ones_like(ndc_origins[..., :1])
            self.rays = Rays(
                origins=ndc_origins,
                directions=ndc_directions,
                viewdirs=self.rays.directions,
                radii=radii,
                lossmult=ones,
                near=ones * self.near,
                far=ones * self.far)

        # Split poses from the dataset and generated poses
        if self.split == 'test':
            self.camtoworlds = self.camtoworlds[n_render_poses:]
            split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            split0, split1 = zip(*split)
            self.render_rays = utils.Rays(*split0)
            self.rays = utils.Rays(*split1)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable 'focus depth' for this dataset.
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w_path[:, 4:5]
        zrate = .5
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(c2w[:3, :4], (np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0., 2. * np.pi, 120):
            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])
            vec2 = self._normalize(camorigin)
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate([
            new_poses,
            np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        if self.split == 'test':
            self.render_poses = new_poses[:, :3, :4]
        return poses_reset


# dataset_dict = {
#     'blender': Blender,
#     'llff': LLFF,
#     'multicam': Multicam,
# }
if __name__ == '__main__':
    # data_dir = '/home/hjx/Documents/multi-blender/chair'
    data_dir = '/media/hjx/dataset/nerf_synthetic/nerf_synthetic/chair'
    # multicam = Multicam(data_dir, split='val', batch_type='single_image')
    # multicam = Multicam(data_dir)
    multicam = Blender(data_dir)
    # multicam = Blender(data_dir, split='val', batch_type='single_image')

    from torch.utils.data import DataLoader

    loader = DataLoader(multicam, batch_size=4096)
    while True:
        for iteration, batch in enumerate(loader):
            print(batch[0].origins.shape)

# This file is modified from official mipnerf
"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
from os import path
import cv2
import numpy as np
import torch
from PIL import Image
import collections
from torch.utils.data import Dataset
import pyexr
from skimage.transform import resize
from skimage import filters
from utils.vis import l2_normalize_np
from skimage import color
from scipy.ndimage import gaussian_filter

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'depth', 'normal', 'mask', 'depth_vars'))
Rays_keys = Rays._fields


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


class BaseDataset(Dataset):
    """BaseDataset Base Class."""

    def __init__(self, data_dir, split, white_bkgd=True, batch_type='all_images', factor=0):
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
        self.it_cam = -1
        self.n_examples = 1
        self.factor = factor

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            x = np.concatenate(x, axis=0)
        return x

    def _train_init(self, num_images):
        """Initialize training."""

        self._load_renderings(num_images)
        self._generate_rays()

        if self.split == 'train':
            assert self.batch_type == 'all_images', 'The batch_type can only be all_images with flatten'
            # flatten the ray and image dimension together.
            self.images = self._flatten(self.images)
            self.rays = namedtuple_map(self._flatten, self.rays)

        else:
            assert self.batch_type == 'single_image', 'The batch_type can only be single_image without flatten'

    def _val_init(self):
        self._load_renderings(None) #always load all val images
        self._generate_rays()

    def _generate_renderings(self):
        """Generating renderings for a camera path."""
        raise ValueError('Implement in different dataset.')

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_renderings(self):
        raise ValueError('Implement in different dataset.')

    def __len__(self):
        if self.split == 'cam': return self.n_examples
        else: 
            return len(self.images)
    
    def _fake_rays(self, index):
        raise ValueError('Implement in different dataset.')

    def __getitem__(self, index):
        if self.split == 'test':
            index = (self.it + 1) % self.n_examples
            self.it += 1
        if self.split == 'cam':
            #index = (self.it_cam + 1) % self.n_examples
            #self.it_cam += 1
            self._fake_rays(index)  
            rays = Rays(*[getattr(self.rays, key)[0] for key in Rays_keys])
            return rays, self.images[0]
        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        return rays, self.images[index]


class Multicam(BaseDataset):
    """Multicam Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', num_images=None):
        super(Multicam, self).__init__(data_dir, split, white_bkgd, batch_type)
        if split == 'train':
            self._train_init(num_images)
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()

    def _load_renderings(self, num_images = None):
        """Load images from disk."""
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as fp:
            self.meta = json.load(fp)[self.split]
        self.meta = {k: np.array(self.meta[k]) for k in self.meta[num_images]}
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

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', factor=0, num_images=None): #set back to 2
        super(Blender, self).__init__(data_dir, split, white_bkgd, batch_type, factor)
        if split == 'train':
            self._train_init(num_images)
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()

    def _load_depthmaps(self, image_path):
        depth = pyexr.open(image_path.replace('.png','_depth_0001.exr')).get('R') #(h,w,1)
        if self.factor == 2: 
            [halfres_h, halfres_w] = [hw // 2 for hw in depth.shape[:2]]        
            depth = resize(depth, (halfres_h, halfres_w), order=0, anti_aliasing=False)
        depth[depth > 100] = 0
        #depth = filters.gaussian(depth, sigma=1.0, truncate=3.)
        #depth[depth > 100] = 0  #values >= 65504 (float16 max) are in fact zero      
        return depth
        #depth = self.transform(depth).flatten().unsqueeze(1) #/ self.scale #(h*w/4,1)

    def _load_normalmaps(self, image_path):
        normals = pyexr.open(image_path.replace('.png','_normal_0001.exr')).get() #(h,w,1)
        mask = (~np.any(normals, axis=-1))
        if self.factor == 2: 
            [halfres_h, halfres_w] = [hw // 2 for hw in normals.shape[:2]]        
            normals = resize(normals, (halfres_h, halfres_w), order=0, anti_aliasing=True)
        normals = l2_normalize_np(normals)
        #normals = (normals + 1)/2
        normals[mask] = 0
        return normals
        #depth = self.transform(depth).flatten().unsqueeze(1) #/ self.scale #(h*w/4,1)
                

    def _load_renderings(self, num_images):
        """Load images from disk."""
        with open(path.join(self.data_dir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
        images = []
        cams = []
        depths = []
        normals = []
        for i in range(len(meta['frames'][:num_images])):
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
            depths.append(self._load_depthmaps(fname))
            normals.append(self._load_normalmaps(fname))

        self.images = images
        self.depths = depths
        self.normals = normals
        #del images
        self.h, self.w = self.images[0].shape[:-1]
        self.camtoworlds = cams
        #del cams
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

        depth = [d for d in self.depths]
        normal = [n for n in self.normals]
        mask = [(d > 0).astype(np.float32) for d in self.depths]

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmults,
            near=nears,
            far=fars,
            depth=depth,
            normal=normal,
            mask = mask)
        #del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs

class Matterport(BaseDataset):
    """Matterport Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', factor=0, num_images=None): #set back to 2
        super(Matterport, self).__init__(data_dir, split, white_bkgd, batch_type, factor)
        if split == 'train':
            self._train_init(num_images)
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()

    def _load_depthmaps(self, image_path):
        image_path = image_path.replace('rgb','target_depth')
        depth = np.array(Image.open(image_path.replace('jpg','png'))) / 1000 #(h,w,1)
        if self.factor == 2: 
            [halfres_h, halfres_w] = [hw // 2 for hw in depth.shape[:2]]        
            depth = resize(depth, (halfres_h, halfres_w), order=0, anti_aliasing=False)
        depth[depth > 6] = 0
        #depth = filters.gaussian(depth, sigma=1.0, truncate=3.)
        #depth[depth > 100] = 0  #values >= 65504 (float16 max) are in fact zero      
        return depth
        #depth = self.transform(depth).flatten().unsqueeze(1) #/ self.scale #(h*w/4,1)

    def _load_normalmaps(self, image_path):
        try:
            normals = np.array(Image.open(image_path.replace('rgb','normal'))) #(h,w,1)
        except FileNotFoundError:
            normals = np.array(Image.open(image_path))

        mask = (~np.any(normals, axis=-1))
        if self.factor == 2: 
            [halfres_h, halfres_w] = [hw // 2 for hw in normals.shape[:2]]        
            normals = resize(normals, (halfres_h, halfres_w), order=0, anti_aliasing=True)
        normals = l2_normalize_np(normals)
        #normals = (normals + 1)/2
        normals[mask] = 0
        return normals
        #depth = self.transform(depth).flatten().unsqueeze(1) #/ self.scale #(h*w/4,1)
                

    def _load_renderings(self, num_images):
        """Load images from disk."""
        with open(path.join(self.data_dir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
        images = []
        cams = []
        depths = []
        normals = []
        for i in range(len(meta['frames'][:num_images])):
            frame = meta['frames'][i]
            fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
                elif self.factor > 0:
                    raise ValueError('Matterport dataset only supports factor=0 or 2, {} '
                                     'set.'.format(self.factor))
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            if self.white_bkgd and image.shape[-1] > 3:
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            images.append(image[..., :3])
            depths.append(self._load_depthmaps(fname))
            normals.append(self._load_normalmaps(fname))

        self.images = images
        self.depths = depths
        self.normals = normals
        #del images
        self.h, self.w = self.images[0].shape[:-1]
        self.camtoworlds = cams
        #del cams
        #camera_angle_x = float(meta['camera_angle_x'])
        #self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.focal = float(meta['fx'])
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
        nears = broadcast_scalar_attribute(self.meta['near']).copy()
        fars = broadcast_scalar_attribute(self.meta['far']).copy()

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        depth = [d for d in self.depths]
        normal = [(n @ c2w[:3, :3].T).copy() for n, c2w in zip(self.normals, self.camtoworlds)]
        mask = [(d > 0).astype(np.float32) for d in self.depths]

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmults,
            near=nears,
            far=fars,
            depth=depth,
            normal=normal,
            mask = mask)
        #del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs

class ScanNet(BaseDataset):
    """ScanNet Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', factor=0, num_images=None): #set back to 2
        super(ScanNet, self).__init__(data_dir, split, white_bkgd, batch_type, factor)
        if split == 'train':
            self._train_init(num_images)
        elif split =='test':
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()
        """elif split =='cam':
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._init_cam()"""

    def load_omnidata_depth(self, image_path):
        image_path = image_path.replace('rgb','omni')
        #omni_depth = (255 - omni_depth)/255
        #omni_depth = cv2.imread(image_path.replace('.jpg','_depth.png'), cv2.IMREAD_ANYDEPTH)
        omni_depth = np.load(image_path.replace('.jpg','_depth.npy'))
        #omni_depth = omni_depth
        return omni_depth

    def prepare_depthmap(self, depth, omni_depth, max_depth):

        d_mask = (depth > max_depth) | (depth < 0.1)  
        """center_mask = np.zeros_like(depth)
        center_mask[100:-100, 50:-50] = 1
        center_mask = center_mask > 0
        d_mask = d_mask & center_mask"""
        #pick_mask = (depth < max_depth * 0.85) & (depth > 0.1 / 0.85)  
        #### just neighbors for fitting --> doesn't make a difference
        # w = 2*int(truncate*sigma + 0.5) + 1 ## Filter width
        # this filter corresponds to 8x8: gaussian_filter(d_mask.astype(np.float64), sigma=0.66, mode='reflect', cval=0.0, truncate=4.0) 
        # 3x3 filter is 
        #blurmask = gaussian_filter(d_mask.astype(np.float64), sigma=1.3, mode='reflect', cval=0.0, truncate=2.0) 
        #blurmask = blurmask > 0
        #neighbor = d_mask ^ blurmask  #logical XOR
        #d = depth[neighbor].flatten().astype(np.float64)
        #o = omni_depth[neighbor].flatten().astype(np.float64)  
          
        #print('d_max', depth[~d_mask].max(), 'd_min:', depth[~d_mask].min())
        # Least squares alignment of depthmaps, using valid pixels to find scale and offset
        d = depth[~d_mask].flatten().astype(np.float64)
        o = omni_depth[~d_mask].flatten().astype(np.float64)
        A = np.linalg.inv( np.array([[(o**2).sum(), o.sum()],[o.sum(), np.ones_like(o).sum()]]) )
        B = np.array([np.dot(o, d), np.dot(d, np.ones_like(d))])
        h = A @ B

        # Depth = scale * omnidepth + offset
        scaled_od = h[0] * omni_depth + h[1]
        #depth[d_mask] = h[0] * omni_depth[d_mask] + h[1]
        #depth = h[0] * omni_depth + h[1]
        od_mean_dev = ((scaled_od)).mean()
        od_mean_std = np.sqrt(((scaled_od - od_mean_dev) **2).mean())
        print('mean', od_mean_dev, 'std:', od_mean_std, 'd_max:', depth.max(), 'd_min: ', depth.min(), 'od_max', (scaled_od).max(), 'od_min:', (scaled_od).min())
        

        #uncertainty on interpolated pixels
        #depth_var = np.ones_like(depth) * 0.02
        depth_var = np.zeros_like(depth)
        x = np.linspace(-2, 2, depth.shape[0])
        y = np.linspace(-2, 2, depth.shape[1])
        xs, ys = np.meshgrid(x, y, indexing='ij')
        zs = np.sqrt(xs**2 + ys**2 + 1) 
        depth_var = zs/100

        depth_var[d_mask] = np.where(0.5*od_mean_std > depth_var[d_mask], 0.5*od_mean_std , depth_var[d_mask])
        #depth_var[d_mask] = od_mean_std
        depth_var = depth_var[..., None].astype(np.float32)
        depth = depth[..., None].astype(np.float32)
        return depth, depth_var

    def _load_depthmaps(self, image_path, max_depth):
        omni_depth = self.load_omnidata_depth(image_path)
        image_path = image_path.replace('rgb','target_depth')
        depth = np.array(Image.open(image_path.replace('jpg','png'))) / 1000 #(h,w,1)
        if self.factor == 2: 
            [halfres_h, halfres_w] = [hw // 2 for hw in depth.shape[:2]]        
            depth = resize(depth, (halfres_h, halfres_w), order=0, anti_aliasing=False)

        depth, depth_var = self.prepare_depthmap(depth, omni_depth, max_depth)
        #depth = filters.gaussian(depth, sigma=1.0, truncate=3.)
        #depth[depth > 100] = 0  #values >= 65504 (float16 max) are in fact zero      
        return depth, depth_var
        #depth = self.transform(depth).flatten().unsqueeze(1) #/ self.scale #(h*w/4,1)

    def _load_normalmaps(self, image_path):
        image_path = image_path.replace('rgb','omni')
        bgr = cv2.imread(image_path.replace('.jpg','_normal.png')) 
        normals = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255
        normals = (normals - 0.5) * 2
        #test_sum = np.linalg.norm(normals, axis=-1)
        #normals = l2_normalize_np(normals)
        #test_sum = np.linalg.norm(normals, axis=-1)
        #normals[..., 2] *= -1
        mask = (~np.any(normals, axis=-1))
        #normals = l2_normalize_np(normals)
        normals[mask] = 0

        return normals

        # nerf space: +X right, +Y down, +Z from screen to me

        """
        def transform_normal_cam():
        '''
            2D3DS space: +X right, +Y down, +Z from screen to me
            Pytorch3D space: +X left, +Y up, +Z from me to screen
        '''
        totensor = transforms.ToTensor()
        def _thunk(x):
            x2 = -(totensor(x) - 0.5) * 2.0
            x2[-1,...] *= -1
            return x2
        return _thunk
        """
    def _init_cam(self):
        with open(path.join(self.data_dir, 'transforms_video.json'), 'r') as fp:
            meta = json.load(fp)
        self.focal = float(meta['frames'][0]['fx'])
        self.n_examples = len(meta['frames'])
        self.h, self.w = (624, 468) #self.images[0].shape[:-1]
        self.images = [np.zeros((self.h, self.w, 3))]
        self.depths = [(self.images[0][..., 0])[..., None]]
        self.depth_vars = self.depths
        self.normals = self.images
        self.near = meta['near']
        self.far = meta['far']        

    def _fake_rays(self, index):
        """Load camera parameters from disk."""
        with open(path.join(self.data_dir, 'transforms_video.json'), 'r') as fp:
            meta = json.load(fp)

        frame = meta['frames'][index]                
        self.camtoworlds = [np.array(frame['transform_matrix'], dtype=np.float32)]

        """Generating rays for one image."""
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
                x * np.ones_like(origins[0][..., :1])
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

        """depth = [d for d in self.depths]
        depth_vars = [dv for dv in self.depth_vars]
        normal = [n for n in self.normals] #transform to view
        mask = [(d > 0).astype(np.float32) for d in self.depths]"""

        depth = self.depths
        depth_vars = self.depth_vars
        normal = self.normals #transform to view
        mask = [(depth > 0).astype(np.float32) for depth in self.depths]

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmults,
            near=nears,
            far=fars,
            depth=depth,
            normal=normal,
            depth_vars=depth_vars,
            mask = mask)

    def _load_renderings(self, num_images):
        """Load images from disk."""
        with open(path.join(self.data_dir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
        with open(path.join(self.data_dir, 'config.json'), 'r') as cp:
            cfg = json.load(cp)
        max_depth = cfg["max_depth"]
        images = []
        cams = []
        depths = []
        depth_vars = []
        normals = []
        #scale = 1/8
        scale = 1.0
        scale_mat = np.diag(np.array([scale,]*4, dtype=np.float32)) #@ pose
        for i in range(len(meta['frames'][:num_images])):
            frame = meta['frames'][i]
            fname = os.path.join(self.data_dir, frame['file_path'])
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
                elif self.factor > 0:
                    raise ValueError('ScanNet dataset only supports factor=0 or 2, {} '
                                     'set.'.format(self.factor))
                
            cams.append(scale_mat @ np.array(frame['transform_matrix'], dtype=np.float32))
            if self.white_bkgd and image.shape[-1] > 3:
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            images.append(image[..., :3].astype(np.float16))
            depth, depth_var = self._load_depthmaps(fname, max_depth)
            depths.append(depth)
            depth_vars.append(depth_var)
            normals.append(self._load_normalmaps(fname))
            
        """pose = np.array(cams)
        min_vertices = pose[:, :3, 3].min(axis=0)
        max_vertices = pose[:, :3, 3].max(axis=0)
        center = (min_vertices + max_vertices) / 2.
        scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
        
        print(self.split, 'min_pose:', min_vertices, 'max_pose:', max_vertices, 'scale:', scale, 'center', center)"""
    
        self.images = images
        self.depths = depths
        self.depth_vars = depth_vars
        self.normals = normals
        self.near = meta['near']
        self.far = meta['far']
        #del images
        self.h, self.w = self.images[0].shape[:-1]
        self.camtoworlds = cams
        #del cams
        #camera_angle_x = float(meta['camera_angle_x'])
        #self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.focal = float(meta['frames'][0]['fx'])
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

        depth = [d for d in self.depths]
        depth_vars = [dv for dv in self.depth_vars]
        #normal = [n for n in self.normals] #transform to view
        normal = [l2_normalize_np(n) for n in self.normals]
        #normal = [l2_normalize_np(((n @ c2w[:3, :3].T)).copy()) for n, c2w in zip(self.normals, self.camtoworlds)]
        #normal_reordered = np.asarray(normal)
        #normal_reordered[:, :, :, [1, 2]] = normal_reordered[:, :, :, [2,1]]
        #normal_reordered[:, :, :, 1] = - normal_reordered[:, :, :, 1]
        #normal = np.split(normal_reordered, normal_reordered.shape[0])
        #normal = [n.squeeze() for n in normal]
        mask = [(d > 0).astype(np.float32) for d in self.depths]

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=lossmults,
            near=nears,
            far=fars,
            depth=depth,
            normal=normal,
            depth_vars=depth_vars,
            mask = mask)
        #del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs


class RealData360(BaseDataset):
    """RealData360 Dataset."""

    def __init__(self, data_dir, split='train', white_bkgd=True, batch_type='all_images', factor=0):
        super(RealData360, self).__init__(data_dir, split, white_bkgd, batch_type, factor)
        if split == 'train':
            self._train_init()
        else:
            # for val and test phase, keep the image shape
            assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
            self._val_init()

    def _load_renderings(self):
        """Load images from disk."""
        # Load images.
        imgdir_suffix = ''
        if self.factor > 0:
            imgdir_suffix = '_{}'.format(self.factor)
        else:
            factor = 1
        imgdir = path.join(self.data_dir, 'images' + imgdir_suffix)
        if not path.exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in sorted(os.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images = []
        for imgfile in imgfiles:
            with open(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                images.append(image)
        images = np.stack(images, axis=-1)

        # Load poses and bds.
        with open(path.join(self.data_dir, 'poses_bounds.npy'), 'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1]:
            raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
                images.shape[-1], poses.shape[-1]))

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / self.factor

        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Recenter poses.
        poses = self._recenter_poses(poses)
        poses = self._spherify_poses(poses)
        # Select the split.
        i_test = np.arange(images.shape[0])[::8]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test])
        if self.split == 'train':
            indices = i_train
        else:
            indices = i_test
        images = images[indices]
        poses = poses[indices]
        bds = bds[indices]
        self._read_camera()
        self.K[:2, :] /= self.factor
        self.K_inv = np.linalg.inv(self.K)
        self.K_inv[1:, :] *= -1
        self.bds = bds
        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        self.n_examples = images.shape[0]

    def _generate_rays(self):
        """Generating rays for all images."""

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
                np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
                indexing='xy')

        xy = res2grid(self.w, self.h)
        pixel_dirs = np.stack([xy[0], xy[1], np.ones_like(xy[0])], axis=-1)
        camera_dirs = pixel_dirs @ self.K_inv.T
        directions = ((camera_dirs[None, ..., None, :] * self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                                  directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.sqrt(
            np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)
        ones = np.ones_like(origins[..., :1])
        near_fars = np.broadcast_to(self.bds[:, None, None, :], [*directions.shape[:-1], 2])
        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=near_fars[..., 0:1],
            far=near_fars[..., 1:2])
        del origins, directions, viewdirs, radii, near_fars, ones, xy, pixel_dirs, camera_dirs

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

    def _read_camera(self):
        import struct
        # modified from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py

        def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
            """Read and unpack the next bytes from a binary file.
            :param fid:
            :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
            :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
            :param endian_character: Any of {@, =, <, >, !}
            :return: Tuple of read and unpacked values.
            """
            data = fid.read(num_bytes)
            return struct.unpack(endian_character + format_char_sequence, data)

        with open(path.join(self.data_dir, 'sparse', '0', 'cameras.bin'), "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            num_params = 4
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            self.K = np.array([[params[0], 0, params[2]],
                               [0, params[1], params[3]],
                               [0, 0, 1]])

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

    def _spherify_poses(self, poses):
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
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        return poses_reset


    



def contract(x):
    # return x
    return (2 - 1/torch.norm(x, dim=-1, keepdim=True)) * x/torch.norm(x, dim=-1, keepdim=True)

if __name__ == '__main__':
    # realdata360 = RealData360('/media/hjx/dataset/360_v2/garden', split='val', batch_type='single_image', factor=8)
    # from utils.visualize_cameras import visualize_cameras
    #
    # colored_camera_dicts = [([0, 1, 0], realdata360.cams_dict)]
    # visualize_cameras(colored_camera_dicts, 1)
    torch.manual_seed(0)
    from torch.autograd.functional import jvp, jacobian
    inputs = torch.randn(1, 2, 3, requires_grad=True)
    print(inputs)
    out = contract(inputs)
    print(out)
    jvpout = jvp(contract, inputs, torch.ones_like(inputs))
    # print(jvpout[0])
    # print(jvpout[1])
    print(jacobian(contract, inputs))

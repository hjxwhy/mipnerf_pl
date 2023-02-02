import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
#import pyexr
from skimage.transform import resize
import collections

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields

class Blender(Dataset):
    def __init__(self, data_dir, split='train', img_wh=(400, 400), num_images=3, white_bkgd=True, batch_type='all_images', factor=2):
        self.root_dir = data_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.num_images = num_images
        self.define_transforms()
        self.read_meta()
        self.white_back = white_bkgd

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh
        self.scale = 1#0.86
        # bounds, common for all scenes
        self.near = 2.0 #/ self.scale
        self.far = 6.0 #/ self.scale
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            factor = 120 // self.num_images
            self.normal_correction = np.array([[1,0,0],[0,0,1],[0,-1,0]])
            self.meta['frames'] = self.meta['frames'][:self.num_images] * factor
            for frame in self.meta['frames']:
                #scale_neus = np.diag([2, 2, 2, 1])
                #scale = np.diag([1,1,1,1])
                pose = (np.array(frame['transform_matrix']))[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                #depth = torch.zeros_like(img[0]).reshape(-1,1)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

                #depth = pyexr.open(image_path.replace('.png','_depth_0001.exr')).get('R') #(h,w,1)
                #depth = pyexr.open(image_path.replace('.png','_distance.exr')).get() #(h,w,1)
                
                #depth = resize(depth, self.img_wh, order=0, anti_aliasing=False)
                #depth[depth > 10] = 0
                #noise = filters.scharr(depth)
                #noise = filters.gaussian(noise, sigma=3)
                #noise = torch.from_numpy((noise-noise.min()) / (noise.max() - noise.min())).flatten().unsqueeze(1)*0.5
                
                #depth = self.transform(depth).flatten().unsqueeze(1) #/ self.scale #(h*w/4,1)
                #depth_uncertainty = (torch.where(depth > 0, 0.5, 0) + noise)
                
                #normal = pyexr.open(image_path.replace('.png','_normal_0001.exr')).get()
                #normal = resize(normal, self.img_wh, order=0, anti_aliasing=False)
                #normal = self.transform(normal).view(3, -1).permute(1, 0) #--> (BS, 3)
                #normal = normal @ pose[:, :3] #use this for view transform. We dont use it because normals are checked in world coordinates
                #normal = (normal + 1)/2  #transform from world to view and scale from [-1,1] to [0, 1] for visualization
                #normal = normal.to(torch.float32)

                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w, scale = self.scale) # both (h*w, 3)
                #near, far = self.near_far_from_sphere(rays_o, rays_d)
                near, far = self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])
                # Distance from each unit-norm direction vector to its x-axis neighbor.
                viewdirs = rays_d.clone().reshape(400, 400, 3)
                #dx = torch.sqrt(torch.sum((viewdirs[:-1, :, :] - viewdirs[1:, :, :]) ** 2, -1))
                #dx = [torch.concatenate([v, v[-2:-1, :]], 0) for v in dx]
                # Cut the distance in half, and then round it out so that it's
                # halfway between inscribed by / circumscribed about the pixel.
                #radii = torch.cat([v[..., None] * 2 / np.sqrt(12) for v in dx], 0).reshape(-1,1)
                radii = torch.ones((400*400, 1)) * 1e-3

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             radii,
                                             near,                                             
                                             far,
                                             viewdirs.view(-1,3)
                                             ],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)            
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = self.all_rays[idx], self.all_rgbs[idx]

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = (torch.FloatTensor(frame['transform_matrix']))[:3, :4]
            
            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            #depth = pyexr.open(img_path.replace('.png','_depth_0001.exr')).get('R') #(h,w,1)

            #depth = resize(depth, self.img_wh, order=0, anti_aliasing=False)
            #depth[depth > 10] = 0
            #noise = filters.scharr(depth)
            #noise = filters.gaussian(noise, sigma=3)
            #noise = torch.from_numpy((noise-noise.min()) / (noise.max() - noise.min())).flatten().unsqueeze(1)
            
            #depth = self.transform(depth).flatten().unsqueeze(1) #/ self.scale#(h*w/4,1)
            #depth_uncertainty = (torch.where(depth > 0, 0.5, 0) + noise)*0.5     

            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            #depth = torch.zeros_like(img[0]).reshape(-1,1)

            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB   

            #normal = pyexr.open(img_path.replace('.png','_normal_0001.exr')).get().astype(np.float32)
            #normal = resize(normal, self.img_wh, order=0, anti_aliasing=False)
            #normal = self.transform(normal).view(3, -1).permute(1, 0)
            #normal = normal @ pose[:, :3] #use this for view transform. We dont use it because normals are checked in world coordinates
            #normal = (normal + 1)/2  #transform from world to view and scale from [-1,1] to [0, 1] for visualization
            #normal = normal.to(torch.float32)
            
            rays_o, rays_d = get_rays(self.directions, c2w, scale = self.scale)
            #near, far = self.near_far_from_sphere(rays_o, rays_d)
            near, far = self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = [np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in rays_d]
            dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

            rays = torch.cat([rays_o, rays_d, 
                              near,
                              far                             
                              ],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    directions = \
        torch.stack([(i-W/2 + 0.5)/focal, -(j - H/2 + 0.5)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w, scale=1.0): #2/2.4 for tractor
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate and scale into unit space
    c2w= torch.diag(torch.tensor([scale,]*3, dtype=torch.float32)) @c2w
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    #rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

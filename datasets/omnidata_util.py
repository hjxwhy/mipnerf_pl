import numpy as np
from PIL import Image
from glob import glob
from pathlib import Path
import cv2
#from skimage.transform import resize


def convert_to_square_image(path_in, path_out):
    image = np.array(Image.open(path_in))
    shape = image.shape[:-1]
    if shape[0] > shape[1]:
        size = shape[0]
    else:
        size = shape[1]

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:shape[0], :shape[1], :] = image
    image_out = Image.fromarray(canvas, "RGB")
    image_out.save(path_out)

def crop(path_in, path_out, size=256):
    image = np.array(Image.open(path_in))
    h, w = image.shape[0], image.shape[1]

    A = image[:size, :size]
    B = image[:size, (w-size):]
    C = image[(h-size):, :size]
    D = image[(h-size):, (w-size):]

    E = image[(size//2):((size//2)+size), :size]
    F = image[(size//2):((size//2)+size), (size//2):((size//2)+size)]

    G = image[(size//2):((size//2)+size), (w-size):]
    H = image[(h-size):, (size//2):((size//2)+size)]
    I = image[(h-size):, (w-size):]

    crops = np.stack([A, B, C, D, E, F, G, H, I], axis=0)
    file_id = path_in.split('/')[-1].rstrip('.jpg')
    path_out = path_out.replace('omnidata', f'omnidata/{file_id}')

    img_dir = '/'.join(path_out.split('/')[:-1])
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    for i in range(crops.shape[0]):
        image_out = Image.fromarray(crops[i], "RGB")
        image_out.save(path_out.replace('.jpg', f'_{i}.jpg'))

def stitch(path_in, path_out, size=256):
    
    imageFiles = glob(path_in + '/*')
    images = []
    for filename in imageFiles:
        img = cv2.imread(filename)
        images.append(img)

    cv2.ocl.setUseOpenCL(False)
    stitcher = cv2.Stitcher_create()
    status, result = stitcher.stitch(images)             

    cv2.imwrite(path_out + '/stitched_images.png',result)
    #load 4 depthmaps
    #stitch

def batched_conversion(path_in_folder, path_out_folder):
    files = glob(path_in_folder + '*.jpg')
    Path(path_out_folder).mkdir(parents=True, exist_ok=True)

    for file in files:
        filename = file.split('/')[-1]
        path_out = path_out_folder + filename.replace('.jpg', '_square.jpg')
        convert_to_square_image(file, path_out)

def warp(path_in, path_out, sizes=(512, 512), task='depth'):
    if task != 'depth': image = cv2.imread(path_in)
    else: image = cv2.imread(path_in, cv2.IMREAD_ANYDEPTH)
    if sizes == (512, 512): # or task == 'depth' :
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_NEAREST
    image_resized = cv2.resize(image, sizes, interpolation = interpolation)
    #path_out = path_out.replace('_{task}.png', f'_{task}.png')
    cv2.imwrite(path_out, image_resized)

def batched_warp(path_in_folder, path_out_folder, sizes=(512, 512), task='depth'):
    if sizes[0] != 624: file_names = '*.jpg' 
    else: file_names = f'*_{task}.png'
    files = glob(path_in_folder + file_names)
    Path(path_out_folder).mkdir(parents=True, exist_ok=True)

    for file in files:
        filename = file.split('/')[-1]
        path_out = path_out_folder + filename
        warp(file, path_out, sizes, task)

if __name__ == '__main__':
    import subprocess
    import os
    try:
        from subprocess import DEVNULL  # Python 3.
    except ImportError:
        DEVNULL = open(os.devnull, 'wb')
    
    #split = 'test'
    #task = 'normal'
    #scene = '708'
    #prep = False
    scenes = ['708', '710', '738', '758', '781']
    inferences = [True, False]
    splits = ['train', 'test']
    tasks = ['depth', 'normal']
   
    for scene in scenes:
        for split in splits:
            for task in tasks:
                #downscale to input resolution
                sizes = (512, 512)
                folder_in = 'rgb'
                folder_out = 'omnidata'            
                in_path = f'/home/sheldrick/master/data/matterport/scene0{scene}_00/{split}/{folder_in}/'
                out_path = f'/home/sheldrick/master/data/matterport/scene0{scene}_00/{split}/{folder_out}/'
                batched_warp(in_path, out_path, sizes, task)

                #run inference and save results
                omni_path_in = f'/home/sheldrick/master/data/matterport/scene0{scene}_00/{split}/omnidata/'
                omni_path_out = f'/home/sheldrick/master/data/matterport/scene0{scene}_00/{split}/omni/'
                Path(omni_path_out).mkdir(parents=True, exist_ok=True)
                subprocess.run(['python', 'demo.py', '--task', f'{task}', '--img_path', omni_path_in, '--output_path', omni_path_out], check=True)

                #resize to original resolution
                if task == 'normal':
                    sizes = (624, 468)
                    folder_in = 'omni'
                    folder_out = 'omni'
                    in_path = f'/home/sheldrick/master/data/matterport/scene0{scene}_00/{split}/{folder_in}/'
                    out_path = f'/home/sheldrick/master/data/matterport/scene0{scene}_00/{split}/{folder_out}/'
                    batched_warp(in_path, out_path, sizes, task)
                #else:
        #rmdir omnidata
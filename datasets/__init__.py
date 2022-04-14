# from .blender import BlenderDataset
# from .llff import LLFFDataset
from .multi_blender import MultiScaleCam

dataset_dict = {
    # 'blender': BlenderDataset,
    # 'llff': LLFFDataset,
    'multi_blender': MultiScaleCam}

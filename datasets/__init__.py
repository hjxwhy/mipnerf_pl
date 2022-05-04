# from .blender import BlenderDataset
# from .llff import LLFFDataset
from .multi_blender import MultiScaleCam
from .datasets import Blender
dataset_dict = {
    'blender': Blender,
    'multi_blender': MultiScaleCam}

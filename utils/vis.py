import torch
# from visdom import Visdom
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# def visualize_nerf_outputs(
#     nerf_out: dict, viz: Visdom, visdom_env: str
# ):
#     """
#     Visualizes the outputs of the `RadianceFieldRenderer`.
#
#     Args:
#         nerf_out: An output of the validation rendering pass.
#         viz: A visdom connection object.
#         visdom_env: The name of visdom environment for visualization.
#     """
#
#     # Show the coarse and fine renders together with the ground truth images.
#     ims_full = torch.cat(
#         [
#             nerf_out[imvar][0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
#             for imvar in ("rgb_coarse", "rgb_fine", "rgb_gt")
#         ],
#         dim=2,
#     )
#     viz.image(
#         ims_full,
#         env=visdom_env,
#         win="images_full",
#         opts={"title": "coarse | fine | target"},
#     )
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def vis_lr(lr_init: float = 5e-4,
           lr_final: float = 5e-6,
           max_steps: int = 2000000,
           lr_delay_steps: int = 2500,
           lr_delay_mult: float = 0.01):
    def get_lr(last_epoch):
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(last_epoch / lr_delay_steps, 0, 1))

        else:
            delay_rate = 1.
        t = np.clip(last_epoch / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return [delay_rate * log_lerp]

    lr = []
    step = []
    for i in range(max_steps):
        step += [i]
        lr += get_lr(i)
    p = np.stack([step, lr], 1)
    plt.plot(p[:, 0], p[:, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def save_image_tensor(image: torch.tensor, height: int, width: int, save_path: str, nhwc: bool = True):
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.dim() == 3:
        image = image[None, ...]
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    elif image.dim() == 4:
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    elif image.dim() == 2:  # flatten
        assert image.shape[0] == height * width
        image = image.reshape(1, height, width, image.shape[-1])
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    else:
        raise NotImplementedError


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views // 3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0]) > 180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


if __name__ == '__main__':
    # import numpy as np
    # from PIL import Image
    # image_path = 'temp/000_d0.png'
    # with open(image_path, 'rb') as image_file:
    #     image = np.array(Image.open(image_file), dtype=np.float32) / 255.
    # print(image.shape)
    # image_tensor = torch.from_numpy(image)[None, ...]
    # print(image_tensor.shape)
    # image_tensor = image_tensor.permute(0, 3, 1, 2)
    # print(image_tensor.shape)
    # torchvision.utils.save_image(image_tensor, 'test.png')
    # image_o = image = np.array(Image.open('test.png'), dtype=np.float32) / 255.
    # print(np.sum((image_o-image)**2))
    lr = vis_lr(max_steps=2000000)

import torch
from pytorch_lightning import LightningModule
from models.mip_nerf import MipNerf
from utils.loss import calc_mse, calc_psnr
from datasets import dataset_dict
from datasets.multi_blender import Rays_keys, Rays
from utils.lr_schedule import MipLRDecay
from torch.utils.data import DataLoader
from utils.vis import stack_rgb


class MipNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(MipNeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.train_randomized = hparams['train.randomized']
        self.val_randomized = hparams['val.randomized']
        self.white_bkgd = hparams['train.white_bkgd']
        self.val_chunk_size = hparams['val.chunk_size']
        self.batch_size = self.hparams['train.batch_size']
        self.mip_nerf = MipNerf(
            num_samples=hparams['nerf.num_samples'],
            num_levels=hparams['nerf.num_levels'],
            resample_padding=hparams['nerf.resample_padding'],
            stop_resample_grad=hparams['nerf.stop_resample_grad'],
            use_viewdirs=hparams['nerf.use_viewdirs'],
            disparity=hparams['nerf.disparity'],
            ray_shape=hparams['nerf.ray_shape'],
            min_deg_point=hparams['nerf.min_deg_point'],
            max_deg_point=hparams['nerf.max_deg_point'],
            deg_view=hparams['nerf.deg_view'],
            density_activation=hparams['nerf.density_activation'],
            density_noise=hparams['nerf.density_noise'],
            density_bias=hparams['nerf.density_bias'],
            rgb_activation=hparams['nerf.rgb_activation'],
            rgb_padding=hparams['nerf.rgb_padding'],
            disable_integration=hparams['nerf.disable_integration'],
            append_identity=hparams['nerf.append_identity'],
            mlp_net_depth=hparams['nerf.mlp.net_depth'],
            mlp_net_width=hparams['nerf.mlp.net_width'],
            mlp_net_depth_condition=hparams['nerf.mlp.net_depth_condition'],
            mlp_net_width_condition=hparams['nerf.mlp.net_width_condition'],
            mlp_skip_index=hparams['nerf.mlp.skip_index'],
            mlp_num_rgb_channels=hparams['nerf.mlp.num_rgb_channels'],
            mlp_num_density_channels=hparams['nerf.mlp.num_density_channels'],
            mlp_net_activation=hparams['nerf.mlp.net_activation']
        )

    def forward(self, batch_rays: torch.Tensor, randomized: bool, white_bkgd: bool):
        # TODO make a multi chunk
        res = self.mip_nerf(batch_rays, randomized, white_bkgd)  # num_layers result
        return res

    def setup(self, stage):
        dataset = dataset_dict[self.hparams['dataset_name']]

        self.train_dataset = dataset(data_dir=self.hparams['data.path'],
                                     split='train',
                                     white_bkgd=self.hparams['train.white_bkgd'],
                                     batch_type=self.hparams['train.batch_type'],
                                     )
        self.val_dataset = dataset(data_dir=self.hparams['data.path'],
                                   split='val',
                                   white_bkgd=self.hparams['val.white_bkgd'],
                                   batch_type=self.hparams['val.batch_type']
                                   )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mip_nerf.parameters(), lr=self.hparams['optimizer.lr_init'])
        scheduler = MipLRDecay(optimizer, self.hparams['optimizer.lr_init'], self.hparams['optimizer.lr_final'],
                               self.hparams['optimizer.max_steps'], self.hparams['optimizer.lr_delay_steps'],
                               self.hparams['optimizer.lr_delay_mult'])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams['train.num_work'],
                          batch_size=self.hparams['train.batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        # make the dataloader to an iter, so which can val n numbers images one time
        return iter(DataLoader(self.val_dataset,
                               shuffle=False,
                               num_workers=self.hparams['val.num_work'],
                               batch_size=1,  # validate one image (H*W rays) at a time
                               pin_memory=True))

    def training_step(self, batch, batch_nb):
        rays, rgbs = batch
        ret = self(rays, self.train_randomized, self.white_bkgd)
        # calculate loss for coarse and fine
        mask = rays.lossmult
        if self.hparams['loss.disable_multiscale_loss']:
            mask = torch.ones_like(mask)
        losses = []
        for (rgb, _, _) in ret:
            losses.append((mask * (rgb - rgbs[..., :3]) ** 2).sum() / mask.sum())
        # The loss is a sum of coarse and fine MSEs
        mse_corse, mse_fine = losses
        loss = self.hparams['loss.coarse_loss_mult'] * mse_corse + mse_fine
        with torch.no_grad():
            psnrs = []
            for (rgb, _, _) in ret:
                psnrs.append(calc_psnr(rgb, rgbs[..., :3]))
            psnr_corse, psnr_fine = psnrs
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_fine, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        _, rgbs = batch
        rgb_gt = rgbs[..., :3]
        coarse_rgb, fine_rgb, val_mask = self.render_image(batch)

        val_mse_corse = (val_mask * (coarse_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
        val_mse_fine = (val_mask * (fine_rgb - rgb_gt) ** 2).sum() / val_mask.sum()

        val_loss = self.hparams['loss.coarse_loss_mult'] * val_mse_corse + val_mse_fine
        val_psnr_corse = calc_psnr(coarse_rgb, rgb_gt)
        val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)

        log = {'val/loss': val_loss, 'val/psnr': val_psnr_fine}
        stack = stack_rgb(rgb_gt, coarse_rgb, fine_rgb)  # (3, 3, H, W)
        self.logger.experiment.add_images('val/GT_coarse_fine',
                                          stack, self.global_step)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)

    def render_image(self, batch):
        rays, rgbs = batch
        _, height, width, _ = rgbs.shape  # N H W C

        # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
        single_image_rays = [getattr(rays, key) for key in Rays_keys]
        val_mask = single_image_rays[-3]

        # flatten each Rays attribute and put on device
        single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
        # get the amount of full rays of an image
        length = single_image_rays[0].shape[0]
        # divide each Rays attr into N groups according to chunk_size,
        # the length of the last group <= chunk_size
        single_image_rays = [[rays_attr[i:i + self.val_chunk_size] for i in range(0, length, self.val_chunk_size)] for
                             rays_attr in single_image_rays]
        # get N, the N for each Rays attr is the same
        length = len(single_image_rays[0])
        # generate N Rays instances
        single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]

        corse_rgb, fine_rgb = [], []
        with torch.no_grad():
            for batch_rays in single_image_rays:
                (c_rgb, _, _), (f_rgb, _, _) = self(batch_rays, self.val_randomized, self.white_bkgd)
                corse_rgb.append(c_rgb)
                fine_rgb.append(f_rgb)

        corse_rgb = torch.cat(corse_rgb, dim=0)
        fine_rgb = torch.cat(fine_rgb, dim=0)

        corse_rgb = corse_rgb.reshape(1, height, width, corse_rgb.shape[-1])  # N H W C
        fine_rgb = fine_rgb.reshape(1, height, width, fine_rgb.shape[-1])  # N H W C
        return corse_rgb, fine_rgb, val_mask

import torch
from torch import nn
from einops import repeat
from models.mip import sample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering, resample_along_rays, max_dilate_weights
from collections import namedtuple
from utils.vis import l2_normalize
import math
from models.ref_utils import generate_ide_fn
from models.mipnerf360_utils import track_linearize, contract, inv_contract, lift_and_diagonalize, generate_basis
import functorch


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int, prop_mlp: bool = False):
        """
          net_depth: The depth of the first part of MLP.
          net_width: The width of the first part of MLP.
          net_depth_condition: The depth of the second part of MLP.
          net_width_condition: The width of the second part of MLP.
          activation: The activation function.
          skip_index: Add a skip connection to the output of every N layers.
          num_rgb_channels: The number of RGB channels.
          num_density_channels: The number of density channels.
        """
        super(MLP, self).__init__()
        self.prop_mlp = prop_mlp
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i - 1) % skip_index == 0 and i > 1:
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.layers = torch.nn.ModuleList(layers)
        del layers
        self.density_layer = torch.nn.Linear(net_width, num_density_channels)
        _xavier_init(self.density_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)  # extra_layer is not the same as NeRF
        _xavier_init(self.extra_layer)
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.view_layers = torch.nn.Sequential(*layers)
        del layers
        if not prop_mlp:
            self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

    def forward(self, x, view_direction=None):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, num_samples, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        if x.dim() > 1:
            num_samples = x.shape[1]
        else: num_samples = 1
        inputs = x  # [B, N, 2*3*L]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        
        raw_density = self.density_layer(x)
        
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            # view_direction: [B, 2*3*L] -> [B, N, 2*3*L]
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            x = torch.cat([bottleneck, view_direction], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.view_layers(x)

        if view_direction is None:
            return raw_density, x

        elif not self.prop_mlp:
            raw_rgb = self.color_layer(x)
            return raw_rgb, raw_density
        
        else:
            return raw_density
        


class MipNerf(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(self, num_samples: int = 128,
                 num_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_resample_grad: bool = True,
                 use_viewdirs: bool = True,
                 disparity: bool = False,
                 depth_sampling: bool = False,
                 ray_shape: str = 'cone',
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_activation: str = 'softplus',
                 density_noise: float = 0.,
                 density_bias: float = -1.,
                 rgb_activation: str = 'sigmoid',
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,
                 append_identity: bool = True,
                 mlp_net_depth: int = 8,
                 mlp_net_width: int = 256,
                 mlp_net_depth_condition: int = 1,
                 mlp_net_width_condition: int = 128,
                 mlp_skip_index: int = 4,
                 mlp_num_rgb_channels: int = 3,
                 mlp_num_density_channels: int = 1,
                 mlp_net_activation: str = 'relu',
                 prop_mlp: bool = False):
        super(MipNerf, self).__init__()
        self.num_levels = num_levels  # The number of sampling levels.
        self.num_samples = num_samples  # The number of samples per level.
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
        self.depth_sampling = depth_sampling #If True, sample ray more densily around depth
        self.ray_shape = ray_shape  # The shape of cast rays ('cone' or 'cylinder').
        self.disable_integration = disable_integration  # If True, use PE instead of IPE.
        self.min_deg_point = min_deg_point  # Min degree of positional encoding for 3D points.
        self.max_deg_point = max_deg_point  # Max degree of positional encoding for 3D points.
        self.use_viewdirs = use_viewdirs  # If True, use view directions as a condition.
        self.deg_view = deg_view  # Degree of positional encoding for viewdirs.
        self.density_noise = density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding  # Dirichlet/alpha "padding" on the histogram.
        self.stop_resample_grad = stop_resample_grad  # If True, don't backprop across levels')
        mlp_xyz_dim = (max_deg_point - min_deg_point) * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim
        self.mlp = MLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       mlp_xyz_dim, mlp_view_dim, prop_mlp=False)
        if prop_mlp:
            self.prop_mlp = MLP(4, 256, mlp_net_depth_condition, mlp_net_width_condition,
                       8, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       mlp_xyz_dim, mlp_view_dim, prop_mlp=prop_mlp)
        if rgb_activation == 'sigmoid':  # The RGB activation.
            self.rgb_activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.rgb_padding = rgb_padding
        if density_activation == 'softplus':  # Density activation.
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError
        self.chunksize = 1024
        self.depth_network = DepthVarianceNetwork(init_val = 3.0) #0.1 variance 2.30258509
        """self.grad_decoder = nn.Sequential(
          nn.Linear(100,128),
          nn.ReLU(),
          nn.Linear(128,128),
          nn.ReLU(),
          nn.Linear(128,128),
          nn.ReLU(),
          nn.Linear(128,3),
          nn.Tanh(),
        )
        self.dir_enc_fun = generate_ide_fn(4)"""
        
        #self.pos_basis_t = torch.from_numpy(generate_basis('icosahedron', 2)).to(torch.float16).to(device='cuda:0') # self.basis_shape : 'icosahedron'  // self.basis_subdivisions : 2

    def forward(self, rays: namedtuple, randomized: bool, white_bkgd: bool, compute_normals: bool = False, eps = 1.0):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """
        batch_size = rays[0].shape[0]
        w_env = torch.empty((self.num_levels - 1, batch_size, self.num_samples), device=rays[0].device)
        t_env = torch.empty((self.num_levels - 1, batch_size, self.num_samples + 1), device=rays[0].device)
        ret = torch.empty((self.num_levels, batch_size, 3), device=rays[0].device)
        #weights = torch.ones_like(rays.near)
        prod_num_samples = 1
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                if self.depth_sampling: s = self.depth_network(torch.zeros([1, 3], device=rays[0].device))[:, :1].clip(1e-6, 1e6)
                else: s = None
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                    rays,
                    s
                )
            else:
                """train_frac = 1.0
                self.dilation_bias = 0.0025
                self.anneal_slope = 10.
                dilation = self.dilation_bias + self.dilation_multiplier * (
                    rays.near - rays.far) / prod_num_samples

                # Record the product of the number of samples seen so far.
                prod_num_samples *= num_samples
                sdist, weights = max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    #domain=(init_s_near, init_s_far),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

                # Optionally anneal the weights as a function of training iteration.
                if self.anneal_slope > 0:
                    # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                    bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                    anneal = bias(train_frac, self.anneal_slope)"""

                if i_level == self.num_levels - 1: 
                    n_samples = 32
                else:
                    n_samples = self.num_samples

                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                    num_samples = n_samples,
                )
                # t_samples: (BS, n_samples+1); mean_covs: (tuple: ((1024, 128, 3), (1024, 128, 3)))
            if self.disable_integration:
                means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
            
            #self.warping_fn = None
            """self.warping_fn = contract
            if self.warping_fn is not None:
                # (means, covs) -> ((BS, N_samples, 3), (BS, N_samples, 3, 3))
                means, covs = track_linearize(self.warping_fn, means_covs[0], means_covs[1]) # makes it all zeros
                means, covs = (
                    lift_and_diagonalize(means, covs, self.pos_basis_t)) # (means, covs) -> ((BS, N_samples, 21), (BS, N_samples, 21)) ???
                
                means_covs = (means, covs)"""

            if (not compute_normals) or (i_level < self.num_levels - 1):
                samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

                # Point attribute predictions
                if self.use_viewdirs:
                    viewdirs_enc = pos_enc(
                        rays.viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )
                    
                    if not self.prop_mlp or (i_level == (self.num_levels - 1)):
                        raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
                    else:
                        raw_density = self.prop_mlp(samples_enc, viewdirs_enc)
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)
            
            elif compute_normals and (i_level == (self.num_levels - 1)):
                raw_rgb, raw_density, normals = self.gradient(means_covs, rays.viewdirs)

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            if not self.prop_mlp or i_level >= self.num_levels - 1:
                rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
                rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            else:
                rgb = torch.zeros((*raw_density.squeeze().shape, 3), device=raw_density.device)
            density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
            
            comp_rgb, depth, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
                depth = rays.depth
            )

            """if compute_normals and i_level > 0:
                #normals = self.dir_enc_fun(normals, torch.ones(*normals.shape[0:2], 19, device='cuda:0'))
                normals = torch.cat([normals*weights.unsqueeze(-1), samples_enc, weights.unsqueeze(-1)], -1)
                normals = self.grad_decoder((normals))
                normals = l2_normalize((normals)) """
            
            ret[i_level] = comp_rgb

            #save the weights and intervals from the prop mlp
            if (i_level < self.num_levels - 1) and self.prop_mlp:
                w_env[i_level] = weights.clone()
                t_env[i_level] = t_samples.clone()
            

        if not compute_normals: normals = torch.zeros((batch_size, n_samples, 3), device=rays[0].device)
        if not self.depth_sampling: s = torch.as_tensor([eps] , device=rays[0].device)
        if self.prop_mlp: 
            envelope_loss = lossfun_outer(t_samples.detach(), weights.detach(), t_env[0], w_env[0])
            if self.num_levels > 2:
                envelope_loss = envelope_loss + lossfun_outer(t_samples.detach(), weights.detach(), t_env[1], w_env[1])
                envelope_loss = envelope_loss/(self.num_levels - 1)

        #if self.prop_mlp: envelope_loss = lossfun_outer(t_samples, weights, t_env, w_env)

        empty_loss, near_loss = lossfun_depth_weight_CDF(rays, t_samples, weights, eps)
        #empty_loss_uncertain, near_loss_uncertain =  lossfun_depth_weight_CDF(rays, t_samples, weights, eps, uncertain=True)
        #empty_loss_coarse, near_loss_coarse = lossfun_depth_weight_CDF(rays, t_env, w_env, eps)
        #empty_loss_coarse_uncertain, near_loss_coarse_uncertain =  lossfun_depth_weight_CDF(rays, t_samples, weights, eps, uncertain=True)
        empty_loss = empty_loss #+  empty_loss_coarse #+ empty_loss_coarse_uncertain
        near_loss = near_loss #+  near_loss_coarse #+ near_loss_coarse_uncertain
        
        #normal_lowerbound_error = normal_loss_lowerbound(rays, t_samples, weights, eps, normals)
        normal_lowerbound_error = torch.zeros_like(near_loss)
        
        
        #empty_loss, near_loss, normal_weights = lossfun_depth_weight(rays, t_samples, weights, eps)
        #empty_loss, near_loss, normal_weights = lossfun_depth_weight_alex(rays, t_samples, weights, eps, densities=density)
        out = {'rgb_coarse': ret[-2], 'rgb_fine': ret[-1], 'depth': depth, 'normal': normals, 'weights': weights, 's': s, 
                            'distortion': lossfun_distortion(t_samples, weights), 'near_loss': near_loss, 'empty_loss': empty_loss,
                            'normal_lowerbound_error': normal_lowerbound_error, 
                            'envelope_loss': envelope_loss
                            #'expected_depth': expected_depth, 'ray_termination':ray_termination
                            }
        return out

    def gradient(self, means_covs, viewdirs):
        means_covs[0].requires_grad_()
        means_covs[1].requires_grad_()
        GraphBools = self.training           
        with torch.enable_grad():
            view_direction_encoded = pos_enc(
                        viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )
            samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  
            raw_rgb, raw_density = self.mlp(samples_enc, view_direction_encoded)
            d_output = torch.ones_like(raw_density, requires_grad=GraphBools, device=raw_density.device)
            normals = torch.autograd.grad(
                outputs=raw_density,
                inputs=means_covs,
                grad_outputs=d_output,
                create_graph=GraphBools,
                retain_graph=GraphBools,
                only_inputs=True)[0]
            normals = -l2_normalize(torch.nan_to_num(normals)) 
        return raw_rgb, raw_density, normals
    
    def gradient_(self, means_covs, viewdirs):
        means_covs[0].requires_grad_()
        means_covs[1].requires_grad_()      
        #with torch.enable_grad():
        means = means_covs[0]
        covs = means_covs[1]
        means_flat = means.reshape((-1, means.shape[-1]))
        covs_flat = covs.reshape((-1,) + covs.shape[len(means.shape) - 1:])

        predict_density_and_grad_fn = functorch.vmap(
            functorch.grad_and_value(self.predict_density, has_aux=True), in_dims=(0, 0))
        raw_grad_density_flat, (raw_density_flat, x_flat) = (
            predict_density_and_grad_fn(means_flat, covs_flat))
            
        # Unflatten the output.
        raw_density = raw_density_flat.reshape(means.shape[:-1]).unsqueeze(-1)            
        x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
        normals = raw_grad_density_flat.reshape(means.shape)
        raw_rgb = self.make_color(x, viewdirs)
        
        normals = -l2_normalize(torch.nan_to_num(normals)) 

        return raw_rgb, raw_density, normals

    def predict_density(self, means, covs):
            means_covs_ = (means, covs)
            samples_enc = integrated_pos_enc(
                means_covs_,
                self.min_deg_point,
                self.max_deg_point,
            )                
            raw_density, x = self.mlp(samples_enc)
            return raw_density.squeeze(), x
        
    def make_color(self, x, view_direction):
        view_direction = pos_enc(
                        view_direction,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )            
        bottleneck = self.mlp.extra_layer(x)
        # Broadcast condition from [batch, feature] to
        # [batch, num_samples, feature] since all the samples along the same ray
        # have the same viewdir.
        # view_direction: [B, 2*3*L] -> [B, N, 2*3*L]
        num_samples = x.shape[1]
        view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
        x = torch.cat([bottleneck, view_direction], dim=-1)
        # Here use 1 extra layer to align with the original nerf model.
        x = self.mlp.view_layers(x)                
        raw_rgb = self.mlp.color_layer(x)
        return raw_rgb

    def chunked_inference(self, i_level, rays, means_covs, chunksize=2048, compute_normals: bool = False):
        B = means_covs[0].shape[0]
        out_chunks = []
        for i in range(0, B, chunksize):
            means_covs_ = means_covs[0][i:i+chunksize], means_covs[1][i:i+chunksize]
            if (not compute_normals) or (i_level !=1):
                samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

                # Point attribute predictions
                if self.use_viewdirs:
                    viewdirs_enc = pos_enc(
                        rays.viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )

                    raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
                    #out_chunks += [self.mlp(samples_enc, viewdirs_enc)]
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)
            
            elif compute_normals and i_level == 1:
                raw_rgb, raw_density, normals, ce = self.gradient(means_covs, rays.viewdirs)
                
            raw_rgb, raw_density = zip(*out_chunks)
            raw_rgb, raw_density = torch.cat(raw_rgb, 0), torch.cat(raw_density, 0)
        return raw_rgb, raw_density


class DepthVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(DepthVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(-self.variance.to(x.device))

def lossfun_distortion(t, w):   #t=z_vals, w=weights. Loss from mip-nerf 360
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, axis=-1), axis=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), axis=-1) / 3

    return loss_inter + loss_intra

def lossfun_depth_weight(rays: namedtuple, tvals_, w, eps):
    """Penalize sum of weights for empty interval
       Penalize squared distance from depth for near inteval"""
    depth = rays.depth    
    tvals = 0.5 * (tvals_[..., :-1] + tvals_[..., 1:]) #* torch.linalg.norm(torch.unsqueeze(rays.directions, dim=-2), dim=-1)
    # models/mip.py:8 here sample point by multiply the interval with the direction without normalized, so
    # the delta is norm(t1*d-t2*d) = (t1-t2)*norm(d)

    dummy_1 = torch.as_tensor([1.0], device = depth.device)
    
    depth_t = depth.broadcast_to(tvals.shape)
    sigma = (eps / 3.) ** 2
    mask_near = ((tvals > (depth - eps)) & (tvals < (depth + eps))).to(depth.dtype).reshape(tvals.shape[0], -1)
    mask_empty = (tvals < (depth - eps)).to(depth.dtype).reshape(tvals.shape[0], -1)
    dist = mask_near * (tvals - depth_t)
    dist = torch.nan_to_num(1.0 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-(dist ** 2 / (2 * sigma ** 2 + 1e-6))))
    dist = (dist/ dist.max()) * mask_near
    near_losses =  (((mask_near * w - dist) ** 2).sum() / torch.maximum(mask_near.sum(), dummy_1))
    empty_losses =  (((mask_empty * w) ** 2).sum() / torch.maximum(mask_empty.sum(), dummy_1))

    normal_weights = ((mask_near * w )) #make distribution over nonzero/nonmasked weights
    return empty_losses, near_losses, normal_weights

def lossfun_depth_weight_alex(rays: namedtuple, tvals_, w, eps, densities):
    """Penalize sum of weights for empty interval
       Penalize squared distance from depth for near inteval"""
    depth = rays.depth    
    tvals = 0.5 * (tvals_[..., :-1] + tvals_[..., 1:]) #* torch.linalg.norm(torch.unsqueeze(rays.directions, dim=-2), dim=-1)
    # models/mip.py:8 here sample point by multiply the interval with the direction without normalized, so
    # the delta is norm(t1*d-t2*d) = (t1-t2)*norm(d)

    dummy_1 = torch.as_tensor([1.0], device = depth.device)
    
    depth_t = depth.broadcast_to(tvals.shape)
    sigma = (eps / 3.) ** 2
    mask_near = ((tvals > (depth - eps)) & (tvals < (depth + eps))).to(depth.dtype).reshape(tvals.shape[0], -1)
    mask_empty = (tvals < (depth - eps)).to(depth.dtype).reshape(tvals.shape[0], -1)
    dist = mask_near * (tvals - depth_t)
    dist = torch.nan_to_num(1.0 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-(dist ** 2 / (2 * sigma ** 2 + 1e-6))))
    dist = (dist/ dist.max()) * mask_near #(BS, N_samples)
    #near_losses =  (((mask_near * w - dist) ** 2).sum() / torch.maximum(mask_near.sum(), dummy_1))
    #normalize sigma
    densities_normalized = nn.functional.softmax(densities.squeeze()) #BS, N_samples, 1
    near_losses = (densities_normalized - dist) /  torch.maximum(mask_near.sum(), dummy_1)
    near_losses = torch.maximum(mask_near*near_losses, torch.zeros_like(near_losses))
    near_losses = (near_losses**2)
    empty_losses =  (((mask_empty * w) ** 2).sum() / torch.maximum(mask_empty.sum(), dummy_1))

    normal_weights = ((mask_near * w )) #make distribution over nonzero/nonmasked weights
    return empty_losses, near_losses, normal_weights

def lossfun_depth_weight_CDF(rays: namedtuple, tvals_, w, eps, uncertain = False):
    """Penalize sum of weights for empty interval
       Penalize squared distance from depth for near inteval"""
    depth = rays.depth #.clone().detach()    

    """if not uncertain:
        mask = rays.depth_vars > 0
        depth[mask] = -5
        beta = 0.02
        #alpha = 0.0
    else:
        mask = rays.depth_vars > 0
        depth[~mask] = -5
        beta = rays.depth_vars
        #alpha = 0.1"""

    tvals = 0.5 * (tvals_[..., :-1] + tvals_[..., 1:])
    dummy_1 = torch.as_tensor([1.0], device = depth.device)
    sigma = (eps / 1.)
    
    #beta = 0.01 * torch.ones_like(rays.depth_vars)
    #beta[rays.depth_vars > 0.01] = rays.depth_vars[rays.depth_vars > 0.01]
    #beta = rays.depth_vars
    #beta = 0.02
    beta = 0.

    mask_near_close = (((tvals) > (depth  - 3*sigma - beta)) & (tvals  < (depth - beta)) & (depth > 0)).to(depth.dtype).reshape(tvals.shape[0], -1)
    mask_near_far = ((tvals  > (depth + beta)) & ((tvals) < (depth  + 3*sigma + beta)) & (depth > 0)).to(depth.dtype).reshape(tvals.shape[0], -1)
    mask_empty = (((tvals) < (depth - 3*sigma - beta)) & (depth > 0)).to(depth.dtype).reshape(tvals.shape[0], -1)
    alpha = 0.
    
    #w shape: BS, N_samples
    m_1 = torch.distributions.normal.Normal(loc = depth - beta, scale = sigma) 
    m_2 = torch.distributions.normal.Normal(loc = depth + beta, scale = sigma) 

    opacity = 1.0 #w.sum(dim=-1).unsqueeze(1)
    cdf_values_1 = opacity * m_1.cdf(tvals) # BS, N_samples
    cdf_values_2 = opacity * m_2.cdf(tvals) # BS, N_samples
    w_cumsum = torch.cumsum(w, dim=-1) # BS, N_samples
    #alpha, tolerance parameter, relates to depth uncertainty
    
    #Before Depth we are upper bounded by the CDF
    near_close_losses = (w_cumsum - cdf_values_1 * opacity - alpha)
    near_close_losses = torch.maximum(mask_near_close*near_close_losses, torch.zeros_like(near_close_losses))
    near_close_losses = (near_close_losses**2).sum()  / torch.maximum(mask_near_close.sum(), dummy_1)

    #After Depth we are lower bounded by the CDF
    near_far_losses = (cdf_values_2 * opacity - w_cumsum - alpha) 
    near_far_losses = torch.maximum(mask_near_far*near_far_losses, torch.zeros_like(near_far_losses))
    near_far_losses = (near_far_losses**2).sum() / torch.maximum(mask_near_far.sum(), dummy_1)

    near_losses = (near_close_losses + near_far_losses)
    empty_losses =  (((mask_empty * w)**2).sum() / torch.maximum(mask_empty.sum(), dummy_1)) 
    return empty_losses, near_losses

def normal_loss_lowerbound(rays: namedtuple, tvals_, w, eps, n_pred):
    depth = rays.depth  
    tvals = 0.5 * (tvals_[..., :-1] + tvals_[..., 1:])
    depth_t = depth.broadcast_to(tvals.shape)
    dummy_1 = torch.as_tensor([1.0], device = depth.device)
    epsilon = 1e-3
    sigma = 0.1 #hyperparam, set in config
    #mask_near = ((tvals > (depth - 2*sigma)) & (tvals < (depth + 2*sigma))).to(depth.dtype).reshape(tvals.shape[0], -1) #BS, N_samples
    mask_near = ((tvals > (depth - 2*sigma)) & (tvals < (depth + epsilon))).to(depth.dtype).reshape(tvals.shape[0], -1) #BS, N_samples FRONT HALF
    g = mask_near * (tvals - depth_t) #BS, N_samples
    #dont need to normalize since we dont care about it being a distribution, its just a bell-shaped function as lowerbound.
    g = torch.nan_to_num(torch.exp(-(g ** 2 / (2 * (6*sigma) ** 2))))
    mask_near = (torch.cumsum(w, -1) < 0.75).to(depth.dtype)


    #normalize the function so that the peak rises to 1 by the end.
    eps_min = 0.15

    normalization = min(eps_min / eps, 1.0)
    g = g * normalization #(BS, N_samples)
    #g = normalization

    n_gt = rays.normal #BS, 3
    mask = (torch.any(rays.normal, dim=-1))
    # MAE of Normal alignment    
    loss = n_gt[:,None,:] * n_pred 
    loss = torch.sum(loss, -1) * mask[..., None]
    loss =  w * torch.maximum((g - loss), torch.zeros_like(loss))
    loss = mask_near * loss  # --> (BS, Num_samples)
    loss = (loss).sum(dim=-1) # --> (BS)
    loss = torch.maximum(torch.zeros_like(loss), loss) ** 2
    loss = loss / torch.maximum(mask_near.sum(), dummy_1)
    return loss


def lossfun_outer(t, w, t_env, w_env, eps=torch.finfo(torch.float16).eps):
  """The proposal weight should be an upper envelope on the nerf weight."""
  _, w_outer = inner_outer(t, t_env, w_env) #t.shape: 129, t_env.shape: 129, w_env.shape: 128
  # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
  # more effective to pull w_outer up than it is to push w_inner down.
  # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
  return torch.maximum(torch.zeros_like(w), w - w_outer)**2 / (w + eps) #w.shape : [BS, 128] , w_outer.shape : [BS, 128]

def inner_outer(t0, t1, y1):
  """Construct inner and outer measures on (t1, y1) for t0."""
  cy1 = torch.concatenate([torch.zeros_like(y1[..., :1]),
                         torch.cumsum(y1, axis=-1)],
                        axis=-1)
  idx_lo, idx_hi = searchsorted(t1, t0)

  cy1_lo = torch.gather(cy1, axis=-1, index=idx_lo)
  cy1_hi = torch.gather(cy1, axis=-1, index=idx_hi)

  y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]
  y0_inner = torch.where(idx_hi[..., :-1] <= idx_lo[..., 1:],
                       cy1_lo[..., 1:] - cy1_hi[..., :-1], 0)
  return y0_inner, y0_outer

def searchsorted(a, v):
  """Find indices where v should be inserted into a to maintain order.
  This behaves like jnp.searchsorted (its second output is the same as
  jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
  faster because it wastes memory to save some compute.
  Args:
    a: tensor, the sorted reference points that we are scanning to see where v
      should lie.
    v: tensor, the query points that we are pretending to insert into a. Does
      not need to be sorted. All but the last dimensions should match or expand
      to those of a, the last dimension can differ.
  Returns:
    (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
    range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
    last index of a.
  """
  i = torch.arange(a.shape[-1], device=a.device)
  v_ge_a = v[..., None, :] >= a[..., :, None]
  idx_lo, _ = torch.max(torch.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2)
  idx_hi, _ = torch.min(torch.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2)
  return idx_lo, idx_hi


if __name__ == '__main__':
    import collections
    Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'depth', 'normal', 'mask'))
    # 
    """ 'Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'depth', 'normal', 'mask'))"""
    torch.manual_seed(0)
    batch_size = 4096
    origins = torch.rand([batch_size, 3]).to('cuda')
    directions = torch.rand(batch_size, 3).to('cuda')
    radii = torch.rand([batch_size, 1]).to('cuda')
    num_samples = 64
    near = torch.rand([batch_size, 1]).to('cuda')
    far = torch.rand([batch_size, 1]).to('cuda')
    viewdir = torch.rand([batch_size, 3]).to('cuda')
    normal = torch.rand([batch_size, 3]).to('cuda')
    lossmult = torch.tensor([0.]).to('cuda')
    depth = torch.rand([batch_size, 1]).to('cuda')
    mask = (depth > 0.5).to(torch.float).to('cuda')
    randomized = True
    disparity = False
    ray_shape = 'cone'
    rays = Rays(origins, directions, origins, radii, lossmult, near, far, depth, normal, mask)

    means = torch.rand(4096, 32, 3, requires_grad=True).to('cuda')
    covs = torch.rand(4096, 32, 3, requires_grad=True).to('cuda')
    min_degree = torch.as_tensor([0.], device='cuda')

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            #integrated_pos_enc((means, covs), 0, 16, True)
            sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape, rays, None)
            
            pass
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
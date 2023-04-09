import torch
import numpy as np
from einops import repeat
from models.mip import sample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering, resample_along_rays, sample, resample
from collections import namedtuple


def l2_normalize(x, eps=torch.tensor(torch.finfo(torch.float32).eps)):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, dim=-1, keepdims=True), eps))

def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int):
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
        num_samples = x.shape[1]
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
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density


class MipNerf(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(self, num_samples: int = 128,
                 num_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_resample_grad: bool = True,
                 use_viewdirs: bool = True,
                 disparity: bool = False,
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
                 mlp_net_activation: str = 'relu'):
        super(MipNerf, self).__init__()
        self.num_levels = num_levels  # The number of sampling levels.
        self.num_samples = num_samples  # The number of samples per level.
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
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
                       mlp_xyz_dim, mlp_view_dim)
        if rgb_activation == 'sigmoid':  # The RGB activation.
            self.rgb_activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.rgb_padding = rgb_padding
        if density_activation == 'softplus':  # Density activation.
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError

    def forward(self, rays: namedtuple, randomized: bool, white_bkgd: bool):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """

        ret = []
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
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
                )
            else:
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
                )
            if self.disable_integration:
                means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
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
            else:
                raw_rgb, raw_density = self.mlp(samples_enc)

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )
            ret.append((comp_rgb, distance, acc, weights, t_samples))

        return ret



    """
    A simple MLP for NeuS.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int):
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
        super(NeusMLP, self).__init__()
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
        num_samples = x.shape[1]
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
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density
  
    
class NeusMLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 multires_xyz: int, multires_view: int):
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
        super(NeusMLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        self.enc_xyz, _ = get_embedder(multires_xyz, 3)
        self.enc_view, _ = get_embedder(multires_view, 6)
        layers = []
        xyz_dim = 3 * multires_xyz * 2 + 3
        view_dim = 3 * multires_view * 2 + 3
        normal_dim = 3 * multires_view * 2 + 3
        # The first part of MLP.
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
        self.pdf_layer = torch.nn.Linear(net_width, 1)
        _xavier_init(self.pdf_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)  # extra_layer is not the same as NeRF
        _xavier_init(self.extra_layer)
        self.grad_layer = torch.nn.Linear(net_width, 3)
        _xavier_init(self.grad_layer)
        layers = []
        # The second part of MLP.
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim + normal_dim
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
        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

    def forward(self, x, view_direction):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, num_samples, 3], points.
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
        if self.training:
            x.requires_grad = True
        num_samples = x.shape[1]
        # encode the position
        x_enc = self.enc_xyz(x)
        inputs = x_enc  # [B, N, 3]
        for i, layer in enumerate(self.layers):
            x_enc = layer(x_enc)
            if i % self.skip_index == 0 and i > 0:
                x_enc = torch.cat([x_enc, inputs], dim=-1)
        
        normal_pred = self.grad_layer(x_enc)
        sdf = self.pdf_layer(x_enc) # [B, N, 1]
        # compute the normal
        # https://github.com/gkouros/refnerf-pytorch/blob/main/internal/models.py line 562
        normal = None
        if self.training:
            normal = torch.autograd.grad(
                sdf,
                x,
                torch.ones_like(sdf),
                retain_graph=True
            )[0]
        else:
            normal = normal_pred
        
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x_enc)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            # view_direction: [B, 2*3*L] -> [B, N, 2*3*L]
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            view = torch.cat([view_direction, normal], dim=-1)
            view = self.enc_view(view)
            x = torch.cat([bottleneck, view], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            for _, layer in enumerate(self.view_layers):
                x = layer(x)
            raw_rgb = self.color_layer(x)
        return raw_rgb, sdf, normal, normal_pred
    

class NeuS(torch.nn.Module):
    def __init__(self, num_samples: int = 128,
                 num_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_resample_grad: bool = True,
                 use_viewdirs: bool = True,
                 disparity: bool = False,
                 ray_shape: str = 'cone',
                 multires_xyz: int = 16,
                 multires_view: int = 4,
                 density_activation: str = 'softplus',
                 density_noise: float = 0.,
                 density_bias: float = -1.,
                 rgb_activation: str = 'sigmoid',
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,
                 mlp_net_depth: int = 8,
                 mlp_net_width: int = 256,
                 mlp_net_depth_condition: int = 4,
                 mlp_net_width_condition: int = 256,
                 mlp_skip_index: int = 4,
                 mlp_num_rgb_channels: int = 3,
                 mlp_num_density_channels: int = 1,
                 mlp_net_activation: str = 'relu'):
        super(NeuS, self).__init__()
        self.num_levels = num_levels  # The number of sampling levels.
        self.num_samples = num_samples  # The number of samples per level.
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
        self.ray_shape = ray_shape  # The shape of cast rays ('cone' or 'cylinder').
        self.disable_integration = disable_integration  # If True, use PE instead of IPE.
        self.multires_xyz = multires_xyz  # Degree of positional encoding for pts.
        self.multires_view = multires_view  # Degree of positional encoding for viewdirs.
        self.use_viewdirs = use_viewdirs  # If True, use view directions as a condition.
        self.density_noise = density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding  # Dirichlet/alpha "padding" on the histogram.
        self.stop_resample_grad = stop_resample_grad  # If True, don't backprop across levels')
        self.mlp = NeusMLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       multires_xyz, multires_view)
        if rgb_activation == 'sigmoid':  # The RGB activation.
            self.rgb_activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.rgb_padding = rgb_padding
        if density_activation == 'softplus':  # Density activation.
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError 
        
    def forward(self, rays: namedtuple, randomized: bool, white_bkgd: bool):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """

        ret = []
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_samples, pts = sample(
                    rays.origins,
                    rays.directions,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                )
            else:
                t_samples, pts = resample(
                    rays.origins,
                    rays.directions,
                    t_samples,
                    weights,
                    randomized,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )

            raw_rgb, raw_density, normal, normal_pred = self.mlp(pts, rays.viewdirs)

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

            density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )
            ret.append((comp_rgb, distance, acc, weights, t_samples, normal, normal_pred))

        return ret
    

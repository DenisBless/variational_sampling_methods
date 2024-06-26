import jax.numpy as jnp
from flax import linen as nn


class TimeEncoder(nn.Module):
    num_hid: int = 2

    def setup(self):
        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.mlp = [
            nn.Dense(2 * self.num_hid),
            nn.gelu,
            nn.Dense(self.num_hid),
        ]

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, time_array_emb):
        time_array_emb = self.get_fourier_features(time_array_emb)
        for layer in self.mlp:
            time_array_emb = layer(time_array_emb)
        return time_array_emb


class StateTimeEncoder(nn.Module):
    num_layers: int = 2
    num_hid: int = 64
    zero_init: bool = False

    def setup(self):
        if self.zero_init:
            last_layer = [
                nn.Dense(self.parent.dim, kernel_init=nn.initializers.zeros_init(),
                         bias_init=nn.initializers.zeros_init())]
        else:
            # last_layer = [nn.Dense(self.parent.dim)]
            last_layer = [
                nn.Dense(self.parent.dim, kernel_init=nn.initializers.normal(stddev=1e-7),
                         bias_init=nn.initializers.zeros_init())]

        self.state_time_net = [
                                  nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) for _ in
                                  range(self.num_layers)
                              ] + last_layer

    def __call__(self, extended_input):
        for layer in self.state_time_net:
            extended_input = layer(extended_input)
        return extended_input


class LangevinScaleNet(nn.Module):
    num_layers: int = 2
    num_hid: int = 64
    lgv_per_dim: bool = False

    def setup(self):
        self.time_coder_grad = [
                                   nn.Dense(self.num_hid)
                               ] + [
                                   nn.Sequential([nn.gelu, nn.Dense(self.num_hid)]) for _ in
                                   range(self.num_layers)
                               ] + [
                                   nn.gelu,
                                   nn.Dense(self.parent.dim if self.lgv_per_dim else 1,
                                            kernel_init=nn.initializers.zeros_init(),
                                            bias_init=nn.initializers.zeros_init())
                               ]

    def __call__(self, time_array_emb):
        for layer in self.time_coder_grad:
            time_array_emb = layer(time_array_emb)
        return time_array_emb


class PISGRADNet(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.

    def setup(self):
        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential([
            nn.Dense(self.num_hid),
            nn.gelu,
            nn.Dense(self.num_hid),
        ])

        self.time_coder_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
            [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                          bias_init=nn.initializers.constant(self.bias_init))])

        self.state_time_net = nn.Sequential([nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
                                                nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
                                                         bias_init=nn.initializers.zeros_init())])

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, time_array, lgv_term):
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)
        t_net2 = self.time_coder_grad(time_array_emb)

        extended_input = jnp.concatenate((input_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        out_state_p_grad = out_state + t_net2 * lgv_term
        return out_state_p_grad

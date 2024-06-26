import jax.numpy as jnp
from flax import linen as nn

from algorithms.common.models.langevin_net import TimeEncoder, StateTimeEncoder


class StateTimeNetwork(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    out_clip: float = 1e4

    def setup(self):
        self.state_time_coder = TimeEncoder(self.num_hid)
        self.state_time_net = StateTimeEncoder(num_layers=self.num_layers, num_hid=self.num_hid,
                                               name='state_time_net', parent=self)

    def __call__(self, input_array, time_array, lgv_term):
        t_embed = self.state_time_coder(time_array)
        if len(input_array.shape) == 1:
            t_embed = t_embed[0]

        extended_input = jnp.concatenate((input_array, t_embed), axis=-1)
        out_state = self.state_time_net(extended_input)
        return jnp.clip(out_state, -self.out_clip, self.out_clip)

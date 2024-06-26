from flax import linen as nn
import jax
import jax.numpy as jnp
import distrax


class NICE(nn.Module):
    """Implements a NICE flow."""
    dim: int
    h_dim: int
    n_steps: int = 4
    h_depth: int = 5

    def setup(self):
        self._halfdim = self.dim // 2
        nets = []
        for _ in range(self.n_steps):
            layers = []
            for _ in range(self.h_depth):
                layers.append(nn.Dense(self.h_dim))
                layers.append(nn.relu)
            layers.append(nn.Dense(self._halfdim))
            net = nn.Sequential(layers)
            nets.append(net)
        self.nets = nn.Sequential(nets)
        parts = []
        for _ in range(self.n_steps):
            shuff = list(reversed(range(self.dim)))
            parts.append(shuff)
        self.parts = nn.Sequential(parts)

        self._logscale = self.param('logscale', lambda rng, shape: jnp.zeros(shape))

    def __call__(self):
        return self.logpx, lambda x: self.reverse(self.forward(x)), self.sample

    def forward(self, x):
        split = self._halfdim
        if self.dim % 2 == 1:
            split += 1

        for part, net in zip(self._parts, self._nets):
            x_shuff = x[:, part]
            xa, xb = x_shuff[:, :split], x_shuff[:, split:]
            ya = xa
            yb = xb + net(xa, rng=None)  # Flax does not require explicit rng argument
            x = jnp.concatenate([ya, yb], -1)

        return x

    def reverse(self, y):
        split = self._halfdim
        if self.dim % 2 == 1:
            split += 1

        for inv_part, net in reversed(list(zip(self._parts, self._nets))):
            ya, yb = y[:, :split], y[:, split:]
            xa = ya
            xb = yb - net(xa, rng=None)  # Flax does not require explicit rng argument
            x_shuff = jnp.concatenate([xa, xb], -1)
            y = x_shuff[:, inv_part]

        return y

    def logpx(self, x):
        z = self.forward(x)
        zs = z * jnp.exp(self._logscale)[None, :]

        pz = distrax.MultivariateNormalDiag(jnp.zeros_like(zs), jnp.ones_like(zs))
        logp = pz.log_prob(zs)
        logp = logp + self._logscale.sum()

        return logp

    def sample(self, n):
        zs = jax.random.normal(jax.random.PRNGKey(0), (n, self.dim))
        z = zs / jnp.exp(self._logscale)[None, :]
        x = self.reverse(z)

        return x

    def reparameterized_sample(self, zs):
        z = zs / jnp.exp(self._logscale)[None, :]
        x = self.reverse(z)

        return x

    def loss(self, x):
        return -self.logpx(x)

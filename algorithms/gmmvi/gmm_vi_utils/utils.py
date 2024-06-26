import jax.numpy as jnp
import jax

def reduce_weighted_logsumexp(logx, w=None, axis=None, keep_dims=False, return_sign=False,):
    if w is None:
      lswe = jax.nn.logsumexp(
          logx,
          axis=axis,
          keepdims=keep_dims)

      if return_sign:
        sgn = jnp.ones_like(lswe)
        return lswe, sgn
      return lswe

    log_absw_x = logx + jnp.log(jnp.abs(w))
    max_log_absw_x = jnp.max(
        log_absw_x, axis=axis, keepdims=True,)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = jnp.where(
        jnp.isinf(max_log_absw_x),
        jnp.zeros([], max_log_absw_x.dtype),
        max_log_absw_x)
    wx_over_max_absw_x = (jnp.sign(w) * jnp.exp(log_absw_x - max_log_absw_x))
    sum_wx_over_max_absw_x = jnp.sum(
        wx_over_max_absw_x, axis=axis, keepdims=keep_dims)
    if not keep_dims:
      max_log_absw_x = jnp.squeeze(max_log_absw_x, axis)
    sgn = jnp.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + jnp.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
      return lswe, sgn
    return lswe
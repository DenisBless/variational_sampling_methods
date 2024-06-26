from typing import NamedTuple, Callable, Optional
import chex
import jax.numpy as jnp


class QuadRegressionState(NamedTuple):
    bias_entry: int
    params: Optional[int]
    num_quad_features: int
    num_features: int
    triu_idx_const: chex.Array

# Todo call method
class QuadRegression(NamedTuple):
    init_quad_reg_state: Callable
    fit_quadratic: Callable


def setup_quad_regression(DIM):
    def init_quad_reg_state():
        return QuadRegressionState(bias_entry=-1,
                                   params=None,
                                   num_quad_features=int(jnp.floor(0.5 * (DIM + 1) * DIM)),
                                   num_features=int(jnp.floor(0.5 * (DIM + 1) * DIM)) + DIM + 1,
                                   triu_idx_const=jnp.array(
                                       jnp.transpose(jnp.stack(jnp.where(jnp.triu(jnp.ones([DIM, DIM], jnp.bool_))))))
                                   )

    def fit_quadratic(quad_reg_state: QuadRegressionState, regularizer: float, num_samples: int, inputs: chex.Array,
                      outputs: chex.Array,
                      weights: chex.Array = None, sample_mean: chex.Array = None, sample_chol_cov: chex.Array = None) \
            -> [chex.Array, chex.Array, chex.Array]:
        def _fit(quad_reg_state: QuadRegressionState, regularizer: float, num_samples: int, inputs: chex.Array,
                 outputs: chex.Array, weights: chex.Array = None) -> chex.Array:
            def _feature_fn(quad_reg_state, num_samples: int, x: chex.Array) -> chex.Array:
                linear_features = x
                constant_feature = jnp.ones((len(x), 1))

                # quad features
                quad_features = jnp.zeros((num_samples, 0))
                for i in range(quad_reg_state.dim):
                    quad_features = jnp.concatenate((quad_features, jnp.expand_dims(x[:, i], axis=1) * x[:, i:]),
                                                    axis=1)

                # stack quadratic features, linear features and constant features
                features = jnp.concatenate((quad_features, linear_features, constant_feature), axis=1)
                return features

            if len(jnp.shape(outputs)) > 1:
                outputs = jnp.squeeze(outputs)
            features = _feature_fn(quad_reg_state, num_samples, x=inputs)

            if weights is not None:
                if len(weights.shape) == 1:
                    weights = jnp.expand_dims(weights, 1)
                weighted_features = jnp.transpose(weights * features)
            else:
                weighted_features = jnp.transpose(features)
            # regression
            reg_mat = jnp.eye(quad_reg_state.num_features) * regularizer
            #
            if quad_reg_state.bias_entry is not None:
                bias_index = jnp.arange(len(reg_mat))[quad_reg_state.bias_entry]
                reg_mat[[bias_index, bias_index]] = [0]
            params = jnp.squeeze(jnp.linalg.solve(weighted_features @ features + reg_mat,
                                                  weighted_features @ jnp.expand_dims(outputs, 1)))
            return params

        whitening = True
        if sample_mean is None:
            assert sample_chol_cov is None
        if sample_chol_cov is None:
            assert sample_mean is None

        # whithening
        if whitening and sample_mean is not None and sample_chol_cov is not None:
            inv_samples_chol_cov = jnp.linalg.inv(sample_chol_cov)
            inputs = (inputs - sample_mean) @ jnp.transpose(inv_samples_chol_cov)

        params = _fit(quad_reg_state, regularizer, num_samples, inputs, outputs, weights)

        qt = jnp.zeros((DIM, DIM))
        qt[quad_reg_state.triu_idx_const[:, 0], quad_reg_state.triu_idx_const[:, 1]] = params[- (DIM + 1)]
        # qt = tf.scatter_nd(quad_reg_state.triu_idx_const, params[:- (quad_reg_state.dim + 1)], [quad_reg_state.dim, quad_reg_state.dim])

        quad_term = - qt - jnp.transpose(qt)
        lin_term = params[-(DIM+ 1):-1]
        const_term = params[-1]

        # unwhitening:
        if whitening and sample_mean is not None and sample_chol_cov is not None:
            quad_term = jnp.transpose(inv_samples_chol_cov) @ quad_term @ inv_samples_chol_cov
            t1 = jnp.dot(jnp.transpose(inv_samples_chol_cov), lin_term)
            t2 = jnp.dot(quad_term, sample_mean)
            lin_term = t1 + t2
            const_term += jnp.sum(sample_mean * (-0.5 * t2 - t1))

        return quad_term, lin_term, const_term

    return QuadRegression(init_quad_reg_state=init_quad_reg_state,
                          fit_quadratic=fit_quadratic)

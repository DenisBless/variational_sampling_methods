"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Dict, NamedTuple, Optional, Callable, Tuple, Protocol, Union

import chex
import jax


LogProbFn = Callable[[chex.Array], chex.Array]


class Point(NamedTuple):
    """State of the MCMC chain, specifically designed for FAB."""
    x: chex.Array
    log_q: chex.Array
    log_p: chex.Array
    grad_log_q: Optional[chex.Array] = None
    grad_log_p: Optional[chex.Array] = None


TransitionOperatorState = chex.ArrayTree

class TransitionOperatorStep(Protocol):
    def __call__(self,
             point: Point,
             transition_operator_state: TransitionOperatorState,
             beta: chex.Array,
             alpha: float,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn) -> Tuple[Point, TransitionOperatorState, Dict]:
        """Perform MCMC step with the intermediate target given by:
            \log target = ((1-beta) + beta*(1-alpha)) * log_q + beta*alpha*log_p
        """


class TransitionOperator(NamedTuple):
    init: Callable[[chex.PRNGKey], chex.ArrayTree]
    step: TransitionOperatorStep
    # Whether the transition operator uses gradients (True for HMC, False for metropolis).
    uses_grad: bool = True


class AISForwardFn(Protocol):
    def __call__(self, sample_q_fn: Callable[[chex.PRNGKey], chex.Array],
                 log_q_fn: LogProbFn, log_p_fn: LogProbFn,
                 ais_state: chex.Array) -> [chex.Array, chex.Array, chex.ArrayTree, Dict]:
        """

        Args:
            sample_q_fn: Sample from base distribution.
            log_q_fn: Base log density.
            log_p_fn: Target log density (note not the same as the AIS target which is p^2/q)
            ais_state: AIS state.

        Returns:
            x: Samples from AIS.
            log_w: Unnormalized log weights from AIS.
            ais_state: Updated AIS state.
            info: Dict with additional information.
        """


def create_point(x: chex.Array, log_q_fn: LogProbFn, log_p_fn: LogProbFn,
                 with_grad: bool = True) -> Point:
    """Create an instance of a `Point` which contains the necessary info on a point for MCMC."""
    chex.assert_rank(x, 1)
    if with_grad:
        log_q, grad_log_q = jax.value_and_grad(log_q_fn)(x)
        log_p, grad_log_p = jax.value_and_grad(log_p_fn)(x)
        return Point(x=x, log_p=log_p, log_q=log_q, grad_log_p=grad_log_p, grad_log_q=grad_log_q)
    else:
        return Point(x=x, log_q=log_q_fn(x), log_p=log_p_fn(x))



def get_intermediate_log_prob(
        log_q: chex.Array,
        log_p: chex.Array,
        beta: chex.Array,
        alpha: Union[chex.Array, float],
        ) -> chex.Array:
    """Get log prob of point according to intermediate AIS distribution.
    Set AIS final target g=p^\alpha q^(1-\alpha).
    log_prob = (1 - beta) log_q + beta log_g.
    """
    return ((1-beta) + beta*(1-alpha)) * log_q + beta*alpha*log_p



def get_grad_intermediate_log_prob(
        grad_log_q: chex.Array,
        grad_log_p: chex.Array,
        beta: chex.Array,
        alpha: Union[chex.Array, float],
) -> chex.Array:
    """Get gradient of intermediate AIS distribution for a point.
    Set AIS final target g=p^\alpha q^(1-\alpha). log_prob = (1 - beta) log_q + beta log_g.
    """
    return ((1-beta) + beta*(1-alpha)) * grad_log_q + beta*alpha*grad_log_p


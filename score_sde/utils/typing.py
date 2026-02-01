from typing import Callable, Tuple

import jax
import jax.numpy as jnp

ParametrisedScoreFunction = Callable[[dict, dict, jnp.ndarray, float], jnp.ndarray]
ScoreFunction = Callable[[jnp.ndarray, float], jnp.ndarray]

SDEUpdateFunction = Callable[
    [jax.Array, jnp.ndarray, float],
    Tuple[
        jnp.ndarray,
        jnp.ndarray,
    ],
]

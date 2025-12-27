from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class HMMResult:
    model: Any
    state_seq: np.ndarray   # (T,)
    state_prob: np.ndarray  # (T, K)
    logprob: float

def fit_hmm_gaussian(
    X: np.ndarray,
    n_states: int = 3,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 0,
) -> Any:
    """Fit a Gaussian HMM using hmmlearn.

    X must be (T, F). Use z-scored / scaled features.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:
        raise ImportError(
            "hmmlearn is required for HMM. Install: pip install hmmlearn\n" + str(e)
        )
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    model = GaussianHMM(
        n_components=int(n_states),
        covariance_type=str(covariance_type),
        n_iter=int(n_iter),
        random_state=int(random_state),
        verbose=False,
    )
    model.fit(X)
    return model

def decode_hmm(model: Any, X: np.ndarray) -> HMMResult:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    logprob, state_seq = model.decode(X, algorithm="viterbi")
    state_prob = model.predict_proba(X)
    return HMMResult(model=model, state_seq=state_seq.astype(np.int64), state_prob=state_prob.astype(np.float32), logprob=float(logprob))

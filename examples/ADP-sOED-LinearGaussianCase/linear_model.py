import numpy as np

def linear_model(stage, theta, d, xp=None):
    """
    Linear model function G(theta, d) = theta * d

    Parameters
    ----------
    stage : int
        The stage index of the experiment.
    theta : array_like of size (n_sample, n_param)
        The value of unknown linear model parameters.
    d : array_like of size (n_sample, n_design)
        The design variable.
    xp : array_like of size (s_sample, n_phys_state), optional(default=None)
        The physical state.

    Returns
    -------
    array_like of size (n_sample, n_obs)
        The output of the linear model.
    """
    global count
    count += max(len(theta), len(d))
    return theta * d
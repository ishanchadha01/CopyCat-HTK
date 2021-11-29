
import os
from scipy.stats import norm
import numpy as np


def feature_max_ll(features: 'np.ndarray', components: list, feature_id: int) -> float:
    '''Returns the max log likelihood of a feature

    Parameters
    ----------
    features : np.ndarray
        Numpy array of feature vectors for a given word and a state.

    components : list
        List containing mean and variance of each Gaussian component of a feature.

    feature_id : int
        ID of feature corresponding to which selected feature is being referred to.
    '''

    # Gets feature value
    feature_vector = features[feature_id]
    max_ll = float('-inf')

    # Iterate over mean and variance lists for each Gaussian component obtained from newMacros
    for mu, var in components:
        mu  = mu[feature_id]
        var = var[feature_id]
        ll  = norm.logpdf(feature_vector, mu, var)
        if ll > max_ll:
            max_ll = ll
    return max_ll



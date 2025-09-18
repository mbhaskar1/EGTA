import math
from typing import List

import numpy as np
from scipy.special import comb

TALAGRAND_MC_ERA_N = 200


def hoeffding_bound(c: float, delta: float, stats: dict, cardinality_multiplier: int = 1):
    m = stats['m']
    cardinality = stats['cardinality']
    log_term = np.log(2) + np.log(cardinality) + np.log(cardinality_multiplier) - np.log(delta)  # equal to ln(2|I|/d)
    return c * np.sqrt(log_term / (2 * m))


def rademacher_bound(c: float, delta: float, stats: dict):
    m = stats['m']
    one_era = stats['one_era']
    log_term = -np.log(delta)
    return 2 * one_era + 3 * c * np.sqrt(log_term / (2 * m))


def rademacher_bound_2(c: float, delta: float, stats: dict):
    m = stats['m']
    log_term = np.log(2) - np.log(delta)
    n = TALAGRAND_MC_ERA_N
    era_bound = stats['mc_era'] + 2 * stats['maximum'] * log_term / (3 * n * m) + \
                c * np.sqrt(log_term / (n * m))
    ra_bound = era_bound + c * log_term / (3*m) + np.sqrt((c * log_term / (2 * np.sqrt(3) * m)) ** 2 +
                                                          c * era_bound * log_term / m)
    return 2 * ra_bound + c * np.sqrt(log_term/(2*m))


def rademacher_bound_3(c: float, delta: float, stats: dict):
    m = stats['m']
    log_term = np.log(2) - np.log(delta)
    n = TALAGRAND_MC_ERA_N
    era_bound = stats['centralized_mc_era'] + 2 * stats['centralized_maximum'] * log_term / (3 * n * m) + \
                c * np.sqrt(log_term / (n * m))
    b_var = b(m)
    chi = 1 + 2 * b_var
    r = c
    ra_bound = era_bound + 2 * r * chi * log_term / (3 * m) + \
                   np.sqrt((r * chi * log_term / (np.sqrt(3) * m)) ** 2 +
                           2 * r * chi * (era_bound + r * b_var) * log_term / m)
    return 2 * ra_bound + c * np.sqrt(log_term/(2*m))


def rademacher_bound_4(c: float, delta: float, stats: dict):
    m = stats['m']
    log_term = np.log(2) - np.log(delta)
    n = TALAGRAND_MC_ERA_N
    era_bound = stats['mc_era'] + 2 * stats['maximum'] * np.sqrt(log_term/(2 * n * m))
    ra_bound = era_bound + c * log_term / (3*m) + np.sqrt((c * log_term / (2 * np.sqrt(3) * m)) ** 2 +
                                                          c * era_bound * log_term / m)
    return 2 * ra_bound + c * np.sqrt(log_term/(2*m))


def rademacher_bound_5(c: float, delta: float, stats: dict):
    m = stats['m']
    log_term = np.log(2) - np.log(delta)
    n = TALAGRAND_MC_ERA_N
    era_bound = stats['centralized_mc_era'] + 2 * stats['centralized_maximum'] * np.sqrt(log_term/(2*n*m))
    b_var = b(m)
    chi = 1 + 2 * b_var
    r = c
    ra_bound = era_bound + 2 * r * chi * log_term / (3 * m) + \
                   np.sqrt((r * chi * log_term / (np.sqrt(3) * m)) ** 2 +
                           2 * r * chi * (era_bound + r * b_var) * log_term / m)
    return 2 * ra_bound + c * np.sqrt(log_term/(2*m))


def bennet_non_uniform(c: float, delta: float, stats: dict, cardinality_multiplier: int = 1):
    cardinality = stats['cardinality']
    m = stats['m']
    log_term = np.log(3) + np.log(cardinality) + np.log(cardinality_multiplier) - np.log(delta)
    sample_variance = stats['sample_variance']
    sample_variance += 2/3 * c * c * log_term / m + np.sqrt(
        (1/3 + 1/(2 * log_term)) * (c ** 2 * log_term / (m-1)) ** 2
        + 2 * (c ** 2) * sample_variance * log_term / m
    )
    epsilon_mu = c * log_term / (3 * m) + np.sqrt(2 * sample_variance * log_term / m)
    return epsilon_mu


def bennet_union_bound(c: float, delta: float, stats: dict, cardinality_multiplier: int = 1,
                       return_matrix: bool = False):
    cardinality = stats['cardinality']
    m = stats['m']
    log_term = np.log(3) + np.log(cardinality) + np.log(cardinality_multiplier) - np.log(delta)  # equal to ln(3|I|/d)
    unbiased_e_wimpy = stats['empirical_wimpy'] * m / (m-1)
    epsilon_v = 2/3 * c * c * log_term / m + np.sqrt(
        (1/3 + 1/(2 * log_term)) * (c ** 2 * log_term / (m-1)) ** 2
        + 2 * (c ** 2) * unbiased_e_wimpy * log_term / m
    )
    epsilon_mu = c * log_term / (3 * m) + np.sqrt(2 * (unbiased_e_wimpy + epsilon_v) * log_term / m)
    if return_matrix:
        return np.full_like(stats['sample_variance'], epsilon_mu)
    return epsilon_mu


def old_bennet_union_bound(c: float, delta: float, stats: dict):
    cardinality = stats['cardinality']
    m = stats['m']
    log_term = np.log(3) + np.log(cardinality) - np.log(delta)  # equal to ln(3|I|/d)
    unbiased_e_wimpy = stats['empirical_wimpy'] * m / (m-1)
    epsilon_v = c * c * log_term / (m - 1) + np.sqrt(
        ((c ** 2) * log_term / (m - 1)) ** 2
        + 2 * (c ** 2) * unbiased_e_wimpy * log_term / (m - 1)
    )
    epsilon_mu = c * log_term / (3 * m) + np.sqrt(2 * (unbiased_e_wimpy + epsilon_v) * log_term / m)
    return epsilon_mu


def talagrand_bound(c: float, delta: float, stats: dict):
    m = stats['m']
    b_var = b(m)
    chi = 1 + 2 * b_var
    r = c
    n = TALAGRAND_MC_ERA_N
    e_wimpy = stats['empirical_wimpy']
    u_e_wimpy = m / (m - 1) * e_wimpy
    log_term = np.log(4) - np.log(delta)  # equal to ln(4/d)
    era_bound = stats['centralized_mc_era'] + 2 * stats['centralized_maximum'] * log_term / (3 * n * m) + \
        np.sqrt(4 * e_wimpy * log_term / (n * m))
    lambda_bound = era_bound + 2 * r * chi * log_term / (3 * m) + \
        np.sqrt((r * chi * log_term / (np.sqrt(3) * m)) ** 2 +
                2 * r * chi * (era_bound + r * b_var) * log_term / m)
    modified_lambda_bound = era_bound + 2 * r * chi * log_term / (3 * m) + \
        np.sqrt((r * chi * log_term / (np.sqrt(3) * m)) ** 2 +
                2 * r * chi * r * b_var * log_term / m)
    modified_lambda_bound_2 = era_bound + 2 * r * chi * log_term / (3 * m) + \
        np.sqrt((r * chi * log_term / (np.sqrt(3) * m)) ** 2 +
                2 * r * chi * era_bound * log_term / m)
    wimpy_bound = u_e_wimpy + r ** 2 * log_term / (m - 1) + \
        np.sqrt(((r ** 2 * log_term) / (m - 1)) ** 2 + 2 * (r ** 2) * u_e_wimpy * log_term / (m - 1))
    modified_wimpy_bound = wimpy_bound
    lambda_term = lambda_bound / (1 - 2 * b_var)
    modified_lambda_term = modified_lambda_bound / (1 - 2 * b_var)
    modified_lambda_term_2 = modified_lambda_bound_2 / (1 - 2 * b_var)
    modified_talagrand = 2 * modified_lambda_term
    modified_talagrand_2 = 2 * modified_lambda_term_2
    return 2 * lambda_term + 2 * r * log_term / (3 * m) + \
        np.sqrt(2 * (wimpy_bound + 4 * r * lambda_term) * log_term / m), 2 * era_bound, 2 * lambda_term, modified_talagrand, modified_talagrand_2


def get_stats(sample_utils: np.ndarray, streaming=False, streaming_object=None, return_results=True, rademacher=True,
              mc_era_n=TALAGRAND_MC_ERA_N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if streaming:
        one_era_ = None
        mc_era_ = None
        centralized_mc_era_ = None
        empirical_mean_variance_ = None
        max_min_ = None
        if streaming_object is not None:
            one_era_ = streaming_object['one_era']
            mc_era_ = streaming_object['mc_era']
            centralized_mc_era_ = streaming_object['centralized_mc_era']
            empirical_mean_variance_ = streaming_object['empirical_mean_variance']
            max_min_ = streaming_object['max_min']
        empirical_mean_variance_ = empirical_mean_variance(sample_utils, streaming=True,
                                                           streaming_object=empirical_mean_variance_)
        if rademacher:
            one_era_result, one_era_ = mc_era(1, sample_utils, streaming=True, streaming_object=one_era_,
                                              return_result=return_results)
            mc_era_result, mc_era_ = mc_era(mc_era_n, sample_utils, streaming=True, streaming_object=mc_era_,
                                            return_result=return_results)
            centralized_mc_era_result, centralized_mc_era_ = centralized_mc_era(mc_era_n, sample_utils,
                                                                                empirical_mean_variance_['total_mean'],
                                                                                streaming=True,
                                                                                streaming_object=centralized_mc_era_,
                                                                                return_result=return_results)
        else:
            one_era_result, one_era_ = None, None
            mc_era_result, mc_era_ = None, None
            centralized_mc_era_result, centralized_mc_era_ = None, None
        max_min_ = max_min(sample_utils, streaming_object=max_min_)
        streaming_object = {'one_era': one_era_,
                            'mc_era': mc_era_,
                            'centralized_mc_era': centralized_mc_era_,
                            'empirical_mean_variance': empirical_mean_variance_,
                            'max_min': max_min_}
        if return_results:
            m = empirical_mean_variance_['total_m']
            sample_mean = np.copy(empirical_mean_variance_['total_mean'])
            sample_variance = np.copy(empirical_mean_variance_['total_variance'])
            return {'one_era': one_era_result,
                    'mc_era': mc_era_result,
                    'centralized_mc_era': centralized_mc_era_result,
                    'm': m,
                    'sample_mean': sample_mean,
                    'sample_variance': sample_variance,
                    'empirical_wimpy': (1 - 1 / m) * np.max(sample_variance),
                    'maximum': max(np.amax(np.abs(max_min_['maximum'])),
                                   np.amax(np.abs(max_min_['minimum']))),
                    'centralized_maximum': max(np.amax(np.abs(max_min_['maximum'] - sample_mean)),
                                               np.amax(np.abs(max_min_['minimum'] - sample_mean))),
                    'cardinality': np.prod(sample_utils.shape[1:])}, streaming_object
        else:
            return None, streaming_object
    empirical_mean_variance_results = empirical_mean_variance(sample_utils)
    m = empirical_mean_variance_results['total_m']
    sample_mean = np.copy(empirical_mean_variance_results['total_mean'])
    sample_variance = np.copy(empirical_mean_variance_results['total_variance'])
    maximum = np.amax(np.abs(sample_utils))
    centralized_maximum = np.amax(np.abs(sample_utils - sample_mean))
    if rademacher:
        one_era_result = mc_era(1, sample_utils)
        mc_era_result = mc_era(mc_era_n, sample_utils)
        centralized_mc_era_result = centralized_mc_era(mc_era_n, sample_utils, sample_mean)
    else:
        one_era_result = None
        mc_era_result = None
        centralized_mc_era_result = None
    return {'one_era': one_era_result,
            'mc_era': mc_era_result,
            'centralized_mc_era': centralized_mc_era_result,
            'm': m,
            'sample_mean': sample_mean,
            'sample_variance': sample_variance,
            'empirical_wimpy': (1 - 1 / m) * np.max(sample_variance),
            'maximum': maximum,
            'centralized_maximum': centralized_maximum,
            'cardinality': np.prod(sample_utils.shape[1:])}


def mc_era(n: int, sample_utils: np.ndarray, streaming=False, streaming_object=None, return_result=True):
    m = sample_utils.shape[0]
    rvs = 2 * np.random.binomial(1, 0.5, size=(n, m)) - 1
    averages = 1 / m * np.tensordot(rvs, sample_utils, axes=([1], [0])).reshape((n, -1))
    if streaming:
        m_past = 0
        if streaming_object is not None:
            m_past = streaming_object['m_past']
            averages = m / (m + m_past) * averages + m_past / (m + m_past) * streaming_object['averages_past']
        streaming_object = {'m_past': m + m_past, 'averages_past': averages}
        if return_result:
            return np.mean(np.amax(np.abs(averages), axis=1)), streaming_object
        else:
            return None, streaming_object
    return np.mean(np.amax(np.abs(averages), axis=1))


def centralized_mc_era(n: int, sample_utils: np.ndarray, total_sample_mean: np.ndarray,
                       streaming=False, streaming_object=None, return_result=True):
    m = sample_utils.shape[0]
    rvs = 2 * np.random.binomial(1, 0.5, size=(n, m)) - 1
    averages = 1 / m * np.tensordot(rvs, sample_utils, axes=([1], [0])).reshape((n, -1))
    if streaming:
        m_past = 0
        rvs_sum = np.sum(rvs, axis=1)
        if streaming_object is not None:
            m_past = streaming_object['m_past']
            averages = m / (m + m_past) * averages + m_past / (m + m_past) * streaming_object['averages_past']
            rvs_sum += streaming_object['rvs_sum']
        streaming_object = {'m_past': m + m_past, 'averages_past': averages, 'rvs_sum': rvs_sum}
        if return_result:
            centralized_averages = averages - 1 / (m+m_past) * np.outer(rvs_sum, total_sample_mean.reshape(-1))
            return np.mean(np.amax(np.abs(centralized_averages), axis=1)), streaming_object
        else:
            return None, streaming_object
    centralized_averages = averages - 1 / m * np.outer(np.sum(rvs, axis=1), total_sample_mean.reshape(-1))
    return np.mean(np.amax(np.abs(centralized_averages), axis=1))


def empirical_mean_variance(sample_utils: np.ndarray, streaming=False, streaming_object=None):
    # https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
    m = sample_utils.shape[0]
    sample_mean = np.mean(sample_utils, axis=0)
    if m == 1:
        sample_variance = 0
    else:
        sample_variance = 1 / (m - 1) * np.sum(np.square(sample_utils - sample_mean), axis=0)
    if streaming:
        total_m = m
        total_mean = sample_mean
        total_variance = sample_variance
        if streaming_object is not None:
            total_m = streaming_object['total_m']
            total_mean = streaming_object['total_mean']
            total_variance = streaming_object['total_variance']
            total_variance = ((total_m - 1) * total_variance + (m - 1) * sample_variance) / (total_m + m - 1) + \
                total_m * m * np.square(total_mean - sample_mean) / ((total_m + m) * (total_m + m - 1))
            total_mean = total_mean * total_m / (total_m + m) + sample_mean * m / (total_m + m)
            total_m = total_m + m
        streaming_object = {'total_m': total_m,
                            'total_mean': total_mean,
                            'total_variance': total_variance}
        return streaming_object
    return {'total_m': m,
            'total_mean': sample_mean,
            'total_variance': sample_variance}


def unbiased_empirical_wimpy(sample_variance: np.ndarray):
    return np.max(sample_variance)


def max_min(sample_utils: np.ndarray, streaming_object=None):
    maximum = np.amax(sample_utils, axis=0)
    minimum = np.amin(sample_utils, axis=0)
    if streaming_object is not None:
        maximum = np.maximum(maximum, streaming_object['maximum'])
        minimum = np.minimum(minimum, streaming_object['minimum'])
    streaming_object = {'maximum': maximum, 'minimum': minimum}
    return streaming_object


def centralization(sample_utils: np.ndarray):
    return sample_utils - np.mean(sample_utils, axis=0)


def b(m: int):
    if m > 1000:
        return np.sqrt(2 / (math.pi * m))
    return comb(m, 1 + np.ceil(m / 2)) * (1 + np.ceil(m / 2)) / (m * (2 ** (m - 1)))


def h(x):
    return (1 + x) * np.log(1 + x) - x

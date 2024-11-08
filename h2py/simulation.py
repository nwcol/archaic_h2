
import numpy as np
import scipy


def setup_regions(L, size):
    """
    
    """
    regions = np.stack(
        (np.arange(0, L, size), np.arange(size, L + size, size)), 
        axis=1,
        dtype=np.int64
    )
    return regions


def get_random_regions(L, scale):
    """
    Generate an array of random regions by sampling lengths from an exponential
    distribution with rate 1 / scale. 
    """
    start, end = 0, 0
    regions = []
    while end < L:
        end = start + int(np.ceil(np.random.exponential(scale)))
        regions.append([start, end])
        start = end
    regions = np.array(regions, dtype=np.int64)
    regions[-1, 1] = L
    return regions


def generate_mutation_map(L, ):

    return


def regenerate_mutation_map(mut_map, corr):


    return


def generate_recombination_map():


    return 


def regenerate_recombination_map():


    return



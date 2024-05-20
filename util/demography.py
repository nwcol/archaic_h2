
"""
Functions for manipulating demes demographies
"""

import demes
import demesdraw
import matplotlib.pyplot as plt
from matplotlib import cm
import moments.Demes.Inference as mi
import numpy as np
import pickle
import pyaml
import yaml


# maps names of representative samples to demes
# mostly necessary because - is not allowed in deme names
name_map = {
    "Altai": "Altai",
    "Chagyrskaya": "Chagyrskaya",
    "Denisova": "Denisovan",
    "Vindija": "Vindija",
    "French-1": "French1",
    "Han-1": "Han1",
    "Papuan-2": "Papuan2",
    "Khomani_San-2": "KhomaniSan2",
    "Yoruba-1": "Yoruba1",
    "Yoruba-3": "Yoruba3"
}


start_times = {
    "Altai": 1.1e5,
    "Chagyrskaya": 8e4,
    "Denisova": 7e4,
    "Vindija": 5e4,
    "French-1": 0,
    "Han-1": 0,
    "Papuan-2": 0,
    "Khomani_San-1": 0,
    "Yoruba-1": 0,
    "Yoruba-3": 0
}


def build_pair_demography(
        deme_x,
        deme_y,
        time_units="years",
        N_A=1e4,
        N_x=1e4,
        N_y=1e4,
        T=1.5e5,
        t_x=0,
        t_y=0,
        m=None,
        generation_time=30,
):
    builder = demes.Builder(
        time_units=time_units, generation_time=generation_time
    )
    builder.add_deme(
        "ancestral", epochs=[{"end_time": T, "start_size": N_A}]
    )
    builder.add_deme(
        deme_x, ancestors=["ancestral"],
        epochs=[{"start_size": N_x, "end_time": t_x}]
    )
    builder.add_deme(
        deme_y, ancestors=["ancestral"],
        epochs=[{"start_size": N_y, "end_time": t_y}]
    )
    if m:
        builder.add_migration(rate=m, demes=[deme_x, deme_y])
    graph = builder.resolve()
    return graph


def build_pair_params(
        deme_x,
        deme_y,
        T_min=1e3,
        T_max=1e6,
        N_A_min100,
        N_A_max=1e5,
        N_min=100,
        N_max=1e5,
        m=None,
        m_min=1e-10,
        m_max=1e-3
):
    """
    Set up a dictionary of parameters for a pairwise divergence model
    """
    param_dict = {
        "parameters": [
            get_deme_param("T", "ancestral", "end_time", T_min, T_max),
            get_deme_param("N_A", "ancestral", "start_size", N_A_min, N_A_max),
            get_deme_param(f"N_{deme_x}", deme_x, "start_size", N_min, N_max),
            get_deme_param(f"N_{deme_y}", deme_y, "start_size", N_min, N_max)
        ]
    }
    if m:
        param_dict["parameters"] += [get_mig_param("m", m_min, m_max)]
    return param_dict


def get_deme_param(name, deme, defines, lower, upper, epoch=0):
    """
    Works only when you need to define a parameter that controls a single
    value

    :param name:
    :param deme:
    :param defines:
    :param lower:
    :param upper:
    :return:
    """
    param = {
        "name": name,
        "values": [{"demes": {deme: {"epochs": {epoch: defines}}}}],
        "lower_bound": lower,
        "upper_bound": upper
    }
    return param


def get_mig_param(name, lower, upper, idx=0):

    param = {
        "name": name,
        "values": [{"migrations": {idx: "rate"}}],
        "lower_bound": lower,
        "upper_bound": upper
    }
    return param


def get_random_params(graph, param_defines):

    # get demes graph dictionary and parameter dictionary
    builder = graph.asdict_simplified()

    # use existing moments functionality to get parameter bounds etc
    param_names, params_0, lower_bound, upper_bound = \
        mi._set_up_params_and_bounds(param_defines, builder)
    constraints = mi._set_up_constraints(param_defines, param_names)

    # draw parameter values uniformly randomly
    constraints_satisfied = False

    while not constraints_satisfied:
        params = np.random.uniform(lower_bound, upper_bound)

        if constraints:
            if np.all(constraints(params) > 0):
                constraints_satisfied = True
        else:
            constraints_satisfied = True

    builder = mi._update_builder(builder, param_defines, params)
    graph = demes.Graph.fromdict(builder)
    return graph


# OLD


def build_pairwise_model(pop1_name, pop2_name, graph_file_name,
                         params_file_name, time_units="years",
                         generation_time=25):
    """
    Set up a graph defining a simple pairwise model with migration

    parameters:
    T divergence time
    m migration rate
    N_a ancestral N_e
    N_1 population 1 N_e
    N_2 population 2 N_e

    :param pop1_name:
    :param pop2_name:
    :param graph_file_name:
    :param params_file_name:
    :param time_units:
    :param generation_time:
    :return:
    """
    init_m = 1e-5
    init_T = 1e5
    init_N = 1e4

    # set up the graph
    builder = demes.Builder(
        time_units=time_units, generation_time=generation_time
    )
    builder.add_deme(
        "ancestral", epochs=[{"end_time": init_T, "start_size": init_N}]
    )
    builder.add_deme(
        pop1_name, ancestors=["ancestral"], epochs=[{"start_size": init_N}]
    )
    builder.add_deme(
        pop2_name, ancestors=["ancestral"], epochs=[{"start_size": init_N}]
    )
    builder.add_migration(rate=init_m, demes=[pop1_name, pop2_name])
    graph = builder.resolve()
    demes.dump(graph, graph_file_name)

    # set up parameters
    param_dict = {
        "parameters": [
            get_param("T", "ancestral", "end_time", 1e2, 1e6),
            get_param("N_a", "ancestral", "start_size", 1e2, 1e5),
            get_param("N_1", pop1_name, "start_size", 1e2, 1e5),
            get_param("N_2", pop2_name, "start_size", 1e2, 1e5),
            {"name": "m", "values": [{"migrations": {0: "rate"}}],
             "lower_bound": 0, "upper_bound": 1}
        ]
    }
    with open(params_file_name, 'w') as file:
        yaml.dump(param_dict, file)
    return 0







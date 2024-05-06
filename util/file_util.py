
"""
Functions for reading and writing my own statistics files (.json, .txt, etc)
"""

import json
import numpy as np
import os


"""
Interconverting dictionaries of row vectors with 2d arrays.
"""


def arr_to_dict(arr, row_names):
    """
    Return a dictionary that maps row_names to rows in arr
    """
    n_rows = len(arr)
    dic = {row_names[i]: arr[i] for i in range(n_rows)}
    return dic


def dict_to_arr(dic):
    """
    Return an array and list of row_names of row vectors in dic
    """
    row_names = list(dic.keys())
    arr = np.array([dic[row_name] for row_name in row_names])
    return arr, row_names


"""
Saving and loading data
"""


def save_arr(file_name, arr, header):

    with open(file_name, "w") as file:
        np.savetxt(file, arr, header=str(header))


def load_vec(file_name):
    """
    Load an array and its header. Single rows are returned as 2d arrays
    """
    with open(file_name, 'r') as file:
        header_line = file.readline().strip("\n").strip("#")
    header = eval(header_line)
    arr = np.loadtxt(file_name)
    return header, arr


def load_arr(file_name):
    """
    Load an array and its header. Single rows are returned as 2d arrays
    """
    with open(file_name, 'r') as file:
        header_line = file.readline().strip("\n").strip("#")
    header = eval(header_line)
    arr = np.loadtxt(file_name)
    # in case the file contains one number and loads as array(N)
    if arr.ndim < 1:
        arr = np.array([arr])
    # if there's a single column in the file...
    if arr.ndim < 2:
        # if the rows dictionary specifies one row; we have a single element
        if "rows" not in header:
            pass
        if len(header["rows"]) == 1:
            arr = arr[np.newaxis, :]
        # if the rows dictionary specifies many rows; then we have a column
        else:
            arr = arr[:, np.newaxis]
    return header, arr


def read_header(file_name):

    with open(file_name, 'r') as file:
        header_str = file.readline().strip("\n").strip("#")
    header = eval(header_str)
    return header


def save_dict_as_arr(file_name, stat_dict, header=None):
    """
    Save a dictionary mapping names to equal-length arrays as values as a
    np array .txt file with a header defining the row names
    The array will be 2d even in the case where there is just one row
    """
    row_names = list(stat_dict.keys())
    n_rows = len(row_names)
    rows = dict(zip(np.arange(n_rows), row_names))
    if not header:
        header = {"rows": rows}
    else:
        header["rows"] = rows
    arr = np.array([stat_dict[rows[idx]] for idx in rows])
    np.savetxt(file_name, arr, header=str(header))


def load_arr_as_dict(file_name):
    """
    Load an array with row names specified in its header and transform it
    into a dict mapping names to rows
    """
    header, arr = load_arr(file_name)
    row_names = header["rows"]
    arr_dict = {row_names[idx]: arr[idx] for idx in row_names}
    return header, arr_dict


def load_as_dict(file_name):
    # uses col/row lists, not dicts
    header, arr = load_arr(file_name)
    row_names = header["rows"]
    arr_dict = {row_names[i]: arr[i] for i in range(len(arr))}
    return header, arr_dict


"""
Conveniently loading statistics from directories
"""

def load_H2(dir, sample_ids):

    site_pairs = np.loadtxt(f"{dir}/site_pairs.txt").sum(0)
    h2_dict = {x: np.loadtxt(f"{dir}/het_pairs_{x}.txt").sum(0) / site_pairs
               for x in sample_ids}
    return h2_dict







"""
Old stuff which isn't used anymore
"""


def sort_window_ids(window_ids):

    sorted_win_ids = []
    for chr in np.arange(1, 23):
        this_chr = [x for x in window_ids if f"chr{chr}_" in x]
        win_ids = [int(x.split("_")[1].lstrip("win")) for x in this_chr]
        idx = searchsort_list(win_ids)
        sorted_win_ids += [this_chr[i] for i in idx]
    return sorted_win_ids


def searchsort_list(lis):

    srt = sorted(lis)
    idx = [lis.index(x) for x in srt]
    return idx


def get_header_str(rows, *args, **kwargs):

    header_dict = {key: kwargs[key] for key in kwargs}
    for arg in args:
        if type(arg) == dict:
            for key in arg:
                header_dict[key] = arg[key]
        else:
            pass
    header_dict["rows"] = rows
    return str(header_dict)


def get_header(**kwargs):

    header_dict = {key: kwargs[key] for key in kwargs}
    return header_dict


def dicts_to_arr(dict_list):
    """
    Transform a list of dictionaries mapping sample ids to rows of statistics
    into a single dictionary mapping
    """
    n_rows = len(dict_list)
    sample_ids = list(dict_list[0]["rows"].keys())
    n_cols = len(dict_list[0]["rows"][sample_ids[0]])
    arr_dict = {
        sample_id: np.zeros((n_rows, n_cols)) for sample_id in sample_ids
    }
    all_dicts = {
        (x["chrom"], x["window_id"]): x["rows"] for x in dict_list
    }
    dict_ids = list(all_dicts.keys())
    dict_ids.sort()
    rows = {}
    for i, ids in enumerate(dict_ids):
        window_dict = all_dicts[ids]
        for sample_id in sample_ids:
            arr_dict[sample_id][i] = window_dict[sample_id]
        rows[i] = f"chr{ids[0]}_win{ids[1]}"
    return rows, arr_dict


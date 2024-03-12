
# Mostly for reading and writing my own statistics files (.json, .txt)


import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys
import os


class Table:

    def __init__(self, arr, rows, cols, row_class, col_class, **kwargs):
        self.arr = arr
        self.rows = rows
        self.row_class = row_class
        self.cols = cols
        self.col_class = col_class

    @classmethod
    def empty(cls, shape):

        return 0

    def __getitem__(self, item):

        row_id, col_id = item
        if not row_id:
            row_idx = slice(None, None, 1)
        elif type(row_id) == slice:
            row_idx = row_id
        else:
            row_idx = self.rows[row_id]
        #
        if not col_id:
            col_idx = slice(None, None, 1)

        else:
            col_idx = self.cols[col_id]
        return self.arr[row_idx, col_idx]


# window files (JSON)


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


# transforming data


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


# saving and loading files

def read_header(file_name):

    with open(file_name, 'r') as file:
        header_str = file.readline().strip("\n").strip("#")
    header = eval(header_str)
    return header


def save_arr(file_name, arr, header):

    file = open(file_name, "w")
    np.savetxt(file, arr, header=str(header))
    file.close()


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
        # if the rows dictionary specifies one row; we have a single row
        if len(header["rows"]) == 1:
            arr = arr[np.newaxis, :]
        # if the rows dictionary specifies many rows; then we have a column
        else:
            arr = arr[:, np.newaxis]
    return header, arr


def load_arr_as_dict(file_name):
    """
    Load an array with row names specified in its header and transform it
    into a dict mapping names to rows
    """
    header, arr = load_arr(file_name)
    row_names = header["rows"]
    arr_dict = {row_names[idx]: arr[idx] for idx in row_names}
    return header, arr_dict


def load_arr_as_structured_dict(file_name):
    # for loading raw stats; contains the chromosome and window ids
    header, arr = load_arr(file_name)
    row_names = header["rows"]
    out = {
        "chrom": int(header["chrom"]),
        "window_id": int(header["window_id"]),
        "statistic": header["statistic"],
        "rows": {row_names[idx]: arr[idx] for idx in row_names}
    }
    return out


# loading several files from directories


def load_arrs(dir_name, tag):
    # load arrays lol
    file_names = [name for name in os.listdir(dir_name) if tag in name]
    dir_name = dir_name.rstrip("/")
    arrs = []
    for file_name in file_names:
        full_file_name = f"{dir_name}/{file_name}"
        header, arr = load_arr(full_file_name)
        arrs.append(arr)
    rows = header["rows"]
    return rows, arrs


def load_dicts(dir_name, tag):

    file_names = [name for name in os.listdir(dir_name) if tag in name]
    dir_name = dir_name.rstrip("/")
    dicts = []
    for file_name in file_names:
        full_file_name = f"{dir_name}/{file_name}"
        header, stat_dict = load_arr_as_dict(full_file_name)
        dicts.append(stat_dict)
    return dicts


def load_structured_dicts(dir_name, tag=None):

    if tag:
        file_names = [name for name in os.listdir(dir_name) if tag in name]
    else:
        file_names = [name for name in os.listdir(dir_name)]
    dir_name = dir_name.rstrip("/")
    dicts = []
    for file_name in file_names:
        full_file_name = f"{dir_name}/{file_name}"
        dicts.append(load_arr_as_structured_dict(full_file_name))
    return dicts





# trash





def load_stat_dir(path, statistic):
    stats = []
    chr_dirs = [x for x in os.listdir(path) if "chr" in x]
    for x in chr_dirs:
        dir_list = os.listdir(f"{path}/{x}")
        for y in dir_list:
            if statistic in y:
                stats.append(load_statistics(f"{path}/{x}/{y}"))
    stat_arr = get_statistic_arr(stats)
    return stat_arr


def load_flat_stat_dir(path, statistic):
    stats = []
    dir_list = os.listdir(path)
    for y in dir_list:
        if statistic in y:
            stats.append(load_statistics(f"{path}/{y}"))
    stat_arr = get_statistic_arr(stats)
    return stat_arr




def get_window_list():

    window_path = f"{data_path}/windows"
    window_names = []
    for i in np.arange(1, 23):
        window_file = open(f"{window_path}/chr{i}_windows.json", 'r')
        window_dict = json.load(window_file)[str(i)]["windows"]
        window_file.close()
        for j in np.arange(40):
            if str(j) in window_dict:
                window_names.append(f"chr{i}_win{j}")
            else:
                break
    return window_names

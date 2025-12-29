import os
import re
import json
import math
import random
import warnings
import cv2 as cv
import numpy as np
import operator
import pandas as pd
import networkx as nx
from shutil import copytree
from functools import reduce
import matplotlib.pyplot as plt
from numpy.ma import masked_array
from scipy.spatial import KDTree
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.filters import farid, threshold_multiotsu
from skimage.morphology import disk, remove_small_objects
from typing import Union, Set, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def encode_string(_str: str):
    import base64
    return (base64.b64encode(_str.encode('ascii'))).decode('ascii')


def decode_string(_str: str):
    import base64
    return (base64.b64decode(_str.encode('ascii'))).decode('ascii')


def read_json(json_file, print_content=False):
    with open(json_file) as j_file:
        contents = json.load(j_file)
    if print_content:
        from pprint import pprint as pretty
        pretty(contents)

    return contents


def write_json(data, json_file, sort_keys: bool = False):
    with open(json_file, 'w') as j_file:
        json.dump(data, j_file, indent=2, sort_keys=sort_keys)


def copy_file(dir_src, dir_des, dict_patient_include):
    import progressbar

    num_patient = sum([len(v) for v in dict_patient_include.values()])
    num = 0
    with progressbar.ProgressBar(max_value=num_patient) as bar:
        for cls in dict_patient_include.keys():
            dir_sub = os.listdir(os.path.join(dir_src, cls))
            for _f in dir_sub:
                if str(_f).split(' ')[0] in dict_patient_include[cls]:
                    num += 1
                    if os.path.isdir(os.path.join(dir_des, cls, str(_f).split(' ')[0])):
                        continue
                    else:
                        print(f'Copy files from {os.path.join(dir_src, cls, _f)} to {dir_des}...')
                        copytree(os.path.join(dir_src, cls, _f), os.path.join(dir_des, cls, str(_f).split(' ')[0]))
                    bar.update(num)


def sort_list_by_idx_e_in_tuple(tuples_list: List[tuple], key_idx: int = 0, skip_idx_e: bool = True,
                                reverse: bool = False, return_index: bool = False):
    """
    :param tuples_list: a list of tuples of the same size (with at least two elements) to be sorted.
    :param key_idx: index of key element in tuple used for sorting.
    :param skip_idx_e: bool on whether to skip the index element for sorting, default: True.
    :param reverse: whether to reverse the order
    :param return_index: bool on whether the indices after sorting should be returned.
    :return: _list_sorted:
    """
    assert len(tuples_list[0]) >= 2

    sorted_list = sorted(tuples_list, key=lambda x: x[key_idx], reverse=reverse)
    if return_index:
        indices = [tuples_list.index(_t) for _t in sorted_list]
    if skip_idx_e:
        if len(tuples_list[0]) == 2:
            if key_idx == 0:
                sorted_list = [_t[1] for _t in sorted_list]
            else:
                sorted_list = [_t[0] for _t in sorted_list]
        else:
            sorted_list = [(*_t[:key_idx], *_t[key_idx:]) for _t in sorted_list]
    if not return_index:
        return sorted_list
    else:
        # noinspection PyUnboundLocalVariable
        return sorted_list, indices


def split_array1d_by_slice_indices(idx_slices: np.ndarray, size: int):
    # It is important that the idx_slices are already sorted!!!
    idx_array_full = np.arange(size)
    # noinspection PyUnusedLocal
    idx_list_split = [None]*(len(idx_slices)+1)
    idx_list_split[0] = idx_array_full[:idx_slices[0]]
    if len(idx_slices) > 1:
        for i in range(1, len(idx_slices)):
            idx_list_split[i] = idx_array_full[idx_slices[i-1]+1:idx_slices[i]]
    idx_list_split[-1] = idx_array_full[idx_slices[-1]+1:]
    return idx_list_split


def get_key_by_value_from_dict(_dict: dict, _value):
    return list(_dict.keys())[list(_dict.values()).index(_value)]


def remove_empty_list_in_dict(_dict):
    key_list = list(_dict.keys())
    for k in key_list:
        if len(_dict[k]) == 0:
            _dict.pop(k)
    return _dict


def initiate_nested_list(size):
    # Don't use [[]] * series_num for initialization. Otherwise, all lists will be appended with the same member.
    # noinspection PyUnusedLocal
    return [[] for i in range(size)]


def concatenate_substrings(_str: str):
    words = _str.split()
    # Initialize the result string
    result = ""
    for i in range(len(words)):
        if i == 0:
            result += words[i]
        else:
            if words[i][0].isupper():
                if result[-1].isupper():
                    result += "-" + words[i]
                else:
                    result += words[i].capitalize()
            else:
                result += words[i].capitalize()
    return result


class NestedDict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def nested_dict_hierarchy_like(_dict: NestedDict, nested_type: str, **kwargs):
    from copy import deepcopy

    def create_np_array():
        if 'size' not in kwargs.keys():
            raise NameError(f"Data size must be provided, either as array shape or as a single number!")
        else:
            return np.zeros(kwargs['size'])

    def get_init_item(_d):
        if isinstance(_d, dict):
            for _k, _v in _d.items():
                if isinstance(_v, dict):
                    get_init_item(_v)
                else:
                    if nested_type == 'list':
                        _d[_k] = list()
                    elif nested_type == 'array':
                        _d[_k] = create_np_array()
                    else:
                        raise NotImplementedError("Only 'list' or 'array' are supported as input!")

    new_dict = deepcopy(_dict)
    get_init_item(new_dict)
    return new_dict


def get_path_to_key_in_nested_dict(_dict: dict, target_key: str):
    # target key name must be unique
    def _get_path(_d, _key):
        nonlocal path
        for k, v in _d.items():
            if k.lower() == _key.lower():
                path += (k, )
            else:
                if isinstance(v, dict) and (_key.lower() not in [_k.lower() for _k in _d.keys()]):
                    path += (k,)
                    _get_path(v, _key)

    path = ()
    _get_path(_dict, target_key)
    return path


def get_value_in_nested_dict_by_target_key(_dict: dict, target_key: str):
    _path = get_path_to_key_in_nested_dict(_dict, target_key)
    return reduce(operator.getitem, _path, _dict)


def get_value_in_nested_dict_by_path(_dict: dict, _path: Union[list, tuple]):
    return reduce(operator.getitem, _path, _dict)


def swap_tuple(_tuple: tuple, a: int, b: int):
    _list = list(_tuple)
    _list[b], _list[a] = _list[a], _list[b]
    return tuple(_list)


def resample_with_low_repeats(_list: list, size: int, random_seed: int = 1):
    np.random.seed(random_seed)
    num = len(_list)
    assert num > 0 and size > 0
    if num >= size:
        indices = np.random.choice(np.arange(num), size, replace=False)
    else:
        indices = np.random.choice(np.repeat(np.arange(num), int(np.ceil(size/num))), size, replace=False)
    new_list = [_list[i] for i in indices]
    return new_list


def set_color_labels_for_display(cls_map: dict):
    num_labels = len(cls_map)
    colors = set_colors(num_labels)
    dict_colors = {k: colors[i, :] for i, k in enumerate(cls_map.keys())}
    return dict_colors


def remove_empty_nested_keys(_dict, reset_dict_to_count=False, count_ini=0):
    """
    This method is used to remove empty elements in nested dictionary upto three level.
    :param _dict: Input nested dictionary
    :param reset_dict_to_count: binary on whether to reset the dictionary as a counter
    :param count_ini: default value 0 as the initial count, valid only if 'reset_dict_to_count' is set to True.
    :return:
    """
    for _k in _dict.keys():
        if isinstance(_dict[_k], dict):
            _k_list = list(_dict[_k].keys())
            for _kk in _k_list:
                if isinstance(_dict[_k][_kk], dict):
                    _kk_list = list(_dict[_k][_kk].keys())
                    for _kkk in _kk_list:
                        if (not isinstance(_dict[_k][_kk][_kkk], int) and len(_dict[_k][_kk][_kkk]) == 0) or \
                                (isinstance(_dict[_k][_kk][_kkk], int) and _dict[_k][_kk][_kkk] == 0):
                            _dict[_k][_kk].pop(_kkk)
                        else:
                            if reset_dict_to_count:
                                _dict[_k][_kk][_kkk] = count_ini

                    if len(_dict[_k][_kk].keys()) == 0:
                        _dict[_k].pop(_kk)
                else:
                    if (not isinstance(_dict[_k][_kk], int) and len(_dict[_k][_kk]) == 0) or \
                            (isinstance(_dict[_k][_kk], int) and _dict[_k][_kk] == 0):
                        _dict[_k].pop(_kk)
                    else:
                        if reset_dict_to_count:
                            _dict[_k][_kk] = count_ini


def get_files_in_folder(file_dir, ext: Union[str, None] = None, string_in_filename: Union[str, None] = None,
                        keep_file_with_str=False, append_dir: bool = False):
    files = os.listdir(file_dir)
    if ext is not None:
        files = [file for file in files if file.endswith(ext)]
    if string_in_filename is not None:
        if keep_file_with_str:
            idx_to_keep = np.where(np.array([string_in_filename in i for i in files]))[0]
        else:
            idx_to_keep = np.where(np.array([string_in_filename not in i for i in files]))[0]
        files = [files[i] for i in idx_to_keep]
    if append_dir:
        return [os.path.join(file_dir, file) for file in files]
    else:
        return files


def clear_fig_in_memory(fig_keep=None):
    all_fig_nums = plt.get_fignums()
    for i in all_fig_nums:
        fig = plt.figure(i)
        if fig_keep is not None:
            if isinstance(fig_keep, list):
                if fig in fig_keep:
                    continue
                else:
                    fig.clear()
                    plt.close(fig)
            else:
                if fig != fig_keep:
                    fig.clear()
                    plt.close(fig)
        else:
            fig.clear()
            plt.close(fig)


def print_highlighted_text(_str):
    print("\x1b[7;33;40m" + _str + "\x1b[0m")


def date_str_match_test(_str):
    return bool(re.match('(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})', _str))


def delete_variables_but(_list: list):
    for var in list(globals().keys()):
        if var not in _list and not var.startswith("__"):
            del globals()[var]


def set_colors(num_colors, cmap_str='turbo'):
    import copy
    """Assigns a random colored material for each found segment"""
    # Use deep copy to avoid modification to inbuilt color map.
    # noinspection PyUnresolvedReferences
    colors_cmp = copy.deepcopy(plt.get_cmap(cmap_str).colors)
    colors_idx = np.linspace(0, 255, num_colors).astype('int')
    color_rgb = np.array([colors_cmp[i] for i in colors_idx])
    return color_rgb


def get_col_idx_by_substr_from_dataframe(df: pd.DataFrame, substr: str):
    indices = [df.columns.get_loc(col) for col in df.columns if substr in col]
    if len(indices) > 0:
        if len(indices) == 1:
            return indices[0]
        else:
            warnings.warn(f'Multiple columns are found to contain substring {substr}: {df.columns[indices]}.')
            return indices
    else:
        warnings.warn(f'No colum names contain substring {substr}!')
        return -1


def normalized_image_to_8bit(img: np.ndarray):
    assert (np.min(img) >= 0 and np.max(img) <= 1)
    img_8bit = 255 * img
    img_8bit = img_8bit.astype(np.uint8)
    return img_8bit


class DirectoryGraph:
    """A class for building directory structures as NetworkX graphs with file filtering."""

    def __init__(self, root_path: str):
        """
        Initialize the DirectoryGraph.

        Args:
            root_path: Path to the root directory to analyze
        """
        self.root_path = os.path.abspath(root_path)
        if not os.path.exists(self.root_path):
            raise ValueError(f"Path does not exist: {root_path}")

        self.graph = nx.DiGraph()
        self._leaf_nodes = None

    def build_graph(
            self,
            include_files: bool = False,
            max_depth: Optional[int] = None,
            file_filter: Optional[Callable[[str], bool]] = None,
    ) -> nx.DiGraph:
        """
        Build a NetworkX graph representing the directory structure.

        Args:
            include_files: If True, include files in the graph
            max_depth: Maximum depth to traverse (None for unlimited)
            file_filter: Function that takes a file path and returns True if
                         the file should be considered when checking directories.

        Returns:
            A NetworkX DiGraph representing the filtered directory structure
        """
        self.graph = nx.DiGraph()
        self._leaf_nodes = None
        require_file_in_leaf = True if file_filter is not None else False
        # Start DFS from root
        stack = [(self.root_path, 0, None, False)]  # (path, depth, parent_path, is_valid)

        while stack:
            current_path, depth, parent_path, parent_valid = stack.pop()

            # Skip if beyond max depth
            if max_depth is not None and depth > max_depth:
                continue

            # Check if we should process this directory
            should_process = True
            if require_file_in_leaf and not parent_valid and depth > 0:
                # If parent wasn't valid and we require files in leaves,
                # skip processing this branch entirely
                should_process = False

            if not should_process:
                continue

            # Initialize this directory's validity
            current_valid = False

            # Check files in this directory
            try:
                items = os.listdir(current_path)
            except (PermissionError, OSError):
                continue

            # Process items
            subdirs = []
            matching_files_found = False

            for item in items:
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    subdirs.append(item_path)
                elif os.path.isfile(item_path):
                    # Check if file matches filter
                    if file_filter is not None and file_filter(item_path):
                        matching_files_found = True

            # Determine if current directory should be added to graph
            should_add_to_graph = False

            if not require_file_in_leaf:
                # Always add directories if no file requirement
                should_add_to_graph = True
                current_valid = True
            else:
                # Only add if we have matching files or valid subdirectories will be added
                if matching_files_found:
                    should_add_to_graph = True
                    current_valid = True
                elif subdirs:
                    # We might add it later if subdirectories are valid
                    # For now, mark as potentially valid
                    current_valid = True

            # Add directory to graph if needed
            if should_add_to_graph:
                self.graph.add_node(
                    current_path,
                    label=os.path.basename(current_path),
                    type='directory',
                    depth=depth,
                    path=current_path
                )

                # Add edge from parent if exists
                if parent_path is not None and parent_path in self.graph:
                    self.graph.add_edge(parent_path, current_path)

                # Add files if requested
                if include_files and matching_files_found:
                    # Add only matching files
                    for item in items:
                        item_path = os.path.join(current_path, item)
                        if os.path.isfile(item_path) and file_filter is not None and file_filter(item_path):
                            self.graph.add_node(
                                item_path,
                                label=item,
                                type='file',
                                depth=depth + 1,
                                path=item_path
                            )
                            self.graph.add_edge(current_path, item_path)

            # Add subdirectories to stack
            for subdir in subdirs:
                stack.append((subdir, depth + 1, current_path, current_valid))

        return self.graph

    def get_leaf_directories(self) -> List[str]:
        """
        Get all leaf directories in the current graph.

        Returns:
            List of full paths to leaf directories
        """
        if self._leaf_nodes is not None:
            return self._leaf_nodes.copy()

        leaf_dirs = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') != 'directory':
                continue

            # A directory is a leaf if it has no outgoing edges to other directories
            has_subdirs = any(
                self.graph.nodes[neighbor].get('type') == 'directory'
                for neighbor in self.graph.successors(node)
            )

            if not has_subdirs:
                leaf_dirs.append(node)

        self._leaf_nodes = leaf_dirs
        return leaf_dirs

    # noinspection PyMethodMayBeStatic
    def create_extension_filter(
            self,
            extensions: Set[str],
            exclude_files: Optional[Set[str]] = None
    ) -> Callable[[str], bool]:
        """
        Create a filter function for specific file extensions.

        Args:
            extensions: Set of file extensions to include (e.g., {'.txt', '.jpg'})
            exclude_files: Set of specific filenames to exclude (case-insensitive)

        Returns:
            A function that can be used as a file_filter in build_graph
        """
        if exclude_files is None:
            exclude_files = set()

        # Convert to lowercase for case-insensitive comparison
        extensions_lower = {ext.lower() for ext in extensions}
        exclude_files_lower = {f.lower() for f in exclude_files}

        def extension_filter(file_path: str) -> bool:
            # Get filename and extension
            filename = os.path.basename(file_path)
            filename_lower = filename.lower()

            # Check if filename is in exclude list
            if filename_lower in exclude_files_lower:
                return False

            # Check extension using os.path.splitext
            _, ext = os.path.splitext(filename)
            return ext.lower() in extensions_lower

        return extension_filter

    def to_networkx(self) -> nx.DiGraph:
        """Return the NetworkX graph object."""
        return self.graph


def get_largest_subset(list_of_sets, thresh=95, seed=None, num_threads=4):
    """Specialized for high thresholds like 95%"""

    n_sets = len(list_of_sets)
    required = int(n_sets * thresh / 100)

    # Phase 1: Ultra-fast counting
    element_counts = {}
    for s in list_of_sets:
        for elem in s:
            element_counts[elem] = element_counts.get(elem, 0) + 1

    # For 95%, we need elements with VERY high frequency
    frequent_elements = [elem for elem, count in element_counts.items()
                         if count >= required]

    # With 95% threshold, frequent_elements will be small
    print(f"{len(frequent_elements)} elements appear in ≥{required} sets")

    if not frequent_elements:
        return None

    # Sort by frequency
    frequent_elements.sort(key=lambda e: -element_counts[e])

    # For 95% threshold, ALL elements in the result MUST be in ALL sets
    # that contain the result. So we can optimize further:

    print('Find sets that contain all frequent elements')
    universal_sets = []
    for i, s in enumerate(list_of_sets):
        if all(elem in s for elem in frequent_elements):
            universal_sets.append(i)

    # If not enough sets contain all elements, we need to find subsets
    if len(universal_sets) < required:
        # We need to find a subset of frequent_elements
        # that appears in enough sets

        # Parallel exploration of subsets
        def explore_subsets(subset_indices, local_seed):
            local_rng = random.Random(local_seed)
            indices = list(subset_indices)
            local_rng.shuffle(indices)

            best_local = set()
            # Try different subset sizes
            for size in range(len(indices), 0, -1):
                # Generate combinations
                from itertools import combinations
                for combo in combinations(indices, size):
                    _candidate = {frequent_elements[_i] for _i in combo}

                    # Quick check
                    count = sum(1 for _s in list_of_sets if _candidate.issubset(_s))
                    if count >= required:
                        if len(_candidate) > len(best_local):
                            best_local = _candidate.copy()
                        break  # Found valid set of this size

            return best_local

        # Split work among threads
        all_indices = list(range(len(frequent_elements)))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            chunk_size = max(1, len(all_indices) // num_threads)

            for i in range(num_threads):
                start = i * chunk_size
                end = start + chunk_size if i < num_threads - 1 else len(all_indices)
                chunk = all_indices[start:end]

                if chunk:  # Only submit if chunk is not empty
                    future = executor.submit(explore_subsets, chunk,
                                             (seed or 0) + i)
                    futures.append(future)

            # Collect results
            best_set = set()
            for future in as_completed(futures):
                candidate = future.result()
                if len(candidate) > len(best_set):
                    best_set = candidate

        return best_set if best_set else None
    else:
        return set(frequent_elements)


def segment_bright_objects_init(img_2d):
    # Use edge enhancing filter as an initial step to segment bright objects
    img_mag_farid = farid(img_2d)
    # noinspection PyTypeChecker
    markers = np.zeros_like(img_mag_farid).astype(np.uint8)
    markers[img_2d < 900] = 1
    # noinspection PyTypeChecker
    markers[img_mag_farid > 120 / np.iinfo(img_2d.dtype).max] = 2
    img_seg = watershed(img_mag_farid, markers)
    # In most cases the following two lines are enough to segment out bright objects. However, because gradient-based
    # methods detect also edges of dark objects in the neighborhood of bright objects, 'fill hole' operation will
    # probably get part of dark objects in the returned segmentations. This is why this method is only the initial step.
    img_seg = ndi.binary_fill_holes(img_seg - 1)
    img_seg = ndi.binary_opening(img_seg, disk(2))
    return img_seg


def inpaint_with_kdtree(img, mask):
    img_masked = masked_array(img, mask)
    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    data = np.array((x[~img_masked.mask], y[~img_masked.mask])).T
    data_to_replace = np.array((x[img_masked.mask], y[img_masked.mask])).T
    img_restored = img.copy()
    img_restored[img_masked.mask] = img_restored[~img_masked.mask][KDTree(data).query(data_to_replace)[1]]
    return img_restored


def get_nonzero_mask_from_image(img):
    mask = np.zeros_like(img, dtype=bool)
    mask[img > 0] = True
    mask = ndi.binary_fill_holes(mask)
    return mask


def get_largest_component(mask: np.ndarray):
    if mask.dtype == np.bool_:
        seg_labeled, seg_num = ndi.label(mask)
    else:
        seg_labeled = mask
        seg_num = len(np.unique(mask[mask > 0]))
    size_arr = np.zeros(seg_num)
    for i in range(seg_num):
        size_arr[i] = np.sum(seg_labeled == i + 1)
    return seg_labeled == np.argmax(size_arr) + 1


def get_bbx_from_roi(img, mask=None):
    """
    Note: It uses the img_mov_reg to calculate the minimum bound rectangle to cut both the original template image and
          the registered moving image
    Args:
        img: input image.

    :return:
        [x, y, w, h]: image ROI specified in the rectangular format
        c
    """
    if mask is None:
        mask = get_nonzero_mask_from_image(img)
        # Make sure that there is only one component in the mask
        mask = get_largest_component(mask)
    # findContours operates on an uint8 image
    mask_image = 255 * mask
    mask_image = mask_image.astype(np.uint8)
    cnt, _ = cv.findContours(mask_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = cnt[0]
    x, y, w, h = cv.boundingRect(contour)
    return [x, y, w, h], contour


def find_bones_2d(img_2d: np.ndarray, verbose: bool = False):
    threshold = threshold_multiotsu(img_2d)
    # noinspection PyTypeChecker
    img_seg = np.digitize(img_2d, bins=threshold)
    img_seg[img_seg == 1] = 0
    img_seg[img_seg == 2] = 1
    img_seg = ndi.binary_closing(ndi.binary_dilation(img_seg, disk(5)), disk(5))
    img_seg = ndi.binary_fill_holes(img_seg)
    img_seg = remove_small_objects(img_seg, min_size=500)
    seg_labeled, num_obj = ndi.label(img_seg)
    if verbose:
        print('Find {} large objects'.format(num_obj))
        for i in range(num_obj):
            seg = seg_labeled == i + 1
            rect, _ = get_bbx_from_roi(img_seg, seg)
            print('Bounding box rectangle: {}'.format(rect))
    return seg_labeled


# noinspection SpellCheckingInspection
def get_image_rotation_by_z(img_3d: np.ndarray, return_angle: bool = False, return_seg: bool = False,
                            verbose: bool = False):
    img_ax = np.max(img_3d, axis=2)
    mask = segment_bright_objects_init(img_ax)
    img_ax_re = inpaint_with_kdtree(img_ax, mask)
    img_ax_bone = find_bones_2d(img_ax_re)
    seg_props = regionprops(img_ax_bone)
    idx_torso = [idx for idx, item in enumerate(seg_props) if item.area == max([i.area for i in seg_props])][0]
    theta_rad = seg_props[idx_torso].orientation  # angle in radians
    theta_deg = np.rad2deg(theta_rad)
    if verbose:
        print('The image is rotated by {} degrees'.format(theta_deg))
    if theta_rad < 0:
        theta_deg = - 90 - theta_deg
        r = Rotation.from_euler('z', - math.pi / 2 - theta_rad)
    else:
        r = Rotation.from_euler('z', theta_rad)
    rmat = r.as_matrix()
    if return_angle:
        return rmat, theta_deg
    else:
        return rmat


def rotate_image_by_z(img_3d: np.ndarray, angle: float):
    return ndi.rotate(img_3d, angle, order=1, cval=np.min(img_3d))



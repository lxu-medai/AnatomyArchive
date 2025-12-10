import os
import re
import util
import pydicom
import numpy as np
import nibabel as nib
# noinspection PyProtectedMember
from shutil import ReadError
from typing import Union, List, Set, Callable


class DicomDirectoryGraph(util.DirectoryGraph):
    def __init__(self, root_path: str):
        super().__init__(root_path)

    # noinspection PyMethodMayBeStatic
    def create_dicom_filter(
            self,
            check_extensions: bool = True,
            extensions: Set[str] = None,
            validate_extensionless: bool = True
    ) -> Callable[[str], bool]:
        """
        Create a filter function for DICOM files.

        Args:
            check_extensions: If True, accept files with DICOM extensions without validation.
                             If False, only validate files without extensions.
            extensions: Set of DICOM file extensions to accept when check_extensions=True
            validate_extensionless: Whether to validate files without extensions using pydicom

        Returns:
            A function that can be used as a file_filter in build_graph
        """

        if extensions is None:
            extensions = {'.dcm', '.dicom'}

        def dicom_filter(file_path: str) -> bool:
            # Get filename
            filename = os.path.basename(file_path)

            # Get file extension
            f_name, ext = os.path.splitext(filename)
            has_extension = bool(ext)  # Check if file has any extension

            # Case 1: File has an extension
            if has_extension:
                if check_extensions:
                    # If we're checking extensions and this is a DICOM extension, accept it
                    if ext.lower() in extensions:
                        return True
                    else:
                        # Not a DICOM extension, don't accept
                        return False
                else:
                    # We're not checking extensions, so skip files with extensions
                    return False

            # Case 2: File has no extension
            else:
                if f_name[0] == '.':
                    return False
                if validate_extensionless:
                    # Validate extensionless files using pydicom
                    try:
                        metafile = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
                        if hasattr(metafile, 'PatientID'):
                            return True
                        else:
                            return False
                    except ReadError:
                        return False
                else:
                    # Not validating extensionless files
                    return False

        return dicom_filter


def get_dicom_leaf_folders(dir_root: str):
    dg = DicomDirectoryGraph(dir_root)
    file_filter = dg.create_dicom_filter()
    _ = dg.build_graph(file_filter=file_filter)
    dirs_dicom = dg.get_leaf_directories()
    if len(dirs_dicom) == 0:
        util.print_highlighted_text(f'No dicom files were detected in any sub-folders of {dir_root}')
        dirs_dicom = -1
    elif len(dirs_dicom) == 1:
        dirs_dicom = dirs_dicom[0]
    else:
        util.print_highlighted_text(f'Multi-scans were detected in sub-folders of {dir_root}')
    return dirs_dicom


def get_axcode_for_orientation(scan_direction: str):
    if scan_direction.lower() not in ['coronal', 'sagittal']:
        if scan_direction.lower() not in ['axial', 'transverse']:
            raise ValueError(f"Scan direction must be one of: 'axial/transverse', 'coronal', 'sagittal'. "
                             f"Got: {scan_direction}")
        else:
            if scan_direction.lower() == 'axial':
                scan_direction = 'transverse'
    """
    Returns all clinically valid orthogonal axcode for a given scan direction.
    """

    orientation_axes = {
        'transverse': {
            'slice_axes': ['S', 'I'],  # Slice direction must be Superior-Inferior
            'in_plane_pairs': [('R', 'A'), ('R', 'P'), ('L', 'A'), ('L', 'P')]
        },
        'coronal': {
            'slice_axes': ['A', 'P'],  # Slice direction must be Anterior-Posterior
            'in_plane_pairs': [('R', 'S'), ('R', 'I'), ('L', 'S'), ('L', 'I')]
        },
        'sagittal': {
            'slice_axes': ['R', 'L'],  # Slice direction must be Left-Right
            'in_plane_pairs': [('A', 'S'), ('A', 'I'), ('P', 'S'), ('P', 'I')]
        }
    }

    config = orientation_axes[scan_direction.lower()]
    lst_axcode = []
    for slice_axis in config['slice_axes']:
        for row_axis, col_axis in config['in_plane_pairs']:
            lst_axcode.append((row_axis, col_axis, slice_axis))

    return lst_axcode


# noinspection SpellCheckingInspection
def detect_dicom_dir(dicom_path: str, dicom_dir_str: Union[str, None] = None, by_name: bool = True):
    if by_name:
        if dicom_dir_str is None:
            dicom_dir_str = 'DICOMDIR'
        dicom_dir_file = os.path.join(dicom_path, dicom_dir_str)
        if os.path.isfile(dicom_dir_file):
            dicom_dir = pydicom.dcmread(dicom_dir_file, force=True, stop_before_pixels=True)
        else:
            dicom_dir = -1
    else:
        # As "by_name" method is set as default, in most cases, this branch is not activated.
        dicom_dir = -1
        for root, dirs, file_names in os.walk(dicom_path):
            for file in file_names:
                try:
                    _dcm = pydicom.dcmread(os.path.join(root, file), force=True, stop_before_pixels=True)
                except pydicom.errors.InvalidDicomError:
                    util.print_highlighted_text(f'Non-DICOM file detected with file name of {os.path.join(root, file)}')
                else:
                    if hasattr(_dcm, 'DirectoryRecordSequence'):
                        dicom_dir = _dcm
                        break
    return dicom_dir


def get_slice_normal(dicom_file: pydicom.dataset.FileDataset):
    r_cos, c_cos = (np.array(dicom_file.ImageOrientationPatient[:3]),
                    np.array(dicom_file.ImageOrientationPatient[3:]))
    slice_normal = np.cross(r_cos, c_cos)
    slice_normal /= np.linalg.norm(slice_normal)  # Unit vector
    return slice_normal


def get_axcode_from_dicom(dicom_file: pydicom.dataset.FileDataset):
    return nib.aff2axcodes(get_affine_from_dicom(dicom_file))


def get_affine_from_dicom(slices: Union[pydicom.dataset.FileDataset, List[pydicom.dataset.FileDataset]],
                          presorted: bool = False):
    def _get_pseudo_affine_from_dicom(_dcm):
        r_cos, c_cos = (np.array(_dcm.ImageOrientationPatient[:3]),
                        np.array(_dcm.ImageOrientationPatient[3:]))
        affine_pseudo = np.eye(4)
        affine_pseudo[:3, 0] = r_cos
        affine_pseudo[:3, 1] = c_cos
        affine_pseudo[:3, 2] = np.cross(r_cos, c_cos)
        return affine_pseudo

    if not isinstance(slices, list):
        # Single slice
        affine = _get_pseudo_affine_from_dicom(slices)
    else:
        slices_sorted = sort_slices_by_position(slices) if not presorted else slices
        dcm0 = slices_sorted[0]
        affine = _get_pseudo_affine_from_dicom(dcm0)
        if len(slices_sorted) > 1:
            sx, sy = [float(x) for x in dcm0.PixelSpacing]
            # It does not consider oblique scans
            sz = get_slice_thickness(slices_sorted)
            t = np.array(dcm0.ImagePositionPatient, dtype=float)
            # affine_lps = np.eye()
            affine[:3, 0] *= sx  # minus sign for X
            affine[:3, 1] *= sy  # minus sign for Y
            affine[:3, 2] *= sz
            affine[:3, 3] = t
    return affine


def get_scan_direction(dicom_slice: pydicom.dataset.FileDataset, return_str: bool = False):
    img_orientation = np.array(dicom_slice.ImageOrientationPatient)
    plane = np.cross(img_orientation[:3], img_orientation[3:])
    flag = np.argwhere(np.abs(plane))[0][0]
    if plane[flag] == -1:
        reverse = True
    else:
        reverse = False
    if return_str:
        dict_flag = {0: 'Coronal', 1: 'Sagittal', 2: 'Transverse'}
        flag = dict_flag[flag]
    return flag, reverse


def sort_slices_by_position(slices: List[pydicom.dataset.FileDataset], return_direction: bool = False, 
                            return_index: bool = False):
    scan_direction, reverse = get_scan_direction(slices[0])
    slices_sorted = sorted(slices, key=lambda _s: _s.ImagePositionPatient[scan_direction], reverse=reverse)
    if (not return_direction) and (not return_index):
        return slices_sorted
    else:
        if return_index:
            indices = [slices_sorted.index(_t) for _t in slices]
            if (not return_direction) and return_index:
                return slices_sorted, indices
            else:
                return slices_sorted, scan_direction, indices
        else:
            return slices_sorted, scan_direction


def get_slice_thickness(slices: List[pydicom.dataset.FileDataset], atol=0.001, presorted: bool = False):
    slices_sorted = sort_slices_by_position(slices) if not presorted else slices
    positions = np.array([_s.ImagePositionPatient for _s in slices_sorted])
    slice_normal = get_slice_normal(slices_sorted[0])
    spacings = np.dot(positions[1:] - positions[:-1], slice_normal)
    if len(set(spacings)) != 1:
        thickness_unq, thickness_idx, thickness_cnt = np.unique(spacings, return_inverse=True, return_counts=True)
        thickness_main = thickness_unq[np.argmax(thickness_cnt)]
        # Find the index of unique thickness value within tolerance limit of the most frequently occurring one.
        # The np.where() returns a tuple of index array.
        thickness_idx_tol = np.where(np.isclose(thickness_unq, thickness_main, atol=atol))[0]
        if len(thickness_idx_tol) != 0:
            # Remove the index of most frequently occurring thickness itself from the found index array
            thickness_idx_tol = np.delete(thickness_idx_tol, np.where(thickness_idx_tol == np.argmax(thickness_cnt)))
            for i in thickness_idx_tol:
                idx_to_replace = spacings == thickness_unq[i]
                spacings[idx_to_replace] = thickness_main
        if np.max(np.unique(spacings)) / thickness_main >= 2:
            print('There are large jumps during scanning...')
    else:
        thickness_main = spacings[0]
    return thickness_main


def check_and_split_slices_by_scan_settings(slices: List[pydicom.dataset.FileDataset], return_index: bool = False):
    orient_list = [tuple(_s.ImageOrientationPatient) for _s in slices]
    orient_arr = np.empty(len(orient_list), dtype=object)
    orient_arr[:] = orient_list
    orient_unq, orient_idx = np.unique(orient_arr, return_inverse=True)
    if len(orient_unq) > 1:
        flag = 1
        list_slices = util.initiate_nested_list(len(orient_unq))
        if return_index:
            list_indices = util.initiate_nested_list(len(orient_unq))
        for i in range(len(orient_unq)):
            _idx_orient_unq = np.argwhere(orient_idx == i).flatten()
            _slices = [slices[_e] for _e in _idx_orient_unq]
            if len(_slices) > 1:
                if not return_index:
                    list_slices[i] = sort_slices_by_position(_slices)
                else:
                    _slices_sorted, _indices = sort_slices_by_position(_slices, return_index=True)
                    list_slices[i] = _slices_sorted
                    # noinspection PyUnboundLocalVariable
                    list_indices[i] = _indices
            else:
                # _indices to be added for return
                list_slices[i] = _slices
        if not return_index:
            return list_slices, flag
        else:
            return list_slices, flag, list_indices
    else:
        if not return_index:
            slices, scan_dir = sort_slices_by_position(slices, return_direction=True)
        else:
            slices, scan_dir, indices = sort_slices_by_position(slices, return_direction=True, return_index=True)
        pos_arr = np.array([_s.ImagePositionPatient for _s in slices])
        pos_arr_ref = pos_arr[:, np.roll(np.arange(3), 1)[scan_dir]]
        pos_unq, pos_idx = np.unique(pos_arr_ref, return_inverse=True)
        if len(pos_unq) > 1:
            flag = 2
            list_slices = util.initiate_nested_list(len(pos_unq))
            if return_index:
                list_indices = util.initiate_nested_list(len(pos_unq))
            for i in range(len(pos_unq)):
                _idx_pos_unq = np.argwhere(pos_idx == i).flatten()
                list_slices[i] = [slices[_i] for _i in _idx_pos_unq]
                if return_index:
                    # noinspection PyUnboundLocalVariable
                    list_indices[i] = [indices[_i] for _i in _idx_pos_unq]
            if not return_index:
                return list_slices, flag
            else:
                return list_slices, flag, list_indices
        else:
            flag = 0
            if not return_index:
                return slices, flag
            else:
                # noinspection PyUnboundLocalVariable
                return slices, flag, indices


def append_dcm_data_to_nested_dict(series_dict: util.NestedDict, study_id: str, series_descr: str,
                                   has_multi_levels: bool, has_multi_exams: bool, has_multi_series: bool,
                                   content_to_append: Union[str, pydicom.dataset.FileDataset],
                                   axc: str, key_str: str = 'Data'):

    def _append_dcm_to_nested_dict(_dict):
        if key_str not in _dict.keys():
            _dict[key_str] = dict()
        if key_str != 'Data':
            if axc not in _dict[key_str].keys():
                _dict[key_str][axc] = content_to_append
            else:
                if isinstance(_dict[key_str][axc], (pydicom.dataset.FileDataset, str)):
                    # noinspection SpellCheckingInspection
                    util.print_highlighted_text(f'Multiple scout images of the same orientation with axcodes '
                                                f'corresponding to {axc} are provided!')
                    _dict[key_str][axc] = [_dict[key_str][axc]]
                else:
                    _dict[key_str][axc].append(content_to_append)
        else:
            if axc not in _dict[key_str].keys():
                _dict[key_str][axc] = list()
            _dict[key_str][axc].append(content_to_append)

    if not has_multi_levels:
        _append_dcm_to_nested_dict(series_dict)
    else:
        if has_multi_exams:
            if not has_multi_series:
                if study_id not in series_dict.keys():
                    _dct = dict()
                    series_dict[study_id] = _dct
                else:
                    _dct = series_dict[study_id]
            else:
                if study_id not in series_dict.keys() or series_descr not in series_dict[study_id].keys():
                    _dct = dict()
                    series_dict[study_id][series_descr] = _dct
                else:
                    _dct = series_dict[study_id][series_descr]
        else:
            if series_descr not in series_dict.keys():
                _dct = dict()
                series_dict[series_descr] = _dct
            else:
                _dct = series_dict[series_descr]
        _append_dcm_to_nested_dict(_dct)
        

def convert_dcm_in_dict_to_nifti(series_dict: util.NestedDict, save_nii_fpath: str, return_dicom_slices: bool = False):
    from dicom2nifti.convert_generic import dicom_to_nifti

    def _convert_slices_to_nifti_image(_series_descr: Union[str, None] = None):

        for _ort in _data_dict.keys():
            _slices = _data_dict[_ort]
            if isinstance(_slices[0], str):
                _slices = [pydicom.dcmread(_s) for _s in _slices]
            _slices_sorted = sort_slices_by_position(_slices)
            if save_nii_fpath is not None:
                _series_descr = f'{patient_id}_{_series_descr}_{_ort}' if isinstance(_series_descr, str) else\
                    f'{patient_id}_{_ort}'
                _nii_filename = os.path.join(save_nii_fpath, f'{_series_descr}.nii.gz')
                try:
                    _ = dicom_to_nifti(_slices_sorted, _nii_filename)['NII']
                except Exception:
                    util.print_highlighted_text(f"Error in converting slices of axcode {_ort} to NifTi image for series"
                                                f" {_series_descr}")
                else:
                    if not return_dicom_slices:
                        _data_dict[_ort] = _nii_filename

    patient_id = series_dict.get('PatientID')
    studies = [_k for _k in series_dict.keys() if _k != 'PatientID']
    for _sid in studies:
        _exam_dict = series_dict[_sid]
        if _sid != 'Data':
            if 'Data' in _exam_dict.keys():
                _data_dict = _exam_dict['Data']
                _convert_slices_to_nifti_image(_sid.replace('.', 'p'))
            else:
                for _k in _exam_dict.keys():
                    if 'Data' in _exam_dict[_k].keys():
                        _data_dict = _exam_dict[_k]['Data']
                        _convert_slices_to_nifti_image(f"{_sid}_{_k.replace('.', 'p')}")
        else:
            _data_dict = _exam_dict
            _convert_slices_to_nifti_image()
            

# noinspection SpellCheckingInspection
# noinspection PyUnresolvedReferences
def split_slices_w_dcm_dir(dicom_dir: pydicom.dicomdir.DicomDir, dicom_path: str, return_dicom_slices: bool = True,
                           select_scan_axis: [str, None] = None, save_nii_fpath: [str, None] = None) -> util.NestedDict:
    """

    :param dicom_dir: DICOMDIR file
    :param dicom_path: root path to DICOMDIR file
    :param return_dicom_slices: bool on whether to return dicom slices, otherwise filenames will be returned in
                                series_dict dict.
    :param select_scan_axis:
    :param save_nii_fpath: file path to save nifti images.
    :return:
    """
    series_dict = util.NestedDict()
    axcode_filter = get_axcode_for_orientation(select_scan_axis) if select_scan_axis is not None else None
    # Currently it assumes that different records come from the same patient. If dicom data from multiple patients are
    # covered in the 'dicom_dir' file, the following function will not work. This choice was made because when this
    # function was written, all 'dicom_dir' files were automatically generated per person by the PACKS system.

    if len(dicom_dir.patient_records) != 1:
        raise ValueError(f'Function split_slices_w_dcm_dir() is meant to process one patient at a time, as it is '
                         f'based on per-patient structured DICOMDIR file.')
    else:
        patient_id = dicom_dir.patient_records[0].PatientID
    series_dict['PatientID'] = patient_id
    has_multi_exams = len(dicom_dir.patient_records[0].children) != 1
    has_multi_series = any([len(_e.children) != 1 for _e in dicom_dir.patient_records[0].children])
    has_multi_levels = has_multi_exams or has_multi_series
    for _exam in dicom_dir.patient_records[0].children:
        study_id = f'{_exam.StudyDate}_{_exam.StudyID}'
        for _series in _exam.children:
            if hasattr(_series, 'SeriesDescription') and _series.SeriesDescription != '':
                series_descr = util.concatenate_substrings(_series.SeriesDescription)
            else:
                series_descr = f'{study_id}_{_series.SeriesNumber}' if not has_multi_exams else\
                    f'{_series.SeriesNumber}'
            skip_counts = 0
            for _dcm_meta in _series.children:
                _file = os.path.join(dicom_path, os.sep.join(_dcm_meta.ReferencedFileID))
                _dcm = pydicom.dcmread(_file)
                if not hasattr(_dcm, 'SliceLocation'):
                    skip_counts += 1
                    continue
                _axc0 = get_axcode_from_dicom(_dcm)
                # Reconstruct axcode as string from _axc0 by removing ',' from tuple.
                _axc = ''.join([e for e in _axc0])
                if axcode_filter is not None and _axc0 not in axcode_filter:
                    continue
                else:
                    _content = _dcm if return_dicom_slices else _file
                    if 'scout' in _dcm.ScanOptions.lower() or 'local' in _dcm.ImageType[2].lower():
                        append_dcm_data_to_nested_dict(series_dict, study_id, series_descr, has_multi_levels,
                                                       has_multi_exams, has_multi_series, _content, _axc,
                                                       key_str='Scout')
                    else:
                        append_dcm_data_to_nested_dict(series_dict, study_id, series_descr, has_multi_levels,
                                                       has_multi_exams, has_multi_series, _content, _axc)
            if skip_counts != 0:
                print('{} invalid slices skipped due to unspecified locations for patient {} of series {} at exam '
                      '{}.'.format(skip_counts, patient_id, series_descr, study_id))

    if save_nii_fpath is not None:
        convert_dcm_in_dict_to_nifti(series_dict, save_nii_fpath, return_dicom_slices)
    return series_dict


def split_slices_wo_dcm_dir(dir_patient: str, return_dicom_slices: bool = True, select_scan_axis: [str, None] = None,
                            save_nii_fpath: [str, None] = None):
    def _append_slices():
        nonlocal patient_id

        skip_counts = 0
        for _file in _files:
            try:
                _slice = pydicom.dcmread(_file)
            except ReadError:
                continue
            if not hasattr(_slice, 'SliceLocation'):
                skip_counts += 1
                continue
            if patient_id is None:
                patient_id = _slice.PatientID
            if patient_id not in result.keys():
                result['PatientID'] = patient_id
            _axc0 = get_axcode_from_dicom(_slice)
            # Reconstruct axcode as string from _axc0 by removing ',' from tuple.
            _axc = ''.join([e for e in _axc0])
            if axcode_filter is not None and _axc0 not in axcode_filter:
                continue
            else:
                _content = _slice if return_dicom_slices else _file
                if 'scout' in _slice.ScanOptions.lower() or 'local' in _slice.ImageType[2].lower():
                    append_dcm_data_to_nested_dict(result, study_id, series_descr, has_multi_levels,
                                                   has_multi_exams, has_multi_series, _content, _axc,
                                                   key_str='Scout')
                else:
                    append_dcm_data_to_nested_dict(result, study_id, series_descr, has_multi_levels,
                                                   has_multi_exams, has_multi_series, _content, _axc)
        if skip_counts != 0:
            print('{} invalid slices skipped due to unspecified locations for patient {} of series {} at exam '
                  '{}.'.format(skip_counts, patient_id, series_descr, study_id))

    dirs_dicom = get_dicom_leaf_folders(dir_patient)
    axcode_filter = get_axcode_for_orientation(select_scan_axis) if select_scan_axis is not None else None
    if isinstance(dirs_dicom, int):
        return dirs_dicom
    else:
        result = util.NestedDict()
        if isinstance(dirs_dicom, list):
            lst_ids = list()
            set_studies = set()
            lst_studies = list()
            set_series = set()
            lst_series = list()
            lst_dcm_files = list()
            for _dir_dcm in dirs_dicom:
                _files = util.get_files_in_folder(_dir_dcm)
                lst_dcm_files.append(_files)
                _dcm = pydicom.dcmread(_files[0], force=True, stop_before_pixels=True)
                _patient_id = _dcm.PatientID
                if _patient_id not in lst_ids:
                    lst_ids.append(_patient_id)
                if len(lst_ids) > 1:
                    raise ValueError(f'Function split_slices_wo_dcm_dir() is meant to process one patient at a time, '
                                     f'in order to be consistent with function split_slices_w_dcm_dir(), though the'
                                     f' internally used function get_dicom_leaf_folders() can work with folders that '
                                     f'differ in patient IDs.')
                study_id = f'{_dcm.StudyDate}_{_dcm.StudyID}'
                lst_studies.append(study_id)
                if study_id not in lst_studies:
                    set_studies.add(study_id)
                if hasattr(_dcm, 'SeriesDescription') and _dcm.SeriesDescription != '':
                    series_descr = util.concatenate_substrings(_dcm.SeriesDescription)
                else:
                    series_descr = f'{study_id}_{_dcm.SeriesNumber}' if hasattr(_dcm, 'SeriesNumber') else study_id
                lst_series.append(series_descr)
                if series_descr not in lst_series:
                    set_series.add(series_descr)
            patient_id = lst_ids[0]
            has_multi_exams = len(set_studies) > 1
            has_multi_series = len(set_series) > 1
            has_multi_levels = has_multi_exams or has_multi_series
            for i, _files in enumerate(lst_dcm_files):
                study_id = lst_studies[i]
                series_descr = lst_series[i]
                _append_slices()
        else:
            _files = util.get_files_in_folder(dirs_dicom)
            patient_id = None
            study_id = ''
            series_descr = ''
            has_multi_levels = False
            has_multi_exams = False
            has_multi_series = False
            _append_slices()
        if save_nii_fpath is not None:
            convert_dcm_in_dict_to_nifti(result, save_nii_fpath, return_dicom_slices)
        return result
        

def get_slices_from_file_path(dicom_path, return_dicom_slices: bool = True, select_scan_axis: [str, None] = None,
                              save_nii_fpath: [str, None] = None):
    dicom_dir = detect_dicom_dir(dicom_path)
    if hasattr(dicom_dir, 'DirectoryRecordSequence'):
        result = split_slices_w_dcm_dir(dicom_dir, dicom_path, return_dicom_slices, select_scan_axis, save_nii_fpath)
    else:
        result = split_slices_wo_dcm_dir(dicom_path, return_dicom_slices, select_scan_axis, save_nii_fpath)
    return result
                                  



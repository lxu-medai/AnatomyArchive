import os
import re
import util
import pydicom
import numpy as np
import nibabel as nib
from typing import Union, List

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


def get_slices_from_file_path(dicom_path, return_dicom_slices: bool = True, select_scan_axis: [str, None] = None,
                              save_nii_fpath: [str, None] = None):
    dicom_dir = detect_dicom_dir(dicom_path)
    # noinspection PyUnresolvedReferences
    if isinstance(dicom_dir, pydicom.dicomdir.DicomDir):
        result = split_slices_by_orientations_with_dcm_dir(dicom_dir, dicom_path, return_dicom_slices,
                                                           select_scan_axis, save_nii_fpath)
        patient_id = result['PatientID']
    else:
        patient_id = None
        result = []
        skip_counts = 0
        axcode_filter = get_axcode_for_orientation(select_scan_axis) if select_scan_axis is not None else None
        for root, dirs, file_names in os.walk(dicom_path):
            for file in file_names:
                file_full = os.path.join(root, file)
                try:
                    _dcm = pydicom.dcmread(file_full)
                except pydicom.errors.InvalidDicomError:
                    util.print_highlighted_text(f'Non-DICOM file detected with file name of {file_full}.')
                else:
                    if not hasattr(_dcm, 'SliceLocation'):
                        skip_counts += 1
                        continue
                    if patient_id is None:
                        patient_id = _dcm.PatientID
                    else:
                        if patient_id != _dcm.PatientID:
                            util.print_highlighted_text(f'Inconsistent patient IDs detected in the DICOM files!')
                            break
                    _axc = get_axcode_from_dicom(_dcm)
                    if axcode_filter is not None and _axc not in axcode_filter:
                        continue
                    else:
                        # Skip scout images
                        if 'scout' in _dcm.ScanOptions.lower() or 'local' in _dcm.ImageType[2].lower():
                            continue
                    result.append(_dcm if return_dicom_slices else file_full)
        if skip_counts != 0:
            print('{} invalid slices skipped due to unspecified locations for patient {}.'.format(skip_counts,
                                                                                                  patient_id))
    return result, patient_id


# noinspection SpellCheckingInspection
def detect_dicom_dir(dicom_path: str, dicom_dir_str: Union[str, None] = None, by_name: bool = True):
    if by_name:
        if dicom_dir_str is None:
            dicom_dir_str = 'DICOMDIR'
        dicom_dir_file = os.path.join(dicom_path, dicom_dir_str)
        if os.path.isfile(dicom_dir_file):
            dicom_dir = pydicom.dcmread(dicom_dir_file)
        else:
            dicom_dir = -1
    else:
        dicom_dir = -1
        for root, dirs, file_names in os.walk(dicom_path):
            for file in file_names:
                try:
                    _dcm = pydicom.dcmread(os.path.join(root, file))
                except pydicom.errors.InvalidDicomError:
                    util.print_highlighted_text(f'Non-DICOM file detected with file name of {os.path.join(root, file)}')
                else:
                    # noinspection PyUnresolvedReferences
                    if isinstance(_dcm, pydicom.dicomdir.DicomDir):
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


def sort_slices_by_position(slices: List[pydicom.dataset.FileDataset], return_direction: bool = False, return_index: bool = False):
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


# noinspection PyUnresolvedReferences
def split_slices_by_orientations_with_dcm_dir(dicom_dir: pydicom.dicomdir.DicomDir, dicom_path: str,
                                              return_dicom_slices: bool = True, select_scan_axis: [str, None] = None,
                                              save_nii_fpath: [str, None] = None):
    if save_nii_fpath is not None:
        from dicom2nifti.convert_generic import dicom_to_nifti

    def _append_dcm_to_nested_dict(_dict, _content, key_str='Data'):
        if key_str not in _dict.keys():
            _dict[key_str] = dict()
        if key_str != 'Data':
            if _axc not in _dict[key_str].keys():
                _dict[key_str][_axc] = _content
            else:
                if isinstance(_dict[key_str][_axc], (pydicom.dataset.FileDataset, str)):
                    # noinspection SpellCheckingInspection
                    util.print_highlighted_text(f'Multiple scout images of the same orientation with axcodes '
                                                f'corresponding to {_axc} are provided!')
                    _dict[key_str][_axc] = [_dict[key_str][_axc]]
                else:
                    _dict[key_str][_axc].append(_content)
        else:
            if _axc not in _dict[key_str].keys():
                _dict[key_str][_axc] = list()
            _dict[key_str][_axc].append(_content)

    def _append_data_to_dict(_content, key_str='Data'):
        if not has_multiple_levels:
            _append_dcm_to_nested_dict(series_dict, _content, key_str)
        else:
            if has_multiple_exams:
                if not has_multiple_series:
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
            _append_dcm_to_nested_dict(_dct, _content, key_str)

    def _convert_slices_to_nifti_image(_series_descr: Union[str, None] = None):

        for _ort in _data_dict.keys():
            _slices = _data_dict[_ort]
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

    series_dict = dict()
    axcode_filter = get_axcode_for_orientation(select_scan_axis) if select_scan_axis is not None else None
    # Currently it assumes that different records come from the same patient
    for _record in dicom_dir.patient_records:
        patient_id = _record.PatientID
        series_dict['PatientID'] = patient_id
        has_multiple_exams = len(_record.children) != 1
        for _exam in _record.children:
            study_id = f'{_exam.StudyDate}_{_exam.StudyID}'
            has_multiple_series = len(_exam.children) != 1
            for i, _series in enumerate(_exam.children):
                has_multiple_levels = has_multiple_exams or has_multiple_series
                if hasattr(_series, 'SeriesDescription'):
                    series_descr = util.concatenate_substrings(_series.SeriesDescription)
                else:
                    series_descr = f'{study_id}_{_series.SeriesNumber}' if not has_multiple_exams else\
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
                        if 'scout' in _dcm.ScanOptions.lower() or 'local' in _dcm_meta.ImageType[2].lower():
                            _append_data_to_dict(_dcm if return_dicom_slices else _file, 'Scout')
                        else:
                            _append_data_to_dict(_dcm if return_dicom_slices else _file)
                if skip_counts != 0:
                    print('{} invalid slices skipped due to unspecified locations for patient {} of series {} at exam '
                          '{}.'.format(skip_counts, patient_id, series_descr, study_id))
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

    return series_dict



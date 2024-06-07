# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

from radtract.tractdensity import tract_density, visitation_count
import nibabel as nib
import numpy as np
from dipy.tracking.streamline import transform_streamlines
from radtract.utils import load_trk_streamlines, save_trk_streamlines
from dipy.io.streamline import load_trk
import argparse
import sys
from multiprocessing import Pool


def density_filter(streamlines: nib.streamlines.array_sequence.ArraySequence, 
                   density, 
                   threshold: float = 0.05,
                   fraction: float = 1.0,
                   calculate_density: bool = False):
    """
    Filter streamlines based on the fraction of the streamline that passes voxels with a density above or equal to a threshold.
    :param streamlines: Streamlines to filter
    :param density: Density image to use for filtering
    :param threshold: Threshold for density
    :param fraction: Fraction of streamline that must pass voxels with density above or equalt to threshold to be kept (defaul is that all points must pass)
    :param calculate_density: If True, calculate density from streamlines
    :return: Filtered streamlines
    """
    
    if type(density) == str:
        density = nib.load(density)

    if type(density) is not nib.Nifti1Image:
        raise Exception('Density must be Nifti1Image!')
    
    if calculate_density:
        density = tract_density(streamlines, density)

    density_data = density.get_fdata()

    # transform streamlines to voxel space
    vox_streamlines = transform_streamlines(streamlines, np.linalg.inv(density.affine))

    print('Applying density filter')

    # loop over each streamline and check the desnity value at each point
    # if the density value is below the threshold, remove the streamline
    filtered_streamlines = []
    i = 0
    for streamline in vox_streamlines:
        count = 0
        for point in streamline:
            if density_data[int(point[0]), int(point[1]), int(point[2])] >= threshold:
                count += 1
        if count / len(streamline) >= fraction:
            filtered_streamlines.append(streamlines[i])
        i += 1

    print(str(len(filtered_streamlines)) + ' of ' + str(len(streamlines)) + ' streamlines remaining')

    return filtered_streamlines


def density_filter_cmdentry():
    """
    Command line entry point for density_filter
    """

    parser = argparse.ArgumentParser(description='Filter streamlines based on density')
    parser.add_argument('--streamlines', type=str, help='Path to the streamlines file (.trk)')
    parser.add_argument('--density', type=str, help='Path to the density image file (.nii.gz)')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for density')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of streamline that must pass voxels with density above or equal to threshold (default is 1.0)')
    parser.add_argument('--calculate_density', action='store_true', help='Calculate density from streamlines')
    parser.add_argument('--output', type=str, help='Path to the output file (.trk)')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    fib = load_trk(args.streamlines, "same", bbox_valid_check=False)
    streamlines = fib.streamlines
    density = nib.load(args.density)

    filtered_streamlines = density_filter(streamlines, density, args.threshold, args.fraction, args.calculate_density)
    save_trk_streamlines(filtered_streamlines, args.output, fib)


def visitation_count_filter(streamlines: nib.streamlines.array_sequence.ArraySequence, 
                            vcount, 
                            threshold: float,
                            fraction: float = 1.0,
                            calculate_vcount: bool = False):
    """
    Filter streamlines based on the fraction of the streamline that passes voxels with a visitation count above or equal to a threshold. 
    :param streamlines: Streamlines to filter
    :param vcount: Visitation count image to use for filtering
    :param threshold: Threshold for visitation count
    :param fraction: Fraction of streamline that must pass voxels with visitation count above or equalt to threshold to be kept (defaul is that all points must pass)
    :param calculate_vcount: If True, calculate visitation count from streamlines
    :return: Filtered streamlines
    """
    
    if type(vcount) == str:
        vcount = nib.load(vcount)

    if type(vcount) is not nib.Nifti1Image:
        raise Exception('Density must be Nifti1Image!')
    
    if calculate_vcount:
        vcount = visitation_count(streamlines, vcount)

    vcount_data = vcount.get_fdata()

    # transform streamlines to voxel space
    vox_streamlines = transform_streamlines(streamlines, np.linalg.inv(vcount.affine))

    print('Applying visitation count filter')

    # loop over each streamline and check the desnity value at each point
    # if the density value is below the threshold, remove the streamline
    filtered_streamlines = []
    i = 0
    for streamline in vox_streamlines:
        count = 0
        for point in streamline:
            if vcount_data[int(point[0]), int(point[1]), int(point[2])] >= threshold:
                count += 1
        if count / len(streamline) >= fraction:
            filtered_streamlines.append(streamlines[i])
        i += 1

    print(str(len(filtered_streamlines)) + ' of ' + str(len(streamlines)) + ' streamlines remaining')

    return filtered_streamlines


def visitation_count_filter_cmdentry():
    """
    Command line entry point for visitation_count_filter
    """

    parser = argparse.ArgumentParser(description='Filter streamlines based on visitation count')
    parser.add_argument('--streamlines', type=str, help='Path to the streamlines file (.trk)')
    parser.add_argument('--vcount', type=str, help='Path to the visitation count image file (.nii.gz)')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for visitation count')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of streamline that must pass voxels with visitation count above or equal to threshold (default is 1.0)')
    parser.add_argument('--calculate_vcount', action='store_true', help='Calculate visitation count from streamlines')
    parser.add_argument('--output', type=str, help='Path to the output file (.trk)')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    fib = load_trk(args.streamlines, "same", bbox_valid_check=False)
    streamlines = fib.streamlines
    vcount = nib.load(args.vcount)

    filtered_streamlines = visitation_count_filter(streamlines, vcount, args.threshold, args.fraction, args.calculate_vcount)
    save_trk_streamlines(filtered_streamlines, args.output, fib)


def mask_overlap_filter(streamlines: nib.streamlines.array_sequence.ArraySequence, 
                        mask, 
                        fraction: float = 1.0):
    """
    Filter streamlines based on the fraction of the streamline that overlaps with a mask. Streamlines that overlap with the mask at least the specified fraction are kept.
    :param streamlines: Streamlines to filter
    :param mask: Mask to use for filtering
    :param fraction: Fraction of streamline that must overlap with mask to be kept (defaul is that all points must pass)
    :return: Filtered streamlines
    """    
    
    if type(mask) == str:
        mask = nib.load(mask)

    if type(mask) is not nib.Nifti1Image:
        raise Exception('Mask must be Nifti1Image!')

    mask_data = mask.get_fdata()

    # transform streamlines to voxel space
    vox_streamlines = transform_streamlines(streamlines, np.linalg.inv(mask.affine))

    print('Applying mask overlap filter')

    filtered_streamlines = []
    i = 0
    for streamline in vox_streamlines:
        count = 0
        for point in streamline:
            if mask_data[int(point[0]), int(point[1]), int(point[2])] > 0:
                count += 1
        if count / len(streamline) >= fraction:
            filtered_streamlines.append(streamlines[i])      
        i += 1

    print(str(len(filtered_streamlines)) + ' of ' + str(len(streamlines)) + ' streamlines remaining')          

    return filtered_streamlines


def mask_overlap_filter_cmdentry():
    """
    Command line entry point for mask_overlap_filter
    """

    parser = argparse.ArgumentParser(description='Filter streamlines based on mask overlap')
    parser.add_argument('--streamlines', type=str, help='Path to the streamlines file (.trk)')
    parser.add_argument('--mask', type=str, help='Path to the mask image file (.nii.gz)')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of streamline that must overlap with mask to be kept (default is 1.0)')
    parser.add_argument('--output', type=str, help='Path to the output file (.trk)')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    fib = load_trk(args.streamlines, "same", bbox_valid_check=False)
    streamlines = fib.streamlines
    mask = nib.load(args.mask)

    filtered_streamlines = mask_overlap_filter(streamlines, mask, args.fraction)
    save_trk_streamlines(filtered_streamlines, args.output, fib)


def endpoint_filter(streamlines: nib.streamlines.array_sequence.ArraySequence, 
                    masks: list,
                    mode: str = 'both'):
    """
    Filter streamlines based on the endpoints of the streamlines. 
    :param streamlines: Streamlines to filter
    :param masks: List of masks to use for filtering. 
    :param mode: Mode for filtering. Options are 'both', 'any', 'one', 'both_labeldiff', 'none'.
    :return: Filtered streamlines
    """
    
    labelmap = None
    ref_image = None
    count = 0
    for m in masks:
        if type(m) == str:
            m = nib.load(m)

        if type(m) is not nib.Nifti1Image:
            raise Exception('Mask must be Nifti1Image!')
            
        ref_image = m
        m_data = m.get_fdata()

        if m_data.shape != ref_image.get_fdata().shape:
            raise Exception('All masks must have the same shape!')
        
        if labelmap is None:
            s = (m_data.shape[0], m_data.shape[1], m_data.shape[2], len(masks))
            labelmap = np.zeros(s)

        m_data[m_data > 0] = 1
        labelmap[:, :, :, count] = m_data
        
        count += 1

    labelmap_mip = np.max(labelmap, axis=3)

    # transform streamlines to voxel space
    vox_streamlines = transform_streamlines(streamlines, np.linalg.inv(ref_image.affine))

    print('Applying endpoint filter')

    filtered_streamlines = []
    i = 0
    for streamline in vox_streamlines:
        p1 = np.round(streamline[0]).astype('int64')
        p2 = np.round(streamline[-1]).astype('int64')

        if mode == 'both':
            if labelmap_mip[p1[0], p1[1], p1[2]] > 0 and labelmap_mip[p2[0], p2[1], p2[2]] > 0:
                filtered_streamlines.append(streamlines[i])
        elif mode == 'any':
            if labelmap_mip[p1[0], p1[1], p1[2]] > 0 or labelmap_mip[p2[0], p2[1], p2[2]] > 0:
                filtered_streamlines.append(streamlines[i])
        elif mode == 'one':
            if labelmap_mip[p1[0], p1[1], p1[2]] > 0 and labelmap_mip[p2[0], p2[1], p2[2]] == 0:
                filtered_streamlines.append(streamlines[i])
            elif labelmap_mip[p1[0], p1[1], p1[2]] == 0 and labelmap_mip[p2[0], p2[1], p2[2]] > 0:
                filtered_streamlines.append(streamlines[i])
        elif mode == 'both_labeldiff':

            labels_p1 = []
            labels_p2 = []
            for i in range(labelmap.shape[3]):
                if labelmap[p1[0], p1[1], p1[2], i] > 0:
                    labels_p1.append(i)
                if labelmap[p2[0], p2[1], p2[2], i] > 0:
                    labels_p2.append(i)
            if len(labels_p1) > 0 and len(labels_p2) > 0:
                if len(labels_p1) == 1 and len(labels_p2) == 1:
                    if labels_p1[0] != labels_p2[0]:
                        filtered_streamlines.append(streamlines[i])
                else:
                    filtered_streamlines.append(streamlines[i])

        elif mode == 'none':
            if labelmap_mip[p1[0], p1[1], p1[2]] == 0 and labelmap_mip[p2[0], p2[1], p2[2]] == 0:
                filtered_streamlines.append(streamlines[i])
        i += 1

    print(str(len(filtered_streamlines)) + ' of ' + str(len(streamlines)) + ' streamlines remaining')

    return filtered_streamlines


def endpoint_filter_cmdentry():
    """
    Command line entry point for endpoint_filter
    """

    parser = argparse.ArgumentParser(description='Filter streamlines based on endpoints')
    parser.add_argument('--streamlines', type=str, help='Path to the streamlines file (.trk)')
    parser.add_argument('--masks', type=str, nargs='+', help='Path to the mask image file (.nii.gz)')
    parser.add_argument('--mode', type=str, default='both', help='Mode for filtering. Options are "both", "any", "one", "both_labeldiff", "none". Default is "both".')
    parser.add_argument('--output', type=str, help='Path to the output file (.trk)')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    fib = load_trk(args.streamlines, "same", bbox_valid_check=False)
    streamlines = fib.streamlines

    filtered_streamlines = endpoint_filter(streamlines, args.masks, args.mode)
    save_trk_streamlines(filtered_streamlines, args.output, fib)


def length_filter(streamlines: nib.streamlines.array_sequence.ArraySequence,
                  min: float = 5,
                  max: float = 95,
                  absolute: bool = False):
    """
    Filter streamlines based on their length.
    :param streamlines: Streamlines to filter
    :param min: Minimum percentile of length to keep
    :param max: Maximum percentile of length to keep
    :param absolute: If True, min and max are interpreted as absolute values instead of percentiles
    :return: Filtered streamlines
    """

    print('Applying length filter')

    lengths = []
    for streamline in streamlines:
        l = 0
        for i in range(len(streamline)-1):
            l += np.linalg.norm(streamline[i+1] - streamline[i])
        lengths.append(l)
        
    lengths = np.array(lengths)
    if not absolute:
        min = np.percentile(lengths, min)
        max = np.percentile(lengths, max)

    filtered_streamlines = []
    for streamline, length in zip(streamlines, lengths):
        if length >= min and length <= max:
            filtered_streamlines.append(streamline)

    print(str(len(filtered_streamlines)) + ' of ' + str(len(streamlines)) + ' streamlines remaining')

    return filtered_streamlines


def length_filter_cmdentry():
    """
    Command line entry point for length_filter
    """

    parser = argparse.ArgumentParser(description='Filter streamlines based on length')
    parser.add_argument('--streamlines', type=str, help='Path to the streamlines file (.trk)')
    parser.add_argument('--min', type=float, default=5, help='Minimum percentile of length to keep')
    parser.add_argument('--max', type=float, default=95, help='Maximum percentile of length to keep')
    parser.add_argument('--absolute', action='store_true', help='If True, min and max are interpreted as absolute values instead of percentiles')
    parser.add_argument('--output', type=str, help='Path to the output file (.trk)')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    fib = load_trk(args.streamlines, "same", bbox_valid_check=False)
    streamlines = fib.streamlines

    filtered_streamlines = length_filter(streamlines, args.min, args.max, args.absolute)
    save_trk_streamlines(filtered_streamlines, args.output, fib)


def process_streamline(args):
    """
    Process a single streamline for curvature filtering (used for multiprocessing)
    """
    streamline = args[0]
    window = args[1]
    threshold = args[2]
    vectors = []
    lengths = []
    for i in range(len(streamline) - 1):
        p1 = streamline[i]
        p2 = streamline[i+1]

        # calculate angle in degrees
        if window is None:
            dotp = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
            if dotp > 1:
                dotp = 1
            elif dotp < -1:
                dotp = -1
            angle = abs(np.arccos(dotp) * 180 / np.pi)
            if angle > threshold:
                return None

        # calculate vector
        vector = p2 - p1

        # calculate segment length
        length = np.linalg.norm(vector)
        lengths.append(length)

        # normalize vector
        vector /= length
        vectors.append(vector)
    
    if window is None:
        return streamline

    # calculate curvature over window. windows is given in mm
    lwindow = []
    vwindow = []
    for l, v in zip(lengths, vectors):
        lwindow.append(l)
        vwindow.append(v)
        if np.sum(lwindow) >= window:
            
            # calculate mean vector
            mean_vector = np.sum(vwindow, axis=0)
            mean_vector /= np.linalg.norm(mean_vector)

            # calculate mean angular deviation from mean vector
            mean_angle = 0
            for v in vwindow:
                dotp = np.dot(mean_vector, v)
                if dotp > 1:
                    dotp = 1
                elif dotp < -1:
                    dotp = -1
                a = np.arccos(dotp) * 180 / np.pi
                mean_angle += a
            mean_angle /= len(vwindow)

            if mean_angle > threshold:
                return None

            # remove elements from front of window until window is smaller than window size
            while np.sum(lwindow) >= window:
                lwindow.pop(0)
                vwindow.pop(0)

    return streamline

def curvature_filter(streamlines: nib.streamlines.array_sequence.ArraySequence,
                     threshold = 30,
                     window = 10):
    """
    Filter streamlines based on their curvature. If dist is not None, then claculate the curvature over a window (in mm). If dist is None, then calculate the curvature between two points.
    """

    print('Applying curvature filter')
    
    filtered_streamlines = []

    with Pool() as pool:
        results = pool.map(process_streamline, zip(streamlines, [window]*len(streamlines), [threshold]*len(streamlines)))
    
    filtered_streamlines = [s for s in results if s is not None]

    print(str(len(filtered_streamlines)) + ' of ' + str(len(streamlines)) + ' streamlines remaining')

    return filtered_streamlines

def curvature_filter_cmdentry():
    """
    Command line entry point for curvature_filter
    """

    parser = argparse.ArgumentParser(description='Filter streamlines based on curvature')
    parser.add_argument('--streamlines', type=str, help='Path to the streamlines file (.trk)')
    parser.add_argument('--threshold', type=float, default=30, help='Threshold for curvature (in degrees)')
    parser.add_argument('--window', type=float, default=10, help='Window for curvature calculation (in mm)')
    parser.add_argument('--output', type=str, help='Path to the output file (.trk)')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    fib = load_trk(args.streamlines, "same", bbox_valid_check=False)
    streamlines = fib.streamlines

    filtered_streamlines = curvature_filter(streamlines, args.threshold, args.window)

    save_trk_streamlines(filtered_streamlines, args.output, fib)

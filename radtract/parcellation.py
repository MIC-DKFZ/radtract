# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

import nibabel as nib
from nibabel.affines import apply_affine
from skimage.morphology import binary_dilation, binary_closing
from sklearn.svm import SVC
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.featurespeed import ResampleFeature
import numpy as np
from dipy.tracking.streamline import transform_streamlines
from dipy.align.reslice import reslice
from scipy.spatial import cKDTree
import os
import argparse
import joblib
import multiprocessing
import sys
from radtract.tractdensity import tract_envelope
from radtract.utils import load_trk_streamlines, save_as_vtk_fib, save_trk_streamlines
from fury.colormap import distinguishable_colormap
import joblib


def estimate_num_parcels(streamlines: nib.streamlines.array_sequence.ArraySequence,
                         reference_image: nib.Nifti1Image,
                         num_voxels: int = 5):
    """
    Estimates the number of parcels when aiming for a parcel size in tract direction of about 'num_voxels' voxels.
    :param streamlines: input streamlines
    :param reference_image: us this image geometry
    :param num_voxels: desired parcel size in tract direction
    :return:
    """
    print('Estimating number of possible parcels for on average ' + str(num_voxels) + ' traversed voxels per parcel.')
    average_spacing = np.mean(reference_image.header['pixdim'][1:4])
    num_voxels_passed = 0

    for s in streamlines:
        num_points = len(s)

        s_len = 0
        for j in range(num_points-1):
            v1 = s[j]
            v2 = s[j + 1]

            d = np.empty(3)
            for i in range(3):
                d[i] = v1[i] - v2[i]
            s_len += np.linalg.norm(d)

        num_voxels_passed += s_len/average_spacing

    num_voxels_passed /= len(streamlines)
    num_parcels = int(np.ceil(num_voxels_passed / num_voxels))
    print('Number of estimated parcels ' + str(num_parcels))

    if num_parcels < 3:
        if num_voxels > 2:
            print('Tract is too short. Trying to estimate number of parcels with ' + str(num_voxels - 1) + ' voxels per parcel.')
            estimate_num_parcels(streamlines, reference_image, num_voxels - 1)
        else:
            raise Exception('Tract is too short. Resulting number of parcels each covering ' + str(num_voxels) + ' is ' + str(num_parcels))

    return num_parcels


def is_flipped(s: np.array,
               ref: np.array):
    """
    Checks if a streamline is flipped compared to a reference streamline using the minimum average direct-flip distance (Garyfallidis et al. 2012).
    :param s: input streamline
    :param ref: reference streamline
    :return: True if s is flipped relative to ref, False if not
    """
    d_direct = 0
    d_flipped = 0

    assert len(s) == len(ref), 'Streamline and reference streamline must have the same number of points.'

    num = len(s)

    for i in range(num):
        p1 = s[i]
        p2 = ref[i]

        a = p1[0] - p2[0]
        b = p1[1] - p2[1]
        c = p1[2] - p2[2]
        d_direct += a*a+b*b+c*c

        p1 = s[num-i-1]
        a = p1[0] - p2[0]
        b = p1[1] - p2[1]
        c = p1[2] - p2[2]
        d_flipped += a*a+b*b+c*c

    if d_direct < d_flipped:
        return False
    return True


def split_parcellation(parcellation: nib.Nifti1Image):
    """
    Splits a parcellation into a list of binary maps.
    :param parcellation: input parcellation
    :return: list of binary maps
    """
    data = parcellation.get_fdata().astype('uint8')
    labels = np.unique(data)
    parcels = []
    for label in labels:
        if label > 0:
            bin_map = np.zeros(data.shape, dtype='uint8')
            bin_map[np.where(data == label)] = 1
            parcel = nib.Nifti1Image(bin_map, header=parcellation.header, affine=parcellation.affine)
            parcels.append(parcel)
    return parcels


def is_inside(index, image):
    """
    Checks if a given index is inside the image.
    :param index:
    :param image:
    :return:
    """
    for i in range(3):
        if index[i] < 0 or index[i] > image.shape[i] - 1:
            return False
    return True


def resample_streamlines(streamlines: nib.streamlines.array_sequence.ArraySequence,
                         nb_points: int):
    """
    Resamples all streamlines to a given number of points.
    :param streamlines: streamlines to resample
    :param nb_points: desired number of points
    :return: resampled streamlines
    """
    streamlines_new = []
    for sl in streamlines:
        feature = ResampleFeature(nb_points=nb_points)
        streamlines_new.append(feature.extract(sl))
    return streamlines_new


def reorient_streamlines(streamlines: nib.streamlines.array_sequence.ArraySequence,
                         start_region: nib.Nifti1Image = None,
                         reference_streamline: np.array = None):
    """
    Reorients streamlines to not be flipped relative to each other.
    :param streamlines: streamlines to reorient
    :param start_region: if set the streamlines are reoriented to all start in this region
    :param reference_streamline: if start_region is not set the streamlines are reoriented to be aligned with this reference streamline
    :return:
    """

    print('Reorienting streamlines...')
    assert start_region is not None or reference_streamline is not None, 'No reorientation possible. Please provide either start and end region or reference streamline.'

    if start_region is not None:
        start_region_data = start_region.get_fdata().astype('uint8')

        streamlines_imagecoords = transform_streamlines(streamlines, np.linalg.inv(start_region.affine))

        s_idxs = []
        e_idxs = []
        for s in streamlines_imagecoords:
            start_index = s[0]
            end_index = s[-1]
            s_idxs.append(start_index)
            e_idxs.append(end_index)

        nonzero = np.nonzero(start_region_data)
        nonzero = np.array([nonzero[0], nonzero[1], nonzero[2]]).T
        dists_s, _ = cKDTree(nonzero, 1, copy_data=True).query(s_idxs, k=1)
        dists_e, _ = cKDTree(nonzero, 1, copy_data=True).query(e_idxs, k=1)

        idx = 0
        oriented_streamlines = []
        for d_s, d_e in zip(dists_s, dists_e):
            if d_s < d_e:
                oriented_streamlines.append(streamlines[idx])
            else:
                oriented_streamlines.append(np.flip(streamlines[idx], axis=0))
            idx += 1

        return oriented_streamlines
    elif reference_streamline is not None:

        streamlines_resampled = resample_streamlines(streamlines, len(reference_streamline))

        oriented_streamlines = []
        idx = 0
        for s in streamlines_resampled:
            if is_flipped(s, reference_streamline):
                oriented_streamlines.append(np.flip(streamlines[idx], axis=0))
            else:
                oriented_streamlines.append(streamlines[idx])
            idx += 1
        return oriented_streamlines


def batch_parcellate_tracts(streamlines: list,
                            binary_envelopes: list,
                            num_parcels: list,
                            start_regions: list,
                            out_parcellation_filenames: list,
                            parcellation_type: str = 'hyperplane',
                            dilate_envelope: bool = False,
                            close_envelope: bool = True,
                            postprocess: bool = False
                            ):
    """
    Convenience function to parcellate multiple tracts.
    :param streamlines:
    :param binary_envelopes:
    :param num_parcels:
    :param start_regions:
    :param out_parcellation_filenames:
    :param parcellation_type:
    :param dilate_envelope:
    :param close_envelope:
    :param postprocess:
    :return:
    """
    assert len(streamlines) == len(binary_envelopes) == len(num_parcels) == len(start_regions) == len(out_parcellation_filenames), 'All inputs must have the same length.'
    for i in range(len(streamlines)):
        parcellate_tract(streamlines=streamlines[i],
                         binary_envelope=binary_envelopes[i],
                         num_parcels=num_parcels[i],
                         start_region=start_regions[i],
                         parcellation_type=parcellation_type,
                         dilate_envelope=dilate_envelope,
                         close_envelope=close_envelope,
                         out_parcellation_filename=out_parcellation_filenames[i],
                         postprocess=postprocess)


def predict_points(argstuple):
    """
    Helper function to predict points in parallel.
    :param argstuple: tuple containing svc and X
    :return: predicted parcel labels
    """
    svc = argstuple[0]
    points = argstuple[1]
    prediction = svc.predict(points)

    return prediction.tolist(), points.tolist()


def parcellate_tract(streamlines: nib.streamlines.array_sequence.ArraySequence,
                     binary_envelope: nib.Nifti1Image = None,
                     reference_image: nib.Nifti1Image = None,
                     num_parcels: int = None,
                     parcellation_type: str = 'hyperplane',
                     start_region: nib.Nifti1Image = None,
                     reference_streamline: np.array = None,
                     dilate_envelope: bool = False,
                     close_envelope: bool = True,
                     out_parcellation_filename: str = None,
                     new_voxel_size: tuple = None,
                     postprocess: bool = False,
                     streamline_space: bool = False,
                     save_intermediate_files: bool = False):
    """
    Parcellate input streamlines into num_parcels parcels.
    :param streamlines: input streamlines in dipy format
    :param binary_envelope: binary mask defining area to parcellate, should cover input streamlines
    :param reference_image: reference image used to automatically calulate binary envelope if binary_envelope is not set. if reference_image is not set, but start_region is set, the start_region is used as reference image for the envelope calculation.
    :param num_parcels: desired number of parcels
    :param parcellation_type: 'hyperplane', 'centerline', or 'static'. 'hyperplane' is the improved approach published in the RadTract paper [REF]
    :param start_region: binary mask defining the start region of the streamlines used for reorientation. if not set, reference streamline is used for reorientation.
    :param reference_streamline: reference streamline for reorientation. only used if start_region is not set. if both are not set, the tract centerline is used as reference for reorientation.
    :param dilate_envelope: dilate binary envelope to obtain a slightly bigger parcellation. this can be useful for visualization purposes.
    :param close_envelope: close holes in binary envelope. this can be useful to close holes resulting from sparse bundles.
    :param out_parcellation_filename: if set, the parcellation is saved to this file
    :param new_voxel_size: resample binary envelope to this voxel size before parcellation
    :param postprocess: remove outliers by voting label of each voxel by label of its 26 neighbors
    :param streamline_space: if set, the parcellation is performed in streamline space instead of voxel space. The output is not a parcellation image but a dict containing the streamline points and the parcel label for each point.
    :param save_intermediate_files: if set, intermediate files, such as the binary envelope, are saved to disk

    :return: parcellation: parcellation as image containing the parcel label for each nonzero voxel of the streamline envelope
    :return: reference_streamline: reference streamline used in the parcellation process (none if start_region is set and reference_streamline is not set)
    :return: reduced_streamlines: reduced number of streamlines used for hyperplane-based parcellation (none if parcellation_type is 'centerline')
    :return: svc: support vector classifier used for hyperplane-based parcellation (none if parcellation_type is 'centerline')
    """
    out_parcellation_filename.replace('.pkl', '.nii.gz')
    if not out_parcellation_filename.endswith('.nii.gz'):
        out_parcellation_filename += '.nii.gz'

    if binary_envelope is None and start_region is None and reference_image is None:
        raise Exception('Either binary_envelope, start_region or reference_image must be set!')

    # load data if inputs are filenames
    if type(streamlines) is str and os.path.isfile(streamlines):
        streamlines = load_trk_streamlines(streamlines)
    if type(binary_envelope) is str and os.path.isfile(binary_envelope):
        binary_envelope = nib.load(binary_envelope)
    if type(reference_image) is str and os.path.isfile(reference_image):
        reference_image = nib.load(reference_image)
    if type(start_region) is str and os.path.isfile(start_region):
        start_region = nib.load(start_region)

    if binary_envelope is None:
        if type(start_region) is nib.Nifti1Image:
            print('Creating binary envelope from start region')
            binary_envelope = tract_envelope(streamlines, start_region)
        elif type(reference_image) is nib.Nifti1Image:
            print('Creating binary envelope from reference image')
            binary_envelope = tract_envelope(streamlines, reference_image)
    else:
        print('Using provided binary envelope')

    assert type(streamlines) is nib.streamlines.array_sequence.ArraySequence, 'Streamlines must be in dipy format!'
    assert type(binary_envelope) is nib.Nifti1Image, 'Binary envelope must be Nifti1Image!'
    if start_region is not None:
        assert type(start_region) is nib.Nifti1Image, 'Start region must be Nifti1Image!'
    assert parcellation_type in ['hyperplane', 'centerline', 'static'], 'Parcellation type must be hyperplane, centerline or static!'

    print('Input number of fibers:', len(streamlines))
    assert len(streamlines) > 0, 'No streamlines found!'

    if num_parcels is None or num_parcels < 1:
        num_parcels = estimate_num_parcels(streamlines=streamlines, reference_image=binary_envelope)

    feature = ResampleFeature(nb_points=num_parcels)
    metric = AveragePointwiseEuclideanMetric(feature)
    if start_region is None:
        print('Creating local reference centroid')
        qb = QuickBundles(threshold=9999., metric=metric)
        local_reference_streamline = qb.cluster(streamlines).centroids[0]
        if reference_streamline is not None and is_flipped(reference_streamline, local_reference_streamline):
            local_reference_streamline = np.flip(local_reference_streamline, axis=0)
        reference_streamline = local_reference_streamline

    oriented_streamlines = reorient_streamlines(streamlines=streamlines, start_region=start_region, reference_streamline=reference_streamline)

    envelope_data = np.copy(binary_envelope.get_fdata().astype('uint8'))
    if dilate_envelope:
        envelope_data = binary_dilation(envelope_data).astype('uint8')
    if close_envelope:
        envelope_data = binary_closing(envelope_data).astype('uint8')

    affine = binary_envelope.affine
    if new_voxel_size is not None:
        old_voxel_sizes = binary_envelope.header.get_zooms()[:3]
        envelope_data, affine = reslice(envelope_data, affine, old_voxel_sizes, new_voxel_size)

    if not streamline_space:
        nonzero = np.where(envelope_data > 0)
        envelope_world_coordinates = []
        envelope_indices = list(zip(nonzero[0], nonzero[1], nonzero[2]))
        for idx in envelope_indices:
            envelope_world_coordinates.append(apply_affine(affine, idx))
        envelope_world_coordinates = np.array(envelope_world_coordinates)

    # define outputs
    reduced_streamlines = None
    svc = None
    streamline_point_parcels = None
    colors = None

    if num_parcels > 1 and parcellation_type == 'hyperplane':
        print('Reducing input bundle')
        threshold = 20
        num_centroids = 0
        reduced_streamlines = None
        while num_centroids < 500 and threshold > 2.0:  # 500 seems to work ok
            qb = QuickBundles(threshold=threshold, metric=metric)
            reduced_streamlines = qb.cluster(oriented_streamlines).centroids
            threshold *= 0.9
            num_centroids = len(reduced_streamlines)
        print('Reduced number of fibers:', num_centroids)

        samples = []
        classes = []
        for s in reduced_streamlines:
            samples += s.tolist()
            classes += np.arange(1, len(s) + 1, 1).tolist()

        samples = np.array(samples)
        classes = np.array(classes)

        print('Fitting parcellation model to ' + str(len(classes)) + ' points')
        svc = SVC(C=1, kernel='rbf')
        svc.fit(X=samples, y=classes)

        if not streamline_space:
            print('Predicting hyperplane-parcellation for ' + str(len(envelope_world_coordinates)) + ' voxels')
            predicted_parcels = svc.predict(X=envelope_world_coordinates)

            i = 0
            for parcel in predicted_parcels:
                envelope_data[envelope_indices[i]] = parcel
                i += 1
        else:
            envelope_data = None
            streamline_point_parcels = dict()
            streamline_point_parcels['points'] = []
            streamline_point_parcels['parcels'] = []

            argslist = []
            for s in oriented_streamlines:
                argslist.append((svc, s))

            points = np.concatenate(oriented_streamlines, axis=0)
            print('Predicting hyperplane-parcellation for ' + str(points.shape[0]) + ' streamline points using ' + str(multiprocessing.cpu_count()-1) + ' cores')
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
            results = pool.map(predict_points, argslist)

            points = []
            predicted_parcels = []
            for result in results:
                predicted_parcels += result[0]
                points += result[1]

            pool.close()
            pool.join()

            streamline_point_parcels['points'] = points
            streamline_point_parcels['parcels'] = predicted_parcels

            check = np.unique(predicted_parcels)

            # if there are parcels without points, add the respective centerline point
            if check.shape[0] != num_parcels:

                qb = QuickBundles(threshold=9999., metric=metric)
                centerline = qb.cluster(oriented_streamlines).centroids[0]
                for i in range(num_parcels):
                    if i+1 not in check:
                        print('WARNING: empty parcel ' + str(i+1) + '. Adding corresponding centerline point to file ' + out_parcellation_filename)
                        streamline_point_parcels['parcels'] = np.append(streamline_point_parcels['parcels'], i+1)
                        streamline_point_parcels['points'] = np.append(streamline_point_parcels['points'], [centerline[i]], axis=0)

        print('Finished hyperplane-based parcellation')

    elif num_parcels > 1 and parcellation_type == 'centerline':
        print('Creating centerline-based parcellation')

        # create centerline
        qb = QuickBundles(threshold=9999., metric=metric)
        centerline = qb.cluster(oriented_streamlines).centroids[0]

        # find nearest centerline point for each envelope index and label accordingly
        if not streamline_space:
            _, segment_idxs = cKDTree(centerline, 1, copy_data=True).query(envelope_world_coordinates, k=1)

            # write parcellation labels to image
            for idx, label in zip(envelope_indices, segment_idxs):
                envelope_data[idx] = label + 1
        else:
            envelope_data = None
            streamline_point_parcels = dict()
            streamline_point_parcels['points'] = []
            streamline_point_parcels['parcels'] = []
            points = np.concatenate(oriented_streamlines, axis=0)
            _, segment_idxs = cKDTree(centerline, 1, copy_data=True).query(points, k=1)
            streamline_point_parcels['points'] = points
            streamline_point_parcels['parcels'] = segment_idxs + 1
            check = np.unique(segment_idxs)

            # if there are parcels without points, add the respective centerline point
            if check.shape[0] != num_parcels:
                for i in range(num_parcels):
                    if i not in check:
                        print('WARNING: empty parcel ' + str(i+1) + '. Adding corresponding centerline point to file ' + out_parcellation_filename)
                        streamline_point_parcels['parcels'] = np.append(streamline_point_parcels['parcels'], i+1)
                        streamline_point_parcels['points'] = np.append(streamline_point_parcels['points'], [centerline[i]], axis=0)

        print('Finished centerline-based parcellation')

    elif num_parcels > 1 and parcellation_type == 'static':
        streamline_space = True
        print('Creating static resampling-based parcellation')
        envelope_data = None
        oriented_streamlines = resample_streamlines(oriented_streamlines, nb_points=num_parcels)
        streamline_point_parcels = dict()
        streamline_point_parcels['points'] = []
        streamline_point_parcels['parcels'] = []
        for s in oriented_streamlines:
            streamline_point_parcels['points'] += s.tolist()
            streamline_point_parcels['parcels'] += np.arange(1, num_parcels + 1, 1).tolist()

    elif num_parcels == 1:
        print('Only 1 parcel requested, parcellation equals envelope.')
    else:
        print('Invalid parcellation type')
        envelope_data = None

    if postprocess and envelope_data is not None:
        print('Postprocessing parcellation')
        data_temp = np.zeros(envelope_data.shape).astype('uint8')
        labels = np.unique(envelope_data)
        idxs = np.where(envelope_data > 0)
        for x, y, z in zip(idxs[0], idxs[1], idxs[2]):
            neighbors = envelope_data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
            max_count = 0
            max_count_label = 0
            for label in labels:
                if label > 0:
                    label_count = len(np.where(neighbors == label)[0])
                    if label_count > max_count:
                        max_count = label_count
                        max_count_label = label
            data_temp[x, y, z] = max_count_label
        envelope_data = data_temp

    lut_cmap = distinguishable_colormap(nb_colors=num_parcels)
    if envelope_data is not None:

        assigned_parcels = np.unique(envelope_data)
        assigned_parcels = assigned_parcels[assigned_parcels > 0]
        if assigned_parcels.shape[0] != num_parcels:
            for i in range(1, num_parcels+1):
                if i not in assigned_parcels:
                    print('WARNING: empty parcel ' + str(i) + 'in file ' + out_parcellation_filename)

        parcellation = nib.Nifti1Image(envelope_data, affine=binary_envelope.affine, dtype='uint8')
        if out_parcellation_filename is not None:
            print('Saving ' + parcellation_type + '-based parcellation to ' + out_parcellation_filename)
            nib.save(parcellation, out_parcellation_filename)

        if save_intermediate_files:
            # create colors list
            colors = []
            oriented_streamlines_voxelspace = transform_streamlines(oriented_streamlines, np.linalg.inv(affine))
            for s in oriented_streamlines_voxelspace:
                num_points = len(s)
                for j in range(num_points):
                    p_cont = s[j]
                    p = np.round(p_cont).astype('int64')
                    label = envelope_data[p[0], p[1], p[2]]
                    color = lut_cmap[label-1]*255.0
                    color = np.append(color, 255.0)
                    colors.append(color)

    elif streamline_point_parcels is not None:
        parcellation = streamline_point_parcels
        joblib.dump(parcellation, out_parcellation_filename.replace('.nii.gz', '.pkl'))

        if save_intermediate_files:
            # create colors list
            colors = []
            for i in range(len(streamline_point_parcels['points'])):
                p = streamline_point_parcels['parcels'][i]
                color = lut_cmap[p-1]*255.0
                color = np.append(color, 255.0)
                colors.append(color)


    if save_intermediate_files:

        if envelope_data is not None:
            new_envelope_data = np.zeros(envelope_data.shape, dtype='uint8')
            new_envelope_data[np.where(envelope_data > 0)] = 1
            new_envelope = nib.Nifti1Image(new_envelope_data, affine=binary_envelope.affine)
            nib.save(new_envelope, out_parcellation_filename.replace('.nii.gz', '_envelope.nii.gz'))

        if reference_streamline is not None:
            save_trk_streamlines(streamlines=[reference_streamline], filename=out_parcellation_filename.replace('.nii.gz', '_reference_streamline.trk'), reference_image=binary_envelope)

        if reduced_streamlines is not None:
            save_trk_streamlines(streamlines=reduced_streamlines, filename=out_parcellation_filename.replace('.nii.gz', '_reduced_streamlines.trk'), reference_image=binary_envelope)

        if colors is not None:
            colors_file_name = out_parcellation_filename.replace('.nii.gz', '_colored.fib')
            save_as_vtk_fib(streamlines=oriented_streamlines, out_filename=colors_file_name, colors=colors)

        if svc is not None:
            joblib.dump(svc, out_parcellation_filename.replace('.nii.gz', '_svc.pkl'))
        

    return parcellation, reference_streamline, reduced_streamlines, svc


def main():

    parser = argparse.ArgumentParser(description='RadTract Tract Parcellation')
    parser.add_argument('--streamlines', type=str, help='Input streamline file (.trk)')
    parser.add_argument('--envelope', type=str, help='Input streamline envelope file. The envelope defines the area of the parcellation. If no envelope is set, the envelope is automatically calculated using --reference or --start as reference image. Either the start region, the reference image or the actual envelope needs to be provided.', default=None)
    parser.add_argument('--reference', type=str, help='Reference image used to automatically calulate binary envelope if the envelope is not set. If reference is not set, but --start is specified, the start region is used as reference image for the envelope calculation.', default=None)
    parser.add_argument('--start', type=str, help='Input binary start region file. Use this to define the start of the tract, in order to avoid reversed parcel ordering between the same tract in different subjects.', default=None)
    parser.add_argument('--num_parcels', type=int, help='Number of parcels (default is an automatic estimation to obtain parcels of ~5 voxels thickness). In multy subject studies, set this to a fixed value to avoid different numbers of parcels between subjects.', default=None)
    parser.add_argument('--type', type=str, help='Type of parcellation (\'hyperplane\', \'centerline\', or \'static\')', default='hyperplane')
    parser.add_argument('--save_intermediate_files', help='Save intermediate files (envelope, colored streamlines, ...)', action='store_true')
    parser.add_argument('--streamline_space', help='If True, no voxel-space parcellation will be created but each streamline point will be assigned a label. The output is a pickled dict with keys \'points\' and \'parcels\', usable for classic tractometry (see features.py or radtract_features command).', action='store_true')
    parser.add_argument('--output', type=str, help='Output parcellation image file')

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    parcellate_tract(streamlines=args.streamlines,
                     parcellation_type=args.type,
                     binary_envelope=args.envelope,
                     reference_image=args.reference,
                     num_parcels=args.num_parcels,
                     start_region=args.start,
                     save_intermediate_files=args.save_intermediate_files,
                     streamline_space=args.streamline_space,
                     out_parcellation_filename=args.output)


if __name__ == '__main__':
    main()

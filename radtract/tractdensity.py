# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

import nibabel as nib
import numpy as np
import vtk
from dipy.tracking.streamline import transform_streamlines
from radtract.parcellation import load_trk_streamlines
from skimage.morphology import binary_closing
import argparse


def intersect_image(spacing, si, ei, sf, ef):
    """
    Calculate the intersection of a line segment with a voxel grid.
    :param spacing:
    :param si: start index
    :param ei: end index
    :param sf: continuous start index
    :param ef: continuous end index
    :return: list of tuples (voxel index, length of segment in voxel)
    """

    out = []
    if np.array_equal(si, ei):
        d = np.empty(3)
        for i in range(3):
            d[i] = (sf[i]-ef[i])*spacing[i]

        out.append((si, np.linalg.norm(d)))
        return out

    bounds = np.empty(6)

    entrance_point = np.empty(3)
    exit_point = np.empty(3)

    start_point = np.empty(3)
    end_point = np.empty(3)

    t0 = vtk.reference(-1)
    t1 = vtk.reference(-1)
    for i in range(3):
        start_point[i] = sf[i]
        end_point[i] = ef[i]

        if si[i] > ei[i]:
            t = si[i]
            si[i] = ei[i]
            ei[i] = t

    for x in range(si[0], ei[0]+1):
        for y in range(si[1], ei[1]+1):
            for z in range(si[2], ei[2]+1):
                bounds[0] = x - 0.5
                bounds[1] = x + 0.5
                bounds[2] = y - 0.5
                bounds[3] = y + 0.5
                bounds[4] = z - 0.5
                bounds[5] = z + 0.5

                entry_plane = vtk.reference(-1)
                exit_plane = vtk.reference(-1)

                hit = vtk.vtkBox.IntersectWithLine(bounds, start_point, end_point, t0, t1, entrance_point, exit_point, entry_plane, exit_plane)
                if hit > 0:
                    if entry_plane >= 0 and exit_plane >= 0:
                        d = np.empty(3)
                        for i in range(3):
                            d[i] = (exit_point[i] - entrance_point[i])*spacing[i]
                        out.append(((x, y, z), np.linalg.norm(d)))
                    elif entry_plane >= 0:
                        d = np.empty(3)
                        for i in range(3):
                            d[i] = (ef[i] - entrance_point[i])*spacing[i]
                        out.append(((x, y, z), np.linalg.norm(d)))
                    elif exit_plane >= 0:
                        d = np.empty(3)
                        for i in range(3):
                            d[i] = (exit_point[i]-sf[i])*spacing[i]
                        out.append(((x, y, z), np.linalg.norm(d)))
    return out


def tract_envelope(streamlines: nib.streamlines.array_sequence.ArraySequence,
                   reference_image: nib.Nifti1Image,
                   do_closing: bool = False,
                   out_image_filename: str = None):
    """
    Convenience function for tract_density that calculates the binary bundle envelope.
    :param streamlines: input streamlines
    :param reference_image: defines geometry of output image
    :param do_closing: morphological closing of the binary image to remove holes
    :param out_image_filename: if not None, the output image will be saved to this file
    :return:
    """
    return tract_density(streamlines, reference_image, True, do_closing, out_image_filename)


def tract_density(streamlines: nib.streamlines.array_sequence.ArraySequence, # input streamlines
                   reference_image: nib.Nifti1Image, # defines geometry of output image
                   binary: bool = False, # if true, the output image will be a binary image with 1 for voxels that are part of the bundle and 0 outside
                   do_closing: bool = False, # morphological closing of the binary image to remove holes
                   out_image_filename: str = None): # if not None, the output image will be saved to this file
    """
    Calculate the tract density image for a set of streamlines using the true length of each streamline segment in each voxel.
    :param streamlines: input streamlines
    :param reference_image: defines geometry of output image
    :param binary: if true, the output image will be a binary image with 1 for voxels that are part of the bundle and 0 outside
    :param do_closing: morphological closing of the binary image to remove holes
    :param out_image_filename: if not None, the output image will be saved to this file
    :return:
    """
    if binary:
        print('Calculating bundle envelope')
    else:
        print('Calculating tract density image')

    # Load streamlines and reference image if they are file paths
    if type(streamlines) is str:
        streamlines = load_trk_streamlines(streamlines)
    if type(reference_image) is str:
        reference_image = nib.load(reference_image)

    # Create an empty image with the same dimensions as the reference image
    image_data = np.copy(reference_image.get_fdata())
    image_data.fill(0)
    affine = reference_image.affine
    spacing = reference_image.header['pixdim'][1:4]

    # Transform streamlines to voxel space
    streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

    # Loop over each streamline and calculate the intersection with the image
    for s in streamlines:
        num_points = len(s)
        for j in range(num_points-1):
            start_index_cont = s[j]
            start_index = np.round(start_index_cont).astype('int64')

            end_index_cont = s[j+1]
            end_index = np.round(end_index_cont).astype('int64')

            segments = intersect_image(spacing, start_index, end_index, start_index_cont, end_index_cont)
            for seg in segments:
                if binary:
                    image_data[seg[0][0], seg[0][1], seg[0][2]] = 1
                else:
                    image_data[seg[0][0], seg[0][1], seg[0][2]] += seg[1]

    # Perform morphological closing if binary and do_closing is True
    if binary and do_closing:
        image_data = binary_closing(image_data)

    # Create Nifti1Image object with the image data
    if binary:
        image_data = image_data.astype('uint8')
        tdi = nib.Nifti1Image(image_data, header=reference_image.header, affine=reference_image.affine, dtype='uint8')
    else:
        tdi = nib.Nifti1Image(image_data, header=reference_image.header, affine=reference_image.affine)

    # Save the image to file if out_image_filename is not None
    if out_image_filename is not None:
        nib.save(tdi, out_image_filename)

    print('done')

    return tdi


def main():
    parser = argparse.ArgumentParser(description='RadTract Tract Density Image')
    parser.add_argument('--streamlines', type=str, help='Input streamline file')
    parser.add_argument('--reference', type=str, help='Reference image file')
    parser.add_argument('--binary', type=bool, help='Output binary envelope instead of tract density image', default=False)
    parser.add_argument('--do_closing', type=bool, help='Perform morphological closing of the envelope to remove holes', default=False)
    parser.add_argument('--output', type=str, help='Output tdi/envelope image file')
    args = parser.parse_args()

    streamlines = load_trk_streamlines(args.streamlines)
    ref_image = nib.load(args.reference)

    tract_density(streamlines=streamlines,
                  reference_image=ref_image,
                  binary=args.binary,
                  do_closing=args.do_closing,
                  out_image_filename=args.output)


if __name__ == '__main__':
    main()

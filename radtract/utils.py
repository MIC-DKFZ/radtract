# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

from dipy.io.streamline import load_trk, save_trk
from fury.io import save_polydata
from fury.utils import lines_to_vtk_polydata, numpy_to_vtk_colors
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import nibabel as nib


def save_trk_streamlines(streamlines: nib.streamlines.array_sequence.ArraySequence, filename: str, reference_image: nib.Nifti1Image):
    """
    Convenience function to save streamlines to a trk file
    :param filename: filename of trk file
    """
    fib = StatefulTractogram(streamlines, reference_image, Space.RASMM)
    save_trk(fib, filename, bbox_valid_check=False)


def load_trk_streamlines(filename: str):
    """
    Convenience function to load streamlines from a trk file
    :param filename: filename of trk file
    :return: streamlines in dipy format
    """
    fib = load_trk(filename, "same", bbox_valid_check=False)
    streamlines = fib.streamlines
    return streamlines

def save_as_vtk_fib(streamlines, out_filename, colors=None):

    polydata, _ = lines_to_vtk_polydata(streamlines)
    if colors is not None:
        vtk_colors = numpy_to_vtk_colors(colors)
        vtk_colors.SetName("FIBER_COLORS")
        polydata.GetPointData().AddArray(vtk_colors)
    save_polydata(polydata=polydata, file_name=out_filename, binary=True)


def main():
    pass


if __name__ == '__main__':
    main()

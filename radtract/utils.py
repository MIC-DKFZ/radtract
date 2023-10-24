# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

from dipy.io.streamline import load_trk, save_trk
from fury.io import save_polydata
from fury.utils import lines_to_vtk_polydata, numpy_to_vtk_colors
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fury.colormap import distinguishable_colormap


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


def plot_parcellation(nifti_file, mip_axis):
    """
    
    """

    image = nib.load(nifti_file)
    data = image.get_fdata()
    mip = np.max(data, axis=mip_axis)
    nb_labels = len(np.unique(mip)) - 1
    fury_cmap = distinguishable_colormap(nb_colors=nb_labels)
    fury_cmap = [np.array([0, 0, 0, 1])] + fury_cmap
    mpl_cmap = ListedColormap(fury_cmap)
    plt.imshow(mip.T, cmap=mpl_cmap, origin='lower')
    plt.show()


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



def main():
    pass


if __name__ == '__main__':
    main()

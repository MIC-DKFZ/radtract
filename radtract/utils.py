# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

from dipy.io.streamline import load_trk, save_trk
from fury.io import save_polydata
from fury.utils import lines_to_vtk_polydata, numpy_to_vtk_colors
from dipy.io.stateful_tractogram import Space, StatefulTractogram, logger
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fury.colormap import distinguishable_colormap
import joblib
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import stats
import os
import zipfile
import uuid


def save_trk_streamlines(streamlines: nib.streamlines.array_sequence.ArraySequence, filename: str, reference):
    """
    Convenience function to save streamlines to a trk file
    :param filename: filename of trk file
    """    
    if type(reference) == StatefulTractogram:
        logger.setLevel('ERROR')
        fib = StatefulTractogram(streamlines, reference, space=reference.space, origin=reference.origin)
    elif type(reference) == nib.Nifti1Image:
        fib = StatefulTractogram(streamlines, reference, Space.RASMM)
    else:
        raise ValueError('Reference has to be either a StatefulTractogram or a Nifti1Image.')
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
    """
    Saves streamlines as vtk fib file.
    :param streamlines: streamlines in dipy format
    :param out_filename: output filename
    :param colors: colors for each streamline point
    :return:
    """
    polydata, _ = lines_to_vtk_polydata(streamlines)
    if colors is not None:
        vtk_colors = numpy_to_vtk_colors(colors)
        vtk_colors.SetName("FIBER_COLORS")
        polydata.GetPointData().AddArray(vtk_colors)
    save_polydata(polydata=polydata, file_name=out_filename, binary=True)


def plot_parcellation(nifti_file, mip_axis, slice=0.5, thickness=0, out_file=None):
    """
    Plots a parcellation as a maximum intensity projection.
    """
    image = nib.load(nifti_file)
    data = image.get_fdata()
    if thickness > 0:
        i = int(slice * data.shape[mip_axis])
        if mip_axis == 0:
            data = data[i-thickness:i+thickness, :, :]
        elif mip_axis == 1:
            data = data[:, i-thickness:i+thickness, :]
        elif mip_axis == 2:
            data = data[:, :, i-thickness:i+thickness]
    mip = np.max(data, axis=mip_axis)
    nb_labels = len(np.unique(mip)) - 1
    fury_cmap = distinguishable_colormap(nb_colors=nb_labels)
    fury_cmap = [np.array([0, 0, 0, 1])] + fury_cmap
    mpl_cmap = ListedColormap(fury_cmap)
    # set figure size
    plt.figure(figsize=(10, 10))
    plt.imshow(mip.T, cmap=mpl_cmap, origin='lower')
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    # remove white padding
    plt.tight_layout(pad=0)

    if out_file is not None:
        plt.savefig(out_file, dpi=600)
    else:
        plt.show()


def estimate_ci(y_true, y_scores):
    """
    Estimates the confidence interval for a ROC curve using the standard error.
    :param y_true: true labels
    :param y_scores: predicted scores
    :return: confidence interval size, AUROC score
    """
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    ci_size = 0

    if len(classes) > 2:

        auc_scores = []
        for i in range(len(classes)):
            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)

        # Compute the standard error and the confidence intervals for each class
        conf_intervals = []
        for i in range(len(classes)):
            n1 = sum(y_true_bin[:, i])
            n2 = len(y_true_bin[:, i]) - n1
            roc_auc = auc_scores[i]
            
            q1 = roc_auc / (2.0 - roc_auc)
            q2 = 2 * roc_auc ** 2 / (1.0 + roc_auc)
            se = np.sqrt((roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2))

            conf_interval = stats.norm.interval(0.95, loc=roc_auc, scale=se)
            conf_intervals.append(conf_interval)

        # Compute weighted average of AUC scores and confidence intervals
        weights = [sum(y_true_bin[:, i]) for i in range(len(classes))]
        avg_auc_score = np.average(auc_scores, weights=weights)
        avg_conf_interval = [np.average([conf_intervals[i][j] for i in range(len(classes))], weights=weights) for j in range(2)]
        ci_size = avg_auc_score - avg_conf_interval[0]
        return ci_size, avg_auc_score

    else:

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        # Compute the standard error and the confidence intervals
        n1 = sum(y_true)
        n2 = len(y_true) - n1
        
        q1 = roc_auc / (2.0 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1.0 + roc_auc)
        se = np.sqrt((roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2))

        conf_interval = stats.norm.interval(0.95, loc=roc_auc, scale=se)
        ci_size = roc_auc - conf_interval[0]
        return ci_size, roc_auc


def load_results(result_pkl, repetitions=10):
    """
    Loads the results of a cross-validation experiment.
    :param result_pkl: path to the result pickle file. format: (predicted probabilities, true labels)
    :return: pandas dataframe with the results as dictionary containing the AUROC scores, confidence intervals and p, y per repetition
    """
    
    aucs = dict()
    aucs['AUROC'] = []
    aucs['CI'] = []
    aucs['p'] = []
    aucs['y'] = []

    p, y = joblib.load(result_pkl)
    nsamples = len(p)//repetitions
    
    for rep in range(repetitions):
        p_rep = p[rep*nsamples:(rep+1)*nsamples]
        y_rep = y[rep*nsamples:(rep+1)*nsamples]

        ci, roc_auc = estimate_ci(y_rep, p_rep)

        aucs['AUROC'].append(roc_auc)
        aucs['CI'].append(ci)
        aucs['p'].append(p_rep)
        aucs['y'].append(y_rep)

    aucs = pd.DataFrame(aucs)
    return aucs


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


def point_label_file_to_mitkpointset(point_label_file, out_path, streamline_files = [], point_size=1.0):
    """
    Converts a point label file to a scene file of mitk pointsets. This is useful to visualize the streamline points with labels as point clouds in MITK Diffusion (https://github.com/MIC-DKFZ/MITK-Diffusion/).
    :param point_label_file: path to point label file
    :param out_path: output path
    :param streamline_files: list of streamline files to include in the mitk scene file
    :return:
    """

    points_per_parcel = pd.read_pickle(point_label_file)

    pointsets = dict()

    for point, parcel in zip(points_per_parcel['points'], points_per_parcel['parcels']):
        
        if parcel not in pointsets.keys():
            pointsets[parcel] = []
        pointsets[parcel].append(point)

    o = out_path + 'scenefile/'
    os.makedirs(o, exist_ok=True)

    # write index
    first_file = 'aaaaaa'
    text = '<?xml version="1.0" encoding="UTF-8"?>'
    text += '<Version Writer="/home/neher/coding/mitk/mitk/Modules/SceneSerialization/src/mitkSceneIO.cpp" Revision="$Revision: 17055 $" FileVersion="1"/>'
    index = 0
    for parcel in sorted(pointsets.keys()):
        ps = first_file + '_P' + str(parcel) + '.mps'
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        props2 = first_file
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        text += '<node UID="' + str(uuid.uuid4()) + '">'
        text += '<data type="PointSet" file="' + ps + '" UID="' + str(uuid.uuid4()) + '">'
        text += '</data>'
        text += '<properties file="' + props2 + '"/>'
        text += '</node>'
    
    for streamline_file in streamline_files:

        name = streamline_file.split('/')[-1]

        ps = first_file + '_' + name
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        props2 = first_file
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        text += '<node UID="' + str(uuid.uuid4()) + '">'
        text += '<data type="FiberBundle" file="' + ps + '" UID="' + str(uuid.uuid4()) + '">'
        text += '</data>'
        text += '<properties file="' + props2 + '"/>'
        text += '</node>'
        
    with open(o + 'index.xml', 'w') as f:
        f.write(text)

    first_file = 'aaaaaa'
    lut_cmap = distinguishable_colormap(nb_colors=len(pointsets.keys()))
    index = 0
    for parcel in sorted(pointsets.keys()):

        ps = first_file + '_P' + str(parcel) + '.mps'
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        props2 = first_file
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        # write pointset
        text = '<?xml version="1.0" encoding="UTF-8"?><point_set_file><file_version>0.1</file_version><point_set><time_series><time_series_id>0</time_series_id><Geometry3D ImageGeometry="false" FrameOfReferenceID="0">'
        text += '<IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0" m_0_2="0" m_1_0="0" m_1_1="1" m_1_2="0" m_2_0="0" m_2_1="0" m_2_2="1"/><Offset type="Vector3D" x="0" y="0" z="0"/><Bounds>'
        text += '<Min type="Vector3D" x="89.933372497558594" y="98.688766479492188" z="-0.39603650569915771"/><Max type="Vector3D" x="127.03989410400391" y="165.80229187011719" z="141.04673767089844"/></Bounds></Geometry3D>'
        i = 0
        for p in pointsets[parcel]:
            text += '<point><id>' + str(i) + '</id><specification>0</specification>'
            text += '<x>' + str(p[0]) + '</x>'
            text += '<y>' + str(p[1]) + '</y>'
            text += '<z>' + str(p[2]) + '</z>'
            text += '</point>'
            i += 1
        text += '</time_series></point_set></point_set_file>'
        with open(o + ps, 'w') as f:
            f.write(text)

        color = lut_cmap[parcel-1]

        # write props2
        text = '<?xml version="1.0" encoding="UTF-8"?>'
        text += '<Version Writer="/home/neher/coding/mitk/mitk/Modules/SceneSerializationBase/src/mitkPropertyListSerializer.cpp" Revision="$Revision: 17055 $" FileVersion="1"/>'
        text += '<property key="pointsize" type="FloatProperty">'
        text += '<float value="' + str(point_size) + '"/>'
        text += '</property>'
        text += '<property key="color" type="ColorProperty">'
        text += '<color r="' + str(color[0]) + '" g="' + str(color[1]) + '" b="' + str(color[2]) + '"/>'
        text += '</property>'
        text += '<property key="name" type="StringProperty">'
        text += '<string value="P' + str(parcel) + '"/>'
        text += '</property>'
        with open(o + props2, 'w') as f:
            f.write(text)

    for streamline_file in streamline_files:

        name = streamline_file.split('/')[-1]

        ps = first_file + '_' + name
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        props2 = first_file
        first_file = first_file[:index] + chr(ord(first_file[index]) + 1) + first_file[index+1:]
        if first_file[index] == 'z':
            index += 1

        os.system('cp ' + streamline_file + ' ' + o + ps)

        # write props2
        text = '<?xml version="1.0" encoding="UTF-8"?>'
        text += '<Version Writer="/home/neher/coding/mitk/mitk/Modules/SceneSerializationBase/src/mitkPropertyListSerializer.cpp" Revision="$Revision: 17055 $" FileVersion="1"/>'
        text += '<property key="shape.tuberadius" type="FloatProperty">'
        text += '<float value="0.1"/>'
        text += '</property>'
        text += '<property key="name" type="StringProperty">'
        text += '<string value="' + name.replace('.fib', '') + '"/>'
        text += '</property>'
        with open(o + props2, 'w') as f:
            f.write(text)

    # create zip of o in out_path
    zipf = zipfile.ZipFile(out_path + 'scenefile.mitk', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(o):
        for file in files:
            zipf.write(os.path.join(root, file), file)


def main():
    pass


if __name__ == '__main__':
    main()


import os
import nibabel as nb
import numpy as np

from pydra.tasks.nibabel.utils import apply_mask

cwd = os.path.dirname(os.path.realpath(__file__))
nifti_test_file = cwd + "/data/t1w.nii"
zeros_mask_file = cwd + "/data/zeros_mask.nii"
ones_mask_file = cwd + "/data/ones_mask.nii"
half_zeros_half_ones_mask_file = cwd + "/data/half_zero_ones_mask.nii"

sorted_array = np.arange(1000).reshape(10, 10, 10)
ones_mask_array = np.ones((10, 10, 10))
zeros_mask_array = np.zeros((10, 10, 10))
m1 = np.zeros((5, 10, 10))
m2 = np.ones((5, 10, 10))
half_zeros_half_ones_mask = np.concatenate((m1, m2), axis=0)


hdr = nb.Nifti1Header()
hdr.set_data_shape((10, 10, 10))
hdr.set_zooms((1.0, 1.0, 1.0))  # set voxel size
hdr.set_xyzt_units(2)  # millimeters
hdr.set_qform(np.diag([1, 2, 3, 1]))
nb.save(
    nb.Nifti1Image(
        sorted_array,
        hdr.get_best_affine(),
        header=hdr,
    ),
    nifti_test_file,
)
nb.save(
    nb.Nifti1Image(
        ones_mask_array,
        hdr.get_best_affine(),
        header=hdr,
    ),
    ones_mask_file,
)
nb.save(
    nb.Nifti1Image(
        zeros_mask_array,
        hdr.get_best_affine(),
        header=hdr,
    ),
    zeros_mask_file,
)
nb.save(
    nb.Nifti1Image(
        half_zeros_half_ones_mask,
        hdr.get_best_affine(),
        header=hdr,
    ),
    half_zeros_half_ones_mask_file,
)


def test_apply_mask_with_ones():

    task = apply_mask(in_file=nifti_test_file, in_mask=ones_mask_file)
    result = task()

    assert np.array_equal(sorted_array, nb.load(result.output.out_file).get_data())


def test_apply_mask_with_zeros():

    task = apply_mask(in_file=nifti_test_file, in_mask=zeros_mask_file)
    result = task()

    assert np.count_nonzero(nb.load(result.output.out_file).get_data()) == 0


def test_apply_mask_with_half_zeros_half_ones():
    task = apply_mask(in_file=nifti_test_file, in_mask=half_zeros_half_ones_mask_file)
    result = task()
    result_data = nb.load(result.output.out_file).get_data()
    nifti_test_file_data = nb.load(nifti_test_file).get_data()

    assert np.count_nonzero(result_data[0:5]) == 0
    assert np.array_equal(result_data[5:10], nifti_test_file_data[5:10])

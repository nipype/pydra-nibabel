import nibabel as nb
import tempfile
from pathlib import Path
import numpy as np
import pytest
from niworkflows.interfaces.nibabel import RegridToZooms
from pydra.tasks.nibabel.utils import apply_mask, regrid_to_zooms


@pytest.fixture
def nifti_header():
    hdr = nb.Nifti1Header()
    hdr.set_data_shape((10, 10, 10))
    hdr.set_zooms((1.0, 1.0, 1.0))  # set voxel size
    hdr.set_xyzt_units(2)  # millimeters
    hdr.set_qform(np.diag([1, 2, 3, 1]))
    return hdr


@pytest.fixture
def tmp_dir():
    return Path(tempfile.mkdtemp())


@pytest.fixture
def sorted_image(nifti_header, tmp_dir):
    sorted_array = np.arange(1000).reshape(10, 10, 10)
    fpath = tmp_dir / "sorted.nii"
    nb.save(
        nb.Nifti1Image(
            sorted_array,
            nifti_header.get_best_affine(),
            header=nifti_header,
        ),
        str(fpath),
    )
    return {"file": fpath, "array": sorted_array}


@pytest.fixture
def ones_mask(nifti_header, tmp_dir):
    ones_mask_array = np.ones((10, 10, 10))
    fpath = tmp_dir / "ones_mask.nii"
    nb.save(
        nb.Nifti1Image(
            ones_mask_array,
            nifti_header.get_best_affine(),
            header=nifti_header,
        ),
        str(fpath),
    )
    return {"file": fpath, "array": ones_mask_array}


@pytest.fixture
def zeros_mask(nifti_header, tmp_dir):
    zeros_mask_array = np.zeros((10, 10, 10))
    fpath = tmp_dir / "zeros_mask.nii"
    nb.save(
        nb.Nifti1Image(
            zeros_mask_array,
            nifti_header.get_best_affine(),
            header=nifti_header,
        ),
        fpath,
    )
    return {"file": fpath, "array": zeros_mask_array}


@pytest.fixture
def half_zeros_mask(nifti_header, tmp_dir):
    half_zeros_half_ones_mask = np.concatenate(
        (np.zeros((5, 10, 10)), np.ones((5, 10, 10))), axis=0
    )
    fpath = tmp_dir / "half_zeros_mask.nii"
    nb.save(
        nb.Nifti1Image(
            half_zeros_half_ones_mask,
            nifti_header.get_best_affine(),
            header=nifti_header,
        ),
        fpath,
    )
    return {"file": fpath, "array": half_zeros_half_ones_mask}


@pytest.fixture
def wrong_shape_mask(nifti_header, tmp_dir):
    wrong_shape_mask = np.arange(1000).reshape(5, 20, 10)
    fpath = tmp_dir / "wrong_shape_mask.nii"
    nb.save(
        nb.Nifti1Image(
            wrong_shape_mask,
            nifti_header.get_best_affine(),
            header=nifti_header,
        ),
        fpath,
    )
    return {"file": fpath, "array": wrong_shape_mask}


@pytest.fixture
def wrong_affine_mask(nifti_header, tmp_dir):
    wrong_affine_mask = np.ones((10, 10, 10))
    nifti_header.set_qform(np.diag([2, 4, 6, 2]))
    fpath = tmp_dir / "wrong_affine_mask.nii"
    nb.save(
        nb.Nifti1Image(
            wrong_affine_mask,
            nifti_header.get_best_affine(),
            header=nifti_header,
        ),
        fpath,
    )
    return {"file": fpath, "array": wrong_affine_mask}


def test_apply_mask_with_ones(sorted_image, ones_mask):
    task = apply_mask(in_file=sorted_image["file"], in_mask=ones_mask["file"])
    result = task()

    assert np.array_equal(
        sorted_image["array"], nb.load(result.output.out_file).get_data()
    )


def test_apply_mask_with_zeros(sorted_image, zeros_mask):
    task = apply_mask(in_file=sorted_image["file"], in_mask=zeros_mask["file"])
    result = task()

    assert np.count_nonzero(nb.load(result.output.out_file).get_data()) == 0


def test_apply_mask_with_half_zeros_half_ones(sorted_image, half_zeros_mask):
    task = apply_mask(in_file=sorted_image["file"], in_mask=half_zeros_mask["file"])
    result = task()
    result_data = nb.load(result.output.out_file).get_data()

    assert np.count_nonzero(result_data[0:5]) == 0
    assert np.array_equal(result_data[5:10], sorted_image["array"][5:10])


def test_apply_mask_raises_exception_with_wrong_shape(sorted_image, wrong_shape_mask):
    with pytest.raises(ValueError, match="Image and mask sizes do not match."):
        task = apply_mask(
            in_file=sorted_image["file"], in_mask=wrong_shape_mask["file"]
        )
        task()


def test_apply_mask_raises_exception_with_wrong_affine(sorted_image, wrong_affine_mask):
    with pytest.raises(
        ValueError, match="Image and mask affines are not similar enough."
    ):
        task = apply_mask(
            in_file=sorted_image["file"], in_mask=wrong_affine_mask["file"]
        )
        task()


# FIXME
# def test_regrid_to_zooms(test_nifti, nifti_header):
# task = regrid_to_zooms(in_file=nifti_header.nifti_test_file, zooms=(1, 1, 1))
# result = task()

# task2 = RegridToZooms()
# task2.inputs.in_file = nifti_header.nifti_test_file
# task2.inputs.zooms = (1, 1, 1)
# result2 = task2.run()

# res = nb.load(result.output.out_file).get_data()
# assert np.array_equal(nifti_header.sorted_array, res)

import os
import nibabel as nb
import numpy as np

from pydra.tasks.nibabel.utils import apply_mask


def test_apply_mask_to_self():
    cwd = os.path.dirname(os.path.realpath(__file__))
    nifti_test_file = cwd + "/data/t1w.nii"

    random_data = np.random.randint(0, 2, size=[10, 10, 10])

    hdr = nb.Nifti1Header()
    hdr.set_data_shape((10, 10, 10))
    hdr.set_zooms((1.0, 1.0, 1.0))  # set voxel size
    hdr.set_xyzt_units(2)  # millimeters
    hdr.set_qform(np.diag([1, 2, 3, 1]))

    nb.save(
        nb.Nifti1Image(
            random_data,
            hdr.get_best_affine(),
            header=hdr,
        ),
        nifti_test_file,
    )

    task = apply_mask(in_file=nifti_test_file, in_mask=nifti_test_file)
    result = task()

    assert np.array_equal(random_data, nb.load(result.output.out_file).get_data())

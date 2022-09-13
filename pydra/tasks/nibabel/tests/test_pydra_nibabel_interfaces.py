#from pathlib import Path
import nibabel as nb
import numpy as np
#import pytest

import os

from pydra.tasks.nibabel.utils import apply_mask

def test_apply_mask():
    # Minimal code to create a NIfTI file using nibabel
    # Create a random Nifti file to satisfy BIDS parsers
    import nibabel as nb
    nifti_test_file = os.path.dirname(os.path.realpath(__file__)) + "/data/t1w.nii"

    hdr = nb.Nifti1Header()
    hdr.set_data_shape((10, 10, 10))
    hdr.set_zooms((1.0, 1.0, 1.0))  # set voxel size
    hdr.set_xyzt_units(2)  # millimeters
    hdr.set_qform(np.diag([1, 2, 3, 1]))
    nb.save(
        nb.Nifti1Image(
            np.random.randint(0, 1, size=[10, 10, 10]),
            hdr.get_best_affine(),
            header=hdr,
        ),
        nifti_test_file
    )
    mask = []

    result = apply_mask(in_file = nifti_test_file, in_mask = mask)
    
    assert result is not []
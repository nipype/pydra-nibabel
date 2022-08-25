import nibabel as nb
import numpy as np
from pydra import mark


@mark.task
@mark.annotate({"return": {"out_file": File}})
def apply_mask(in_file: File, in_mask: File, threshold: float = 0.5):

    img = nb.load(in_file)
    msknii = nb.load(in_mask)
    msk = msknii.get_fdata() > threshold

    out_file = fname_presuffix(in_file, suffix="_masked", newpath=runtime.cwd)

    if img.dataobj.shape[:3] != msk.shape:
        raise ValueError("Image and mask sizes do not match.")

    if not np.allclose(img.affine, msknii.affine):
        raise ValueError("Image and mask affines are not similar enough.")

    if img.dataobj.ndim == msk.ndim + 1:
        msk = msk[..., np.newaxis]

    masked = img.__class__(img.dataobj * msk, None, img.header)
    masked.to_filename(self._results["out_file"])

    return out_file

from pathlib import Path
from typing import Tuple
import nibabel as nb
import numpy as np
from pydra import mark
from pydra.engine.specs import File

import os.path as op


@mark.task
@mark.annotate({"return": {"out_file": File}})
def apply_mask(in_file: File, in_mask: File, threshold: float = 0.5):

    img = nb.load(in_file)
    msknii = nb.load(in_mask)
    msk = msknii.get_fdata() > threshold

    out_file = fname_presuffix(in_file, suffix="_masked", newpath=str(Path.cwd()))

    if img.dataobj.shape[:3] != msk.shape:
        raise ValueError("Image and mask sizes do not match.")

    if not np.allclose(img.affine, msknii.affine):
        raise ValueError("Image and mask affines are not similar enough.")

    if img.dataobj.ndim == msk.ndim + 1:
        msk = msk[..., np.newaxis]

    masked = img.__class__(img.dataobj * msk, None, img.header)
    masked.to_filename(out_file)

    return out_file


def fname_presuffix(fname, prefix="", suffix="", newpath=None, use_ext=True):
    """Manipulates path and name of input filename

    Parameters
    ----------
    fname : string
        A filename (may or may not include path)
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    Absolute path of the modified filename

    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    >>> from nipype.interfaces.base import Undefined
    >>> fname_presuffix(fname, 'pre', 'post', Undefined) == \
            fname_presuffix(fname, 'pre', 'post')
    True

    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ""

    # No need for isdefined: bool(Undefined) evaluates to False
    if newpath:
        pth = op.abspath(newpath)
    return op.join(pth, prefix + fname + suffix + ext)


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext


@mark.task
@mark.annotate({"return": {"out_file": File}})
def regrid_to_zooms(
    in_file: File,
    zooms: Tuple,
    order: int = 3,
    clip: bool = True,
    smooth: (bool or float) = False,
):
    """Change the resolution of an image (regrid)."""

    out_file = fname_presuffix(in_file, suffix="_regrid", newpath=str(Path.cwd()))
    resample_by_spacing(
        in_file,
        zooms,
        order=order,
        clip=clip,
        smooth=smooth,
    ).to_filename(out_file)
    return out_file


def resample_by_spacing(in_file, zooms, order=3, clip=True, smooth=False):
    """Regrid the input image to match the new zooms."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import map_coordinates

    if isinstance(in_file, (str, Path)):
        in_file = nb.load(in_file)

    # Prepare output x-forms
    sform, scode = in_file.get_sform(coded=True)
    qform, qcode = in_file.get_qform(coded=True)

    hdr = in_file.header.copy()
    zooms = np.array(zooms)

    # Calculate the factors to normalize voxel size to the specific zooms
    pre_zooms = np.array(in_file.header.get_zooms()[:3])

    # Calculate an affine aligned with cardinal axes, for simplicity
    card = nb.affines.from_matvec(np.diag(pre_zooms))
    extent = card[:3, :3].dot(np.array(in_file.shape[:3]))
    card[:3, 3] = -0.5 * extent

    # Cover the FoV with the new grid
    new_size = np.ceil(extent / zooms).astype(int)
    offset = (extent - np.diag(zooms).dot(new_size)) * 0.5
    new_card = nb.affines.from_matvec(np.diag(zooms), card[:3, 3] + offset)

    # Calculate the new indexes
    new_grid = np.array(
        np.meshgrid(
            np.arange(new_size[0]),
            np.arange(new_size[1]),
            np.arange(new_size[2]),
            indexing="ij",
        )
    ).reshape((3, -1))

    # Calculate the locations of the new samples, w.r.t. the original grid
    ijk = np.linalg.inv(card).dot(
        new_card.dot(np.vstack((new_grid, np.ones((1, new_grid.shape[1])))))
    )

    if smooth:
        from scipy.ndimage import gaussian_filter

        if smooth is True:
            smooth = np.maximum(0, (pre_zooms / zooms - 1) / 2)
        data = gaussian_filter(in_file.get_fdata(), smooth)
    else:
        data = np.asarray(in_file.dataobj)

    # Resample data in the new grid
    resampled = map_coordinates(
        data,
        ijk[:3, :],
        order=order,
        mode="constant",
        cval=0,
        prefilter=True,
    ).reshape(new_size)
    if clip:
        resampled = np.clip(resampled, a_min=data.min(), a_max=data.max())

    # Set new zooms
    hdr.set_zooms(zooms)

    # Get the original image's affine
    affine = in_file.affine.copy()
    # Determine rotations w.r.t. cardinal axis and eccentricity
    rot = affine.dot(np.linalg.inv(card))
    # Apply to the new cardinal, so that the resampling is consistent
    new_affine = rot.dot(new_card)

    if qcode != 0:
        hdr.set_qform(new_affine.dot(np.linalg.inv(affine).dot(qform)), code=int(qcode))
    if scode != 0:
        hdr.set_sform(new_affine.dot(np.linalg.inv(affine).dot(sform)), code=int(scode))
    if (scode, qcode) == (0, 0):
        hdr.set_qform(new_affine, code=1)
        hdr.set_sform(new_affine, code=1)

    # Create a new x-form affine, aligned with cardinal axes, 1mm3 and centered.
    return nb.Nifti1Image(resampled, new_affine, hdr)

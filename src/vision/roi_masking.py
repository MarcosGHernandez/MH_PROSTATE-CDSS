"""
Phase 14 - Prostate ROI Masking Module
=======================================
Loads and applies whole-gland prostate masks from the PI-CAI anatomical
delineations (Bosma22b AI segmenter) to constrain lesion predictions to
the anatomical prostate region.

The key insight: any detected lesion outside the prostate is a false positive
by anatomical definition. ROI masking enforces this hard constraint.
"""
import logging
import numpy as np
from pathlib import Path

import monai.transforms as mt

log = logging.getLogger("roi_masking")

# Default path to the AI-generated whole-gland masks (Bosma22b, 1501 patients)
GLAND_MASK_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data/raw/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b"
)


class ProstateROIMask:
    """
    Loads a whole-gland prostate mask and applies it as a binary constraint
    on segmentation predictions.

    Args:
        spacing (tuple): Target voxel spacing (must match the lesion prediction).
        fallback_ellipsoid (bool): If True and no mask found, generate a conservative
            ellipsoidal anatomical proxy instead of failing silently.
    """

    def __init__(
        self,
        spacing: tuple = (0.5, 0.5, 3.0),
        fallback_ellipsoid: bool = True,
    ):
        self.spacing = spacing
        self.fallback_ellipsoid = fallback_ellipsoid
        self._load_transform = self._build_load_transform()

    def _build_load_transform(self):
        return mt.Compose([
            mt.LoadImage(image_only=True),
            mt.EnsureChannelFirst(),
            mt.Spacing(pixdim=self.spacing, mode="nearest"),
            mt.Orientation(axcodes="RAS"),
            mt.EnsureType(),
        ])

    def load_mask(self, gland_mask_path: str) -> np.ndarray:
        """
        Load and resample a NIfTI whole-gland mask to match the prediction spacing.

        Returns:
            Binary numpy array of shape (X, Y, Z).
        """
        tensor = self._load_transform(gland_mask_path)
        mask = tensor.numpy().squeeze()   # (X, Y, Z)
        return (mask > 0).astype(np.uint8)

    def apply(
        self,
        prediction: np.ndarray,
        gland_mask: np.ndarray,
        target_shape: tuple = None,
    ) -> np.ndarray:
        """
        Apply the prostate gland mask to the lesion prediction via logical AND.

        If shapes differ (due to padding/cropping in the preprocessing pipeline),
        the mask is cropped or padded symmetrically to match the prediction volume.

        Args:
            prediction: (X, Y, Z) binary lesion prediction.
            gland_mask: (X, Y, Z) binary prostate mask (1 = inside prostate).
            target_shape: Optional override for final output shape.

        Returns:
            Masked prediction where extra-prostatic voxels are zeroed out.
        """
        pred_shape = prediction.shape

        if gland_mask.shape != pred_shape:
            # Pad or crop the mask to match prediction shape dimension by dimension
            gland_mask = self._resize_to_match(gland_mask, pred_shape)

        masked = prediction * gland_mask
        log.debug(
            f"ROI Masking: {prediction.sum()} raw voxels -> {masked.sum()} after masking"
            f" (dropped {prediction.sum() - masked.sum()} extra-prostatic voxels)"
        )
        return masked.astype(np.uint8)

    @staticmethod
    def _resize_to_match(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Symmetrically pad or center-crop `arr` to `target_shape`."""
        result = np.zeros(target_shape, dtype=arr.dtype)
        slices_src = []
        slices_dst = []
        for src_dim, tgt_dim in zip(arr.shape, target_shape):
            if src_dim >= tgt_dim:
                start = (src_dim - tgt_dim) // 2
                slices_src.append(slice(start, start + tgt_dim))
                slices_dst.append(slice(0, tgt_dim))
            else:
                pad = (tgt_dim - src_dim) // 2
                slices_src.append(slice(0, src_dim))
                slices_dst.append(slice(pad, pad + src_dim))

        result[tuple(slices_dst)] = arr[tuple(slices_src)]
        return result

    def make_fallback_ellipsoid(self, shape: tuple, center_fraction=0.55) -> np.ndarray:
        """
        Generate a conservative ellipsoidal prostate proxy when no mask is available.
        The ellipsoid covers the central 55% of the volume in each axis
        — a conservative anatomical prior for the prostate location in
        standardized RAS volumes.

        This is NOT clinically accurate but prevents total loss of the
        masking capability when a true segmentation is unavailable.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cx, cy, cz = [s // 2 for s in shape]
        rx = int(shape[0] * center_fraction / 2)
        ry = int(shape[1] * center_fraction / 2)
        rz = int(shape[2] * center_fraction / 2)

        x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
        ellipsoid = ((x - cx) ** 2 / rx**2 + (y - cy) ** 2 / ry**2 + (z - cz) ** 2 / rz**2) <= 1
        mask[ellipsoid] = 1
        log.warning("Using fallback ellipsoid prostate proxy — not clinically validated.")
        return mask

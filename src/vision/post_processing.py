"""
Phase 14 - Clinical Post-Processing Pipeline
=============================================
Sequential filter stack applied to raw U-Net sigmoid outputs:

    [Logits] -> Sigmoid
             -> ROI Masking (drop extra-prostatic voxels)  ← Phase 14 NEW
             -> Dynamic Threshold (0.38)
             -> Morphological Closing (3x3x1)
             -> CCA Filter (> 50mm³)
             -> [Clinical Prediction]
"""
import numpy as np
from scipy import ndimage
import logging

log = logging.getLogger("post_processing")


class ClinicalPostProcessor:
    """
    Applies clinical post-processing filters to raw 3D U-Net predictions.

    Args:
        threshold:       Sigmoid threshold for binary segmentation (default 0.38).
        min_volume_mm3:  Minimum lesion volume in mm³ per PI-RADS criteria (default 50).
        spacing:         Voxel spacing (x, y, z) in mm — must match inference spacing.
        use_roi_masking: Whether to apply whole-gland prostate ROI masking (Phase 14).
    """

    def __init__(
        self,
        threshold: float = 0.38,
        min_volume_mm3: float = 50.0,
        spacing: tuple = (0.5, 0.5, 3.0),
        use_roi_masking: bool = True,
    ):
        self.threshold = threshold
        self.min_volume_mm3 = min_volume_mm3
        self.spacing = spacing
        self.voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³/voxel
        self.use_roi_masking = use_roi_masking

        if use_roi_masking:
            from roi_masking import ProstateROIMask
            self._roi_masker = ProstateROIMask(spacing=spacing)
        else:
            self._roi_masker = None

    def __call__(
        self,
        raw_pred: np.ndarray,
        gland_mask_path: str = None,
        gland_mask_array: np.ndarray = None,
    ) -> np.ndarray:
        """
        Apply the full post-processing pipeline.

        Args:
            raw_pred:         3D numpy array — either raw logits or [0,1] probabilities.
            gland_mask_path:  Path to a NIfTI whole-gland prostate mask (Phase 14).
            gland_mask_array: Pre-loaded gland mask array (alternative to path).

        Returns:
            Binary 3D numpy array (uint8) — the cleaned lesion prediction.
        """
        # ── Step 0: Sigmoid (if logits) ────────────────────────────────────
        if raw_pred.max() > 1.0 or raw_pred.min() < 0.0:
            import torch
            probs = torch.sigmoid(torch.from_numpy(raw_pred.astype(np.float32))).numpy()
        else:
            probs = raw_pred.astype(np.float32)

        # ── Step 1: ROI Masking (Phase 14) ─────────────────────────────────
        if self.use_roi_masking and self._roi_masker is not None:
            gland_mask = None

            if gland_mask_array is not None:
                gland_mask = gland_mask_array
            elif gland_mask_path is not None:
                try:
                    gland_mask = self._roi_masker.load_mask(gland_mask_path)
                except Exception as e:
                    log.warning(f"Could not load gland mask from {gland_mask_path}: {e}")

            if gland_mask is not None:
                # Apply ROI masking to the probability map BEFORE thresholding
                # This zeroes out any probability outside the prostate boundary
                if gland_mask.shape != probs.shape:
                    gland_mask = self._roi_masker._resize_to_match(gland_mask, probs.shape)
                probs = probs * gland_mask.astype(np.float32)
                log.debug("Phase 14 ROI masking applied.")
            else:
                # Fallback to ellipsoid if no mask is available
                if self._roi_masker.fallback_ellipsoid:
                    gland_mask = self._roi_masker.make_fallback_ellipsoid(probs.shape)
                    probs = probs * gland_mask.astype(np.float32)

        # ── Step 2: Dynamic Thresholding ───────────────────────────────────
        binary_pred = (probs >= self.threshold).astype(np.uint8)

        if not np.any(binary_pred):
            return binary_pred

        # ── Step 3: Morphological Closing ──────────────────────────────────
        # Kernel 3x3x1: bridges in-plane gaps without smearing axially
        structure = np.ones((3, 3, 1), dtype=int)
        closed_pred = ndimage.binary_closing(binary_pred, structure=structure).astype(np.uint8)

        # ── Step 4: CCA Filter (drop sub-clinical noise) ───────────────────
        labeled_array, num_features = ndimage.label(closed_pred)
        cleaned_pred = np.zeros_like(closed_pred)

        kept = 0
        dropped = 0
        if num_features > 0:
            component_sizes = np.bincount(labeled_array.ravel())
            for val in range(1, num_features + 1):
                volume_mm3 = component_sizes[val] * self.voxel_volume
                if volume_mm3 >= self.min_volume_mm3:
                    cleaned_pred[labeled_array == val] = 1
                    kept += 1
                else:
                    dropped += 1

        log.debug(
            f"CCA: {num_features} components -> kept {kept}, dropped {dropped} "
            f"(threshold={self.min_volume_mm3}mm³)"
        )

        return cleaned_pred

"""
Shadow extraction and volume estimation for floating-head oil tanks.

Uses HSV/LAB color space enhancement, thresholding, and morphological
operations to extract crescent shadows from detected tank regions.
Volume is estimated as 1 - (inner_shadow_area / outer_shadow_area).

Adapted from NTUA ECE thesis — removed fastai dependency, uses pure numpy/PIL.
"""

import cv2
import numpy as np
from skimage import filters, measure, segmentation, morphology, color


def crop(box, image_array, factor_x=0.5, factor_y=0.6):
    """Crop a region around a bounding box with margins.

    Args:
        box: [y_min, x_min, y_max, x_max]
        image_array: numpy array (H, W, 3) in [0, 1] float or [0, 255] uint8
        factor_x, factor_y: margin factors

    Returns:
        (cropped_image, bbox_relative_to_crop)
    """
    y_min, x_min, y_max, x_max = box

    margin_x = int((x_max - x_min) * factor_x)
    margin_y = int((y_max - y_min) * factor_y)

    h, w = image_array.shape[:2]
    c_y_min = max(y_min - margin_y, 0)
    c_y_max = min(y_max + int(margin_y // 2), h)
    c_x_min = max(x_min - margin_x, 0)
    c_x_max = min(x_max + margin_x, w)

    margin_y_true = y_min - c_y_min
    margin_x_true = x_min - c_x_min

    c_bbox_relative = [
        margin_y_true,
        margin_x_true,
        (y_max - y_min) + margin_y_true,
        (x_max - x_min) + margin_x_true,
    ]

    c_tank_crop = image_array[c_y_min:c_y_max, c_x_min:c_x_max]
    return c_tank_crop, c_bbox_relative


class Tank:
    """Process a single detected tank to extract shadows and estimate volume."""

    def __init__(self, box, image_array, factor_x=0.5, factor_y=0.6):
        """
        Args:
            box: [y_min, x_min, y_max, x_max] in pixel coords
            image_array: numpy array (H, W, 3) float in [0, 1] range
        """
        self.image_array = image_array
        self.gt_coords = (box[0], box[1], box[2], box[3])
        y_min, x_min, y_max, x_max = self.gt_coords

        margin_x = int((x_max - x_min) * factor_x)
        margin_y = int((y_max - y_min) * factor_y)

        h, w = image_array.shape[:2]
        self.y_min = max(y_min - margin_y, 0)
        self.y_max = min(y_max + int(margin_y // 2), h)
        self.x_min = max(x_min - margin_x, 0)
        self.x_max = min(x_max + margin_x, w)

        margin_y_true = y_min - self.y_min
        margin_x_true = x_min - self.x_min

        self.bbox_relative = [
            margin_y_true,
            margin_x_true,
            (y_max - y_min) + margin_y_true,
            (x_max - x_min) + margin_x_true,
        ]

        # Crop the tank region — image_array is (H, W, 3)
        self.tank_crop = self.image_array[
            self.y_min : self.y_max, self.x_min : self.x_max
        ].copy()

        self.volume = 0.0
        self.blank = np.zeros(self.tank_crop.shape[:2])
        self.contours_select = []

        if self.tank_crop.size > 0:
            self.proc_tank()
            self.get_regions()

    def proc_tank(self):
        """Enhance shadows using HSV/LAB color spaces and threshold."""
        # Ensure float [0, 1] for skimage
        tank = self.tank_crop.astype(np.float64)
        if tank.max() > 1.0:
            tank = tank / 255.0

        hsv = color.rgb2hsv(tank)
        V = hsv[:, :, 2]

        lab = color.rgb2lab(tank)
        l1 = lab[:, :, 0]
        l3 = lab[:, :, 2]

        # Enhanced image: -(l1 + l3) / (V + 1)
        self.tank_hsv = -(l1 + l3) / (V + 1)

        # Threshold values
        try:
            t1 = filters.threshold_minimum(self.tank_hsv)
        except RuntimeError:
            t1 = filters.threshold_otsu(self.tank_hsv)
        t2 = filters.threshold_mean(self.tank_hsv)

        self.tank_thresh = self.tank_hsv > (0.5 * t1 + 0.4 * t2)

        # Morphological cleanup → labeled image
        cleaned = morphology.closing(self.tank_thresh)
        cleaned = morphology.area_closing(cleaned)
        cleared = segmentation.clear_border(filters.hessian(cleaned))
        self.label_image = measure.label(cleared)

    def get_regions(self):
        """Extract shadow regions and compute volume estimate."""
        self.regions_all = measure.regionprops(self.label_image)
        self.regions = []

        for region in self.regions_all:
            if intersection(self.bbox_relative, region.bbox) > 300:
                if region.area > 25:
                    b = region.bbox
                    thresh_mean = self.tank_thresh[b[0] : b[2], b[1] : b[3]].mean()
                    if abs(thresh_mean - region.image.mean()) < 0.06:
                        self.regions.append(region)

        areas = np.array([r.area for r in self.regions])

        if len(areas) > 1:
            idx2, idx1 = areas.argsort()[-2:]
            self.volume = 1 - self.regions[idx2].area / self.regions[idx1].area
        else:
            idx2, idx1 = 0, 0
            self.volume = 1.0

        # Create mask with just the two main shadow regions
        self.blank = np.zeros(self.tank_crop.shape[:2])

        if self.regions:
            for region in [self.regions[idx1], self.regions[idx2]]:
                y_min, x_min, y_max, x_max = region.bbox
                self.blank[y_min:y_max, x_min:x_max] += region.image.astype("uint8")

        # Find contours
        self.contours = measure.find_contours(self.blank, 0.5)
        if len(self.contours) > 1:
            contour_idxs = np.array([len(c) for c in self.contours]).argsort()[-2:]
        elif len(self.contours) == 1:
            contour_idxs = [0]
        else:
            contour_idxs = []
        self.contours_select = [self.contours[i] for i in contour_idxs]


def intersection(bb1, bb2):
    """Calculate pixel area intersection between two bounding boxes."""
    y_min1, x_min1, y_max1, x_max1 = bb1
    y_min2, x_min2, y_max2, x_max2 = bb2

    x_left = max(x_min1, x_min2)
    x_right = min(x_max1, x_max2)
    y_top = max(y_min1, y_min2)
    y_bottom = min(y_max1, y_max2)

    return max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)


def check_bb(bbox, shape):
    """
    Exclude bounding boxes that touch the image edge (tank extends beyond frame).
    """
    h, w = shape[:2]
    ymin, xmin, ymax, xmax = bbox
    if xmin <= 2 or xmin >= w - 2:
        return False
    if xmax <= 2 or xmax >= w - 2:
        return False
    if ymin <= 2 or ymin >= h - 2:
        return False
    if ymax <= 2 or ymax >= h - 2:
        return False
    return True


class MultiTank:
    """Process multiple detected tanks and compute volumes."""

    def __init__(self, bbs, image_array):
        """
        Args:
            bbs: list of [y_min, x_min, y_max, x_max] bounding boxes
            image_array: numpy array (H, W, 3) float in [0, 1] range
        """
        self.image_array = image_array
        # Filter out edge-touching boxes
        self.bbs = [b for b in bbs if check_bb(b, image_array.shape)]
        self.tanks = []
        for b in self.bbs:
            try:
                self.tanks.append(Tank(b, image_array))
            except Exception as e:
                print(f"Shadow extraction error for box {b}: {e}")

        self.mask = np.zeros(image_array.shape[:2])
        self.mask_binary = self.mask > 0
        if self.tanks:
            self._create_masks()

    def get_volumes(self):
        """Return list of volume estimates (floats) for each tank."""
        return [tank.volume for tank in self.tanks]

    def _create_masks(self):
        """Create combined shadow mask from all tanks."""
        mask = np.zeros(self.image_array.shape[:2])
        for i, tank in enumerate(self.tanks):
            tank_blank = (tank.blank > 0) * (i + 1)
            mask[tank.y_min : tank.y_max, tank.x_min : tank.x_max] += tank_blank
        self.mask = mask
        self.mask_binary = mask > 0

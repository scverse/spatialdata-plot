import numpy as np
from skimage.segmentation import find_boundaries, relabel_sequential


def _format_labels(labels):
    """Formats a label list."""
    formatted_labels = labels.copy()
    unique_labels = np.unique(labels)

    if 0 in unique_labels:
        # logger.warning("Found 0 in labels. Reindexing ...")
        formatted_labels += 1

    if ~np.all(np.diff(unique_labels) == 1):
        # logger.warning("Labels are non-consecutive. Relabeling ...")
        formatted_labels, _, _ = relabel_sequential(formatted_labels)

    return formatted_labels


def _label_segmentation_mask(segmentation: np.ndarray, annotations: dict) -> np.ndarray:
    """Relabels a segmentation according to the annotations df."""
    labeled_segmentation = segmentation.copy()
    all_cells = []

    for k, v in annotations.items():
        mask = np.isin(segmentation, v)
        labeled_segmentation[mask] = k
        all_cells.extend(v)

    # remove cells that are not indexed
    neg_mask = ~np.isin(segmentation, all_cells)
    labeled_segmentation[neg_mask] = 0

    return labeled_segmentation


def _render_label(mask, cmap_mask, img=None, alpha=0.2, alpha_boundary=1.0, mode="inner"):
    colored_mask = cmap_mask(mask)

    mask_bool = mask > 0
    mask_bound = np.bitwise_and(mask_bool, find_boundaries(mask, mode=mode))

    # blend
    if img is None:
        img = np.zeros(mask.shape + (4,), np.float32)
        img[..., -1] = 1

    im = img.copy()
    # print(colored_mask.shape, im.shape, np.zeros((im.shape[0], im.shape[1], 1)).shape)
    if im.shape[-1] == 3:
        im = np.concatenate([im, np.ones((im.shape[0], im.shape[1], 1))], axis=-1)

    # print(im.shape, im.dtype)
    # print(mask_bool.shape)

    im[mask_bool] = alpha * colored_mask[mask_bool] + (1 - alpha) * im[mask_bool]
    im[mask_bound] = alpha_boundary * colored_mask[mask_bound] + (1 - alpha_boundary) * im[mask_bound]

    return im

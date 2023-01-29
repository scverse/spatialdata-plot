from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def _get_linear_colormap(colors: list, background: str):
    return [LinearSegmentedColormap.from_list(c, [background, c], N=256) for c in colors]


def _get_listed_colormap(color_dict: dict):
    sorted_labels = sorted(color_dict.keys())
    colors = [color_dict[k] for k in sorted_labels]
    cmap = ListedColormap(["black"] + colors, N=len(colors) + 1)
    return cmap

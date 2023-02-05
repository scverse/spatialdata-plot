import spatialdata as sd


def _confirm_membership(sd: sd.SpatialData, key: str, slot: str) -> None:
    """
    Helper function to check if a key is part of the respective slot.
    """

    if not isinstance(key, str):

        raise TypeError("Parameter 'key' must be of type 'str'.")

    if not isinstance(slot, str):

        raise TypeError("Parameter 'slot' must be of type 'str'.")

    valid_slots = ["images", "labels", "points", "polygons", "shapes", "table"]
    if slot not in valid_slots:

        raise ValueError(f"Parameter 'slot' must be one of {valid_slots}.")

    if len(getattr(sd, slot)) == 0:

        raise ValueError(f"Slot '{slot}' is empty.")
    
    if key not in getattr(sd, slot).keys():

        raise ValueError(f"Key '{key}' not found in slot '{slot}'.")
    
    return True
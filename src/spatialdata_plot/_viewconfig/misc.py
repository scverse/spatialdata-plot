import re
from collections import Counter
from enum import Enum


class VegaAlignment(Enum):
    LEFT = "start"
    CENTER = "middle"
    RIGHT = "end"

    @classmethod
    def from_matplotlib(cls, alignment: str) -> str:
        """Convert Matplotlib horizontal alignment to Vega alignment."""
        mapping = {"left": cls.LEFT, "center": cls.CENTER, "right": cls.RIGHT}
        return mapping.get(alignment, cls.CENTER).value


def _count_trailing(num: float) -> int | None:
    str_num = str(num)
    if "." in str_num:
        return len(str_num.split(".")[1])
    return 0


def enforce_common_decimal_format(values: list[float]) -> list[float]:
    most_common_decimal = Counter([_count_trailing(num) for num in values]).most_common(1)[0][0]
    return [round(num, most_common_decimal) for num in values]


def strip_call(s: str) -> str:
    """Strip leading digit and underscore from call name."""
    return re.sub(r"^\d+_", "", s)



class colors(object):

    starlight = "#00B0F0"
    white = "#FFFFFF"
    blue = "#00B0F0"
    green = "#00B050"
    gold = "#F6CA4C"
    orange = "ED9D50"
    purple = "#7030A0"
    pink = "#E23192"
    black = "#000000"
    red = "#C00000"


class ColorUtils(object):

    @staticmethod
    def as_rgb(as_hex, alpha=None):
        """Converts from #FFFFFF => (255, 255, 255) with optional alpha between 0 and 1 => (255)"""
        rgba = as_hex.lstrip('#')
        rgba = tuple(int(rgba[i:i+2], 16) for i in (0,2,4))
        if alpha is not None: rgba += (int(255*(alpha)),)
        return rgba
COLORS = {
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "brown": (120, 75, 45),
    "beige": (225, 200, 165),
    "red": (200, 30, 30),
    "orange": (230, 130, 30),
    "yellow": (235, 210, 50),
    "green": (40, 150, 70),
    "blue": (40, 90, 190),
    "navy": (25, 35, 90),
    "purple": (120, 50, 160),
    "pink": (230, 130, 175),
}


class ColorNamer:
    """Snaps an RGB triple to the nearest allowed taxonomy color option."""

    def nearest(self, rgb, options: list[str]) -> str:
        known = [o for o in options if o in COLORS]
        if not known:
            return options[0] if options else "unknown"
        return min(known, key=lambda name: self._distance(rgb, COLORS[name]))

    def _distance(self, a, b) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

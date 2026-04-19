# examples/composite_overlay.plugins.py
#
# LOCAL PLUGIN — auto-loaded when running composite_overlay.pxl
#
# This file is discovered automatically by PixLang because it sits
# alongside composite_overlay.pxl and shares the same stem name.
#
# Run: pixlang run examples/composite_overlay.pxl --verbose
# You should see this plugin listed in the plugin section of verbose output.
#
# Adds: STAMP_VERSION — overlays a version watermark in the bottom-right corner.

import cv2
import numpy as np


def register(registry):
    """Called by PixLang's plugin loader with the live registry instance."""

    @registry.register("STAMP_VERSION", source="local:composite_overlay")
    def cmd_stamp_version(image, version_text: str = "v0.2"):
        """STAMP_VERSION ["text"] — stamp a small version label bottom-right.

        Watermarks the image with a semi-transparent dark pill containing
        the version string. Useful for traceability in inspection outputs.

        Examples:
            STAMP_VERSION
            STAMP_VERSION "build-42"
        """
        from pixlang.commands.builtin import _require_image, _unwrap, _wrap
        _require_image("STAMP_VERSION", image)
        img, meta = _unwrap(image)

        if len(img.shape) == 2:
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            canvas = img.copy()

        h, w = canvas.shape[:2]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        scale      = 0.5
        thickness  = 1
        label      = f" {version_text} "

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

        # Draw dark background pill
        pad    = 6
        rx1    = w - tw - pad * 2 - 8
        ry1    = h - th - pad * 2 - 8
        rx2    = w - 8
        ry2    = h - 8

        overlay = canvas.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # Draw text
        tx = rx1 + pad
        ty = ry2 - pad - baseline
        cv2.putText(canvas, label, (tx, ty), font, scale, (180, 230, 255), thickness, cv2.LINE_AA)

        result = _wrap(canvas)
        result.update(meta)
        return result

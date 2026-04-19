# plugins_example/pixlang_denoise/plugin.py
"""
pixlang-denoise
═══════════════
An example third-party PixLang plugin that adds advanced denoising commands.

This demonstrates the entry-point plugin pattern.

To install and activate:
    pip install -e plugins_example/pixlang_denoise/

Once installed, `pixlang commands` will show these commands under
source "pixlang-denoise", and they work in any .pxl pipeline automatically.

New commands:
    NLM_DENOISE [h] [template_size] [search_size]
        Non-local means denoising — much stronger than Gaussian blur.
        Preserves edges while removing noise.

    BILATERAL [d] [sigma_color] [sigma_space]
        Bilateral filter — edge-preserving smoothing.
        Blurs within similar-colour regions only.
"""
import cv2
import numpy as np


def register(registry):
    """
    Entry point called by PixLang's plugin loader.
    Receives the live CommandRegistry instance.
    """

    @registry.register("NLM_DENOISE", source="pixlang-denoise")
    def cmd_nlm_denoise(image, h: float = 10, template_size: int = 7, search_size: int = 21):
        """NLM_DENOISE [h] [template] [search] — Non-local means denoising.

        Superior noise removal that preserves fine detail better than Gaussian.
        Computationally heavier — use for final pre-processing stages.

        Parameters:
            h             — filter strength (higher = more noise removed, less detail)
            template_size — size of the patch used for comparison (odd, default 7)
            search_size   — size of the window searched for similar patches (odd, default 21)

        Examples:
            NLM_DENOISE
            NLM_DENOISE 15
            NLM_DENOISE 10 7 21
        """
        from pixlang.commands.builtin import _require_image, _unwrap, _wrap
        _require_image("NLM_DENOISE", image)
        img, meta = _unwrap(image)

        # NLM works on grayscale or colour; handle both
        if len(img.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(
                img,
                None,
                h=float(h),
                templateWindowSize=int(template_size),
                searchWindowSize=int(search_size),
            )
        else:
            denoised = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=float(h),
                hColor=float(h),
                templateWindowSize=int(template_size),
                searchWindowSize=int(search_size),
            )

        result = _wrap(denoised)
        result.update(meta)
        return result

    @registry.register("BILATERAL", source="pixlang-denoise")
    def cmd_bilateral(image, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        """BILATERAL [d] [sigma_color] [sigma_space] — edge-preserving bilateral filter.

        Smooths textures while keeping hard edges sharp. Ideal as a pre-processing
        step before edge detection or feature extraction.

        Parameters:
            d           — diameter of each pixel neighbourhood (default 9)
            sigma_color — range sigma: larger = farther colors blended (default 75)
            sigma_space — spatial sigma: larger = farther pixels blended (default 75)

        Examples:
            BILATERAL
            BILATERAL 9 75 75
            BILATERAL 15 100 100
        """
        from pixlang.commands.builtin import _require_image, _unwrap, _wrap
        _require_image("BILATERAL", image)
        img, meta = _unwrap(image)

        filtered = cv2.bilateralFilter(
            img,
            int(d),
            float(sigma_color),
            float(sigma_space),
        )

        result = _wrap(filtered)
        result.update(meta)
        return result

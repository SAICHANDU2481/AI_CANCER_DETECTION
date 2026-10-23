import os
import argparse
import logging
from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter
import yaml
from PIL import Image

from .normalization import normalize_intensity

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("WienerFilter")


def local_adaptive_wiener(
    image: np.ndarray,
    window_size: Tuple[int, int] = (5, 5),
    noise_variance: Optional[float] = None
) -> np.ndarray:
    """
    Applies adaptive 2D spatial Wiener filtering based on local statistical variance.
    Mathematical formulation:
        f_hat(x, y) = mu(x, y) + max(0, (sigma^2(x, y) - nu^2) / sigma^2(x, y)) * (g(x, y) - mu(x, y))
    where:
        - g(x, y) is the observed degraded/noisy mammogram
        - mu(x, y) is local neighborhood mean
        - sigma^2(x, y) is local neighborhood variance
        - nu^2 is the estimated additive noise variance

    Args:
        image: 2D grayscale array with values in [0, 1].
        window_size: (height, width) kernel window.
        noise_variance: Pre-specified noise variance; if None, estimated automatically as average local variance.

    Returns:
        Denoised 2D array in [0, 1].
    """
    img = image.astype(np.float64)
    
    # Local mean
    local_mean = uniform_filter(img, size=window_size)
    
    # Local mean of squared image
    local_mean_sq = uniform_filter(img ** 2, size=window_size)
    
    # Local variance: Var(X) = E[X^2] - (E[X])^2
    local_var = np.maximum(0.0, local_mean_sq - (local_mean ** 2))
    
    if noise_variance is None:
        # Estimate overall noise variance as the mean of local variances
        noise_variance = float(np.mean(local_var))
    
    # Calculate Wiener gain factor: max(0, (Var - NoiseVar) / Var)
    gain = np.where(
        local_var > 1e-10,
        np.maximum(0.0, (local_var - noise_variance) / (local_var + 1e-10)),
        0.0
    )
    
    # Reconstructed signal
    filtered = local_mean + gain * (img - local_mean)
    return normalize_intensity(np.clip(filtered, 0.0, 1.0))


def frequency_domain_wiener(
    image: np.ndarray,
    psf_kernel_size: int = 3,
    nsr: float = 0.01
) -> np.ndarray:
    """
    Applies frequency domain Wiener deconvolution filter:
        H_w(u, v) = H*(u, v) / (|H(u, v)|^2 + NSR)

    Args:
        image: 2D grayscale array [0, 1].
        psf_kernel_size: Size of point-spread function (blur kernel).
        nsr: Noise-to-Signal Power Ratio (regularization constant K).

    Returns:
        Restored 2D array in [0, 1].
    """
    img = image.astype(np.float64)
    h, w = img.shape
    
    # Define Point Spread Function (Gaussian or uniform blur kernel)
    psf = np.ones((psf_kernel_size, psf_kernel_size)) / (psf_kernel_size ** 2)
    
    # Pad PSF to image size
    psf_padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    
    # Shift PSF center
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)
    
    # Fast Fourier Transforms
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)
    
    # Wiener Transfer Function
    H_wiener = H_conj / (np.abs(H) ** 2 + nsr)
    
    # Apply filter in frequency domain and inverse FFT
    F_hat = G * H_wiener
    restored = np.real(np.fft.ifft2(F_hat))
    
    return normalize_intensity(np.clip(restored, 0.0, 1.0))


def apply_wiener_filter(
    image: np.ndarray,
    window_size: Union[int, Tuple[int, int]] = (5, 5),
    noise_variance: Optional[float] = None,
    mode: str = "scipy"
) -> np.ndarray:
    """
    High-level entry point for Wiener filtering a mammogram image.

    Args:
        image: 2D numpy array or 3D single-channel image.
        window_size: Filter window size (e.g. (5, 5) or 5).
        noise_variance: Predefined noise variance or None for auto.
        mode: "scipy" (signal.wiener), "local" (local adaptive), or "frequency".

    Returns:
        Wiener-filtered 2D numpy array [0.0, 1.0].
    """
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image.squeeze(2)
        else:
            # Convert RGB to Luminance Grayscale
            image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
            
    norm_img = normalize_intensity(image, 0.0, 1.0)
    
    if isinstance(window_size, int):
        mysize = (window_size, window_size)
    else:
        mysize = tuple(window_size)

    if mode == "scipy":
        try:
            filtered = signal.wiener(norm_img, mysize=mysize, noise=noise_variance)
            return normalize_intensity(np.nan_to_num(filtered, nan=0.0))
        except Exception as e:
            logger.warning(f"scipy.signal.wiener failed ({e}), falling back to local adaptive Wiener.")
            return local_adaptive_wiener(norm_img, window_size=mysize, noise_variance=noise_variance)
    elif mode == "frequency":
        return frequency_domain_wiener(norm_img, psf_kernel_size=mysize[0], nsr=0.02)
    else:
        return local_adaptive_wiener(norm_img, window_size=mysize, noise_variance=noise_variance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Wiener filtering on synthetic or real mammogram.")
    parser.add_argument("--image", type=str, default=None, help="Path to mammogram image")
    parser.add_argument("--window", type=int, default=5, help="Wiener filter window size")
    parser.add_argument("--out", type=str, default="results/preprocessing/wiener_demo.png", help="Output path")
    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        img_arr = np.array(Image.open(args.image))
    else:
        logger.info("Generating synthetic mammographic tissue phantom for Wiener filter demonstration...")
        # Create synthetic mammogram tissue phantom with simulated speckle noise
        np.random.seed(42)
        y, x = np.mgrid[:512, :512]
        # Breast contour
        breast_mask = ((x - 200)**2 + (y - 256)**2) < 220**2
        phantom = np.zeros((512, 512), dtype=np.float32)
        phantom[breast_mask] = 0.45 + 0.2 * np.sin(x[breast_mask] / 30.0) + 0.15 * np.cos(y[breast_mask] / 40.0)
        # Add simulated lesion
        lesion_mask = ((x - 280)**2 + (y - 220)**2) < 28**2
        phantom[lesion_mask] += 0.35
        # Add speckle and Gaussian quantum mottle noise
        noise = np.random.normal(0, 0.06, (512, 512)) + np.random.exponential(0.03, (512, 512))
        img_arr = np.clip(phantom + noise * breast_mask, 0.0, 1.0)

    filtered = apply_wiener_filter(img_arr, window_size=(args.window, args.window))
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_img = Image.fromarray((filtered * 255).astype(np.uint8))
    out_img.save(args.out)
    logger.info(f"Wiener filtering completed successfully. Output saved to: {args.out}")

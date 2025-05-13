import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt



def overlay_centered(background: np.ndarray,
                     foreground: np.ndarray,
                     center: tuple[int, int]) -> np.ndarray:
    """
    Place the foreground image onto the background, centered at the given coordinate.

    Parameters:
    background (np.ndarray): 2D array representing the background image.
    foreground (np.ndarray): 2D array representing the image to overlay.
    center (tuple[int, int]): (row, col) in background where the center of foreground should be placed.

    Returns:
    np.ndarray: New image with foreground overlayed on background.
    """
    # Copy background to avoid modifying original
    result = background.copy()

    bg_h, bg_w = background.shape
    fg_h, fg_w = foreground.shape
    cy, cx = center

    # Compute top-left corner of where to place foreground
    top = cy - fg_h // 2
    left = cx - fg_w // 2

    # Determine overlapping region coordinates
    fg_y_start = max(0, -top)
    fg_x_start = max(0, -left)
    fg_y_end = min(fg_h, bg_h - top)
    fg_x_end = min(fg_w, bg_w - left)

    bg_y_start = max(0, top)
    bg_x_start = max(0, left)
    bg_y_end = bg_y_start + (fg_y_end - fg_y_start)
    bg_x_end = bg_x_start + (fg_x_end - fg_x_start)

    # Overlay
    result[bg_y_start:bg_y_end, bg_x_start:bg_x_end] = \
        foreground[fg_y_start:fg_y_end, fg_x_start:fg_x_end]

    return result

def rotate_image(image: np.ndarray,
                 angle: float,
                 reshape: bool = False,
                 order: int = 1,
                 mode: str = 'constant',
                 cval: float = 0.0,
                 center: tuple[float, float] | None = None) -> np.ndarray:
    """
    Rotate a 2D image by a specified angle around a given center.

    Parameters:
    image (np.ndarray): 2D array representing the input image.
    angle (float): Rotation angle in degrees, positive values rotate counter-clockwise.
    reshape (bool): If True, output shape is adapted to contain the whole rotated image.
                    If False, the output has the same shape as the input (parts may be clipped).
    order (int): The order of the spline interpolation (0=nearest, 1=bilinear, etc.).
    mode (str): Points outside the boundaries are filled according to this mode (e.g. 'constant', 'wrap').
    cval (float): Value to fill past edges if mode='constant'.
    center (tuple[float, float], optional): (row, col) coordinates of rotation center.
                                         If None, uses the image center.

    Returns:
    np.ndarray: The rotated image as a 2D array.
    """
    if center is None:
        # Default center is the geometric center of the image
        center = ((image.shape[0] - 1) / 2.0, (image.shape[1] - 1) / 2.0)
    # ndimage.rotate rotates around center of the array by default, but only if reshape=False.
    # To rotate around an arbitrary center, we first translate, rotate, then translate back.

    # Create an affine transform matrix for rotation around center
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    # Affine matrix components
    a = cos_a
    b = -sin_a
    d = sin_a
    e = cos_a

    # Compute the offset to keep the center fixed
    cy, cx = center
    offset = np.array([cy - a * cy - b * cx,
                       cx - d * cy - e * cx])

    # Apply affine transformation
    rotated = ndimage.affine_transform(
        image,
        matrix=np.array([[a, b], [d, e]]),
        offset=offset,
        output_shape=None if reshape else image.shape,
        order=order,
        mode=mode,
        cval=cval
    )
    return rotated

def generate_corkscrew_simple(radius,  N=4, center=(0, 0), fact=(5/8)): 
    x = np.linspace(-radius, radius, radius*2)
    y = np.linspace(-radius, radius, radius*2)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    Phi = np.arctan2(Y,X)
        
    answer = ((np.ceil((N)*((R/radius)**(1/fact))))*Phi)%(2*np.pi) 
        
    return answer

def antialias_gaussian(image: np.ndarray,
                       sigma: float = 1.0,
                       mode: str = 'reflect',
                       truncate: float = 4.0) -> np.ndarray:
    """
    Apply a Gaussian filter to perform antialiasing.

    Parameters:
    image (np.ndarray): 2D array input image.
    sigma (float): Standard deviation for Gaussian kernel.
    mode (str): How to handle borders.
    truncate (float): Truncate the filter at this many standard deviations.

    Returns:
    np.ndarray: Smoothed image reducing aliasing artifacts.
    """
    return ndimage.gaussian_filter(image, sigma=sigma, mode=mode, truncate=truncate)



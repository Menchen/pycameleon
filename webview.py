#!/usr/bin/env python

import argparse

import time
import cv2
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import numpy as np
import pycameleon
import os
from numba import jit, njit
import re
from pathlib import Path


def create_custom_colormap(stops, gamma=1.0, size=256):
    """
    Create a custom nonlinear color gradient.

    stops: list of (position, (R, G, B)) where position in [0,1]
    gamma: controls nonlinearity (gamma correction)
    size:  number of discrete steps in the lookup table
    """
    # Ensure stops are sorted
    stops = sorted(stops, key=lambda s: s[0])
    gradient = np.zeros((size, 3), dtype=np.float32)

    # Generate color map by segment interpolation
    for i in range(len(stops) - 1):
        x1, c1 = stops[i]
        x2, c2 = stops[i + 1]
        idx1, idx2 = int(x1 * (size - 1)), int(x2 * (size - 1))
        for j, k in enumerate(np.linspace(0, 1, idx2 - idx1 + 1)):
            k_gamma = k**gamma
            gradient[idx1 + j] = (1 - k_gamma) * np.array(c1) + k_gamma * np.array(c2)

    return gradient / 255.0  # normalize for OpenCV use


def apply_colormap_custom(gray, colormap):
    """
    Apply a custom colormap (LUT) to a grayscale image.
    """
    # Map normalized NDVI to LUT indices
    lut_size = len(colormap)
    indices = gray * (lut_size - 1)
    # Get integer part and fractional part
    idx0 = np.floor(indices).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, lut_size - 1)
    frac = (indices - idx0)[..., None]  # keep channel dimension

    # Linear interpolation between LUT entries
    rgb = (1 - frac) * colormap[idx0] + frac * colormap[idx1]
    return rgb


def sanitize_filename(name: str) -> str:
    """Convert any string into a safe filename."""
    # Replace invalid filename characters with underscores
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    # Optional: trim spaces and limit length
    safe = safe.strip().replace(" ", "_")[:255]
    return safe or "untitled"


def read_xml_to_string(name: str, directory: str = ".") -> str:
    """Sanitize filename and write content safely."""
    safe_name = sanitize_filename(name)
    path = Path(directory) / f"{safe_name}.xml"
    xml = path.read_text(encoding="utf-8")
    return xml


lambdaOrder = [0, 7, 1, 8, 2, 4, 3, 9, 1, 8, 0, 7, 3, 6, 2, 5]
crossTalkCoef = [
    0.9948,
    -0.0966,
    -0.0005,
    -0.0100,
    -0.0167,
    -0.0251,
    -0.0384,
    -0.1422,
    -0.2110,
    -0.0842,
    0.0788,
    1.0723,
    0.0840,
    0.0103,
    0.0147,
    0.0094,
    0.0087,
    -0.0138,
    -0.0265,
    0.0141,
    0.0510,
    0.1239,
    0.9772,
    -0.0789,
    0.0126,
    0.0100,
    -0.0002,
    -0.0655,
    -0.0815,
    0.0115,
    0.0190,
    0.0204,
    0.0352,
    1.0717,
    0.1085,
    0.0058,
    -0.0100,
    -0.0465,
    -0.0488,
    -0.0481,
    0.0117,
    0.0147,
    0.0211,
    0.1238,
    0.9482,
    -0.0459,
    -0.0290,
    -0.0393,
    -0.0286,
    -0.0789,
    -0.0040,
    -0.0031,
    -0.0104,
    -0.0009,
    0.0279,
    1.1097,
    -0.0261,
    -0.0427,
    -0.0331,
    -0.0082,
    -0.0108,
    -0.0099,
    -0.0207,
    -0.0219,
    0.0067,
    0.0175,
    1.1393,
    -0.0461,
    -0.0593,
    -0.0077,
    -0.0486,
    -0.0352,
    -0.0236,
    -0.0277,
    -0.0450,
    -0.0463,
    -0.0042,
    1.4309,
    -0.0441,
    -0.1777,
    -0.0640,
    -0.0615,
    -0.0310,
    -0.0344,
    -0.0429,
    -0.0345,
    -0.0408,
    0.0162,
    1.5354,
    -0.3215,
    -0.0280,
    -0.0250,
    -0.0313,
    -0.0319,
    -0.0139,
    -0.0006,
    0.0008,
    -0.0512,
    -0.0025,
    1.7008,
]
crossTalkCoef = np.array(crossTalkCoef).reshape((10, 10))

# initialize
elementCount = np.zeros(max(lambdaOrder) + 1, dtype=int)
for x in range(0, 16):
    elementCount[lambdaOrder[x]] += 1


@njit(nopython=True, parallel=True, fastmath=True, cache=True)
def extractChannel(image, lambdaOrder):
    offSetX = 0
    offSetY = 0
    tVal = np.empty(16, dtype=np.int32)
    # image = np.random.randint(0,255,(2048,2048))
    imgCanal = np.zeros((512, 512, max(lambdaOrder) + 1), dtype=np.int32)
    for i in range(512):
        for j in range(512):
            tVal[0] = image[i * 4 + offSetX, j * 4 + offSetY]
            tVal[1] = image[i * 4 + offSetX + 1, j * 4 + offSetY]
            tVal[2] = image[i * 4 + offSetX + 2, j * 4 + offSetY]
            tVal[3] = image[i * 4 + offSetX + 3, j * 4 + offSetY]
            tVal[4] = image[i * 4 + offSetX, j * 4 + offSetY + 1]
            tVal[5] = image[i * 4 + offSetX + 1, j * 4 + offSetY + 1]
            tVal[6] = image[i * 4 + offSetX + 2, j * 4 + offSetY + 1]
            tVal[7] = image[i * 4 + offSetX + 3, j * 4 + offSetY + 1]
            tVal[8] = image[i * 4 + offSetX, j * 4 + offSetY + 2]
            tVal[9] = image[i * 4 + offSetX + 1, j * 4 + offSetY + 2]
            tVal[10] = image[i * 4 + offSetX + 2, j * 4 + offSetY + 2]
            tVal[11] = image[i * 4 + offSetX + 3, j * 4 + offSetY + 2]
            tVal[12] = image[i * 4 + offSetX, j * 4 + offSetY + 3]
            tVal[13] = image[i * 4 + offSetX + 1, j * 4 + offSetY + 3]
            tVal[14] = image[i * 4 + offSetX + 2, j * 4 + offSetY + 3]
            tVal[15] = image[i * 4 + offSetX + 3, j * 4 + offSetY + 3]

            for w in range(len(lambdaOrder)):
                imgCanal[i, j, lambdaOrder[w]] += tVal[w] / elementCount[lambdaOrder[w]]

    return imgCanal


@njit(nopython=True, fastmath=True, cache=True)
def applyCrossTalk(img, coef):
    # normalize img
    norm_coef = np.empty(img.shape[2])
    flatten_view = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    for i in range(len(norm_coef)):
        norm_coef[i] = 1 / flatten_view[:, i].max()
    img = img * norm_coef.reshape((1, 1, -1))
    result = np.empty(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = img[i, j] @ crossTalkCoef
    return result


@njit(nopython=True, parallel=True, fastmath=True, cache=True)
def calculate_ndvi(img, red_index, nir_index):
    red = img[:, :, red_index].astype(np.float32)
    nir = img[:, :, nir_index].astype(np.float32)
    ndvi = (nir - red) / (nir + red)
    nmin = ndvi.min()
    nmax = ndvi.max()
    # ndvi = (ndvi - nmin)/(nmax-nmin)
    ndvi = (ndvi + np.ones(ndvi.shape)) / 2
    return np.append(img, ndvi[:, :, np.newaxis], axis=2)


@njit(nopython=True, parallel=True, fastmath=True, cache=True)
def calculate_ndvi2(img, red_index, nir_index):
    epsilon = 1e-6
    red = img[:, :, red_index].astype(np.float32) / 255.0
    nir = img[:, :, nir_index].astype(np.float32) / 255.0
    # red = cv2.medianBlur(red, 3)
    # nir = cv2.medianBlur(nir, 3)

    ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red + epsilon))
    low_end = -0.1
    high_end = 0.3
    ndvi_clipped = np.clip(ndvi, low_end, high_end)
    # ndvi = (ndvi + 1) / 2.0
    ndvi_normalized = (ndvi_clipped - low_end) / (high_end - low_end)
    # mapped = apply_colormap_custom(ndvi_normalized, colormap)
    ndvi = (ndvi_normalized * 255).astype(np.uint8)
    return ndvi


# stops = [
#     (0.00, (0, 0, 128)),  # Deep blue  → NDVI -1.0 (water)
#     (0.10, (0, 0, 255)),  # Blue       → NDVI -0.8 (wet/non-veg)
#     (0.25, (150, 75, 0)),  # Brown      → NDVI -0.5 to 0.0 (bare soil)
#     (0.35, (255, 255, 0)),  # Yellow     → NDVI +0.1 (dry vegetation)
#     (0.55, (144, 238, 144)),  # Light green → NDVI +0.2 (start of vegetation)
#     (0.65, (0, 200, 0)),  # Green      → NDVI +0.5 (moderate vegetation)
#     (1.00, (0, 100, 0)),  # Dark green → NDVI +1.0 (dense vegetation)
# ]
# stops = [
#     (0.00, (0, 0, 255)),  # blue (low NDVI)
#     (0.25, (0, 255, 255)),  # cyan
#     (0.50, (0, 255, 0)),  # green
#     (0.75, (255, 255, 0)),  # yellow
#     (1.00, (255, 0, 0)),  # red (dense vegetation)
# ]
stops = [
    (0.00, (0, 0, 0)),  # blue (low NDVI)
    (1.00, (255, 255, 255)),  # red (dense vegetation)
]


# Create a nonlinear colormap (gamma < 1 = faster low values, > 1 = slower)
colormap = create_custom_colormap(stops, gamma=2.4)


def run_canny(num_frames: int | None) -> None:
    # Create a new video capture

    frame_nr = 0
    cam = pycameleon.enumerate_cameras()[0]
    try:
        cam.close()
        cam.open()
        cam.load_context_xml(read_xml_to_string(str(cam)))
        cam.execute_command("DeviceReset")
        cam.close()
        print(1)
        time.sleep(5)

        cam = pycameleon.enumerate_cameras()[0]
        cam.open()
        print(2)
        # cam.load_context()
        cam.load_context_xml(read_xml_to_string(str(cam)))
        print(3)
        # print(cam.isdone_command("DeviceReset"))
        payload = cam.start_streaming(1)
        print(4)

        if num_frames and frame_nr >= num_frames:
            return

        while True:
            # Read the frame
            img = cam.receive(payload)
            rs = np.hstack([img[:, :, np.newaxis]])
            rs = extractChannel(img, lambdaOrder)
            rs = applyCrossTalk(rs, crossTalkCoef)
            ndvi = calculate_ndvi2(rs, 5, 9)

            # Get the current frame time. On some platforms it always returns zero.
            # frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_time_ms = 0
            if frame_time_ms != 0:
                rr.set_time("frame_time", duration=1e-3 * frame_time_ms)

            rr.set_time("frame_nr", sequence=frame_nr)
            frame_nr += 1

            # Log the original image
            # rr.log("image/rgb", rr.Image(img, color_model="BGR"))
            rr.log("image/gray", rr.Image(ndvi))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cam.close()
        print("Camera closed")
        exit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streams a local system camera and runs the canny edge detector."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Which camera device to use. (Passed to `cv2.VideoCapture()`)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None, help="The number of frames to log"
    )

    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(
        args,
        "rerun_example_live_camera_edge_detection",
        default_blueprint=rrb.Vertical(
            rrb.Spatial2DView(origin="/image/gray", name="Video"),
        ),
    )

    run_canny(args.num_frames)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import time
    import cv2
    import rerun as rr  # pip install rerun-sdk
    import rerun.blueprint as rrb
    import numpy as np
    import pycameleon
    import os
    from numba import jit, njit, prange
    import re
    from pathlib import Path
    import timeit
    return Path, mo, njit, np, prange, pycameleon, re, rr, rrb, time


@app.cell(hide_code=True)
def _(Path, np, re):
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
    return (read_xml_to_string,)


@app.cell(hide_code=True)
def _(np):

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


    lambdaOrder = np.array([0, 7, 1, 8, 2, 4, 3, 9, 1, 8, 0, 7, 3, 6, 2, 5])
    elementCount = np.zeros(max(lambdaOrder) + 1, dtype=int)
    for x in range(0, 16):
        elementCount[lambdaOrder[x]] += 1
    return crossTalkCoef, elementCount, lambdaOrder


@app.cell(hide_code=True)
def _(njit, np, prange):

    @njit(parallel=True, fastmath=True)
    def extractChannel_block_optimized(image, lambdaOrder, elementCount, H=512, W=512, block_x=4, block_y=4, is_lambda_order_column_major=True):
        assert image.shape[0] // block_x == W, "Incompatible lambdaOrder W size"
        assert image.shape[1] // block_y == H, "Incompatible lambdaOrder H size"

        numChannels = np.max(lambdaOrder) + 1
        imgCanal = np.zeros((H, W, numChannels), dtype=np.float32)

        # convert from column major to row major
        if is_lambda_order_column_major:
            own_lambda = lambdaOrder.reshape((block_x, block_y)).T.flatten()
        else:
            own_lambda = lambdaOrder

        block_size = block_x * block_y

        for i in prange(H):
            for j in range(W):
                base_x = i * block_x
                base_y = j * block_y
                # Flatten block
                blockVals = image[base_x:base_x + block_x, base_y:base_y + block_y].ravel()
                # Use a direct loop but precompute division
                for w in range(block_size):
                    ch = own_lambda[w]
                    imgCanal[i, j, ch] += blockVals[w] / elementCount[ch]

        return imgCanal.astype(np.uint8)
    return


@app.cell(hide_code=True)
def _(njit, np):
    @njit(parallel=True, fastmath=True)
    def extractChannel(image, lambdaOrder, elementCount):
        offSetX = 0
        offSetY = 0
        tVal = np.empty(16, dtype=np.int32)
        # image = np.random.randint(0,255,(2048,2048))
        imgCanal = np.zeros((512, 512, max(lambdaOrder) + 1), dtype=np.float32)
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

        return imgCanal.astype(np.uint8)
    return


@app.cell(hide_code=True)
def _(njit, np, prange):
    @njit(parallel=True, fastmath=True)
    def extractChannel_block(image, lambdaOrder,elementCount,H=512,W=512,block_x=4,block_y=4,is_lambda_order_column_major=True):
        assert image.shape[0] // block_x == W, "Incompatible lambdaOrder W size"
        assert image.shape[1] // block_y == H, "Incompatible lambdaOrder H size"
        numChannels = np.max(lambdaOrder) + 1
        imgCanal = np.zeros((H, W, numChannels), dtype=np.float32)

        # convert from column major to row major
        if is_lambda_order_column_major:
            own_lambda = lambdaOrder.reshape((block_x,block_y)).T.flatten()
        else:
            own_lambda = lambdaOrder

        inv_elementCount = 1.0 / elementCount
        for i in prange(H):
            for j in range(W):
                base_x = i * block_x
                base_y = j * block_y
                # Grab block directly
                blockVals = image[base_x:base_x + block_x, base_y:base_y + block_y].ravel()

                for w in range(len(own_lambda)):
                    ch = own_lambda[w]
                    imgCanal[i, j, ch] += float(blockVals[w]) * inv_elementCount[ch]

        return imgCanal.astype(np.uint8)

    # Non parallel 
    # @njit(fastmath=True)
    # def applyCrossTalk(img, coef):
    #     # normalize img
    #     norm_coef = np.empty(img.shape[2])
    #     flatten_view = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    #     for i in range(len(norm_coef)):
    #         norm_coef[i] = 1 / flatten_view[:, i].max()
    #     img = img * norm_coef.reshape((1, 1, -1))
    #     result = np.empty(img.shape, dtype=np.float32)
    #     for i in range(img.shape[0]):
    #         for j in range(img.shape[1]):
    #             result[i, j] = img[i, j] @ crossTalkCoef
    #     return result
    return (extractChannel_block,)


@app.cell
def _(njit, np, prange):
    @njit(fastmath=True, parallel=True)
    def apply_cross_talk(img, coef):
        """
        Apply cross-talk correction to a 3D image array.
        img: np.ndarray, shape (H, W, C)
        coef: np.ndarray, shape (C, C) cross-talk correction matrix
        """
        H, W, C = img.shape
        result = np.empty((H, W, C), dtype=np.float32)

        # Normalize per channel
        norm_coef = np.empty(C, dtype=np.float32)
        for c in range(C):
            max_val = 0.0
            for i in range(H):
                for j in range(W):
                    val = img[i, j, c]
                    if val > max_val:
                        max_val = val
            if max_val > 0.0:
                norm_coef[c] = 1.0 / max_val
            else:
                norm_coef[c] = 1.0

        # Apply normalization and cross-talk correction
        for i in prange(H):
            for j in range(W):
                tmp = np.empty(C, dtype=np.float32)
                # Normalize
                for c in range(C):
                    tmp[c] = img[i, j, c] * norm_coef[c]

                # Apply cross-talk (manual dot product for Numba speed)
                for c in range(C):
                    s = 0.0
                    for k in range(C):
                        s += tmp[k] * coef[k, c]
                    result[i, j, c] = s
        return result
    return (apply_cross_talk,)


@app.cell
def _(njit, np):
    @njit(parallel=True, fastmath=True)
    def calculate_ndvi2(img, red_index, nir_index,low_end=-0.1,high_end=0.3,epsilon=1e-6):
        epsilon = 1e-6
        red = img[:, :, red_index].astype(np.float32) / 255.0
        nir = img[:, :, nir_index].astype(np.float32) / 255.0

        ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red + epsilon))
        ndvi_clipped = np.clip(ndvi, low_end, high_end)
        ndvi_normalized = (ndvi_clipped - low_end) / (high_end - low_end)
        ndvi = (ndvi_normalized * 255).astype(np.uint8)
        return ndvi
    return (calculate_ndvi2,)


@app.cell
def _(elementCount, extractChannel_block, lambdaOrder, np):
    test_data = np.random.randint(0,255,size=(2048,2048))
    extracted_test_data = extractChannel_block(test_data,lambdaOrder,elementCount)
    return


@app.cell
def _(
    apply_cross_talk,
    calculate_ndvi2,
    crossTalkCoef,
    elementCount,
    extractChannel_block,
    lambdaOrder,
    np,
    pycameleon,
    read_xml_to_string,
    rr,
    time,
):
    def run_viewer(num_frames: int | None) -> None:
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
            rr.log("/info", rr.TextDocument("Notebook setup complete ✅"))
            if num_frames and frame_nr >= num_frames:
                return

            while True:
                # Read the frame
                img = cam.receive(payload)
                rs = np.hstack([img[:, :, np.newaxis]])
                rs = extractChannel_block(img, lambdaOrder,elementCount)
                rs = apply_cross_talk(rs, crossTalkCoef)
                ndvi = calculate_ndvi2(rs, 5, 9)

                # Get the current frame time. On some platforms it always returns zero.
                # frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_time_ms = 0
                if frame_time_ms != 0:
                    rr.set_time("frame_time", duration=1e-3 * frame_time_ms)

                rr.set_time("frame_nr", sequence=frame_nr)
                frame_nr += 1

                for i in range(rs.shape[2]):
                    rr.log(f"image/{i}", rr.Image(ndvi))
                # Log the original image
                # rr.log("image/rgb", rr.Image(img, color_model="BGR"))
                rr.log("image/ndvi", rr.Image(ndvi))
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cam.close()
            print("Camera closed")
    return (run_viewer,)


@app.cell
def _(mo):
    get_server,set_server = mo.state(False)
    return get_server, set_server


@app.cell
def _(get_server, mo, rr, rrb, run_button, set_server):
    mo.stop(not run_button.value or get_server(), mo.md("Waiting to start"))

    def start_server():
        rr.init("WebViewer", spawn=False)
        my_blueprint = rrb.Horizontal(
            contents=[rrb.Spatial2DView(origin="/image/ndvi", name="NDVI"),
            rrb.Grid(contents=[
                rrb.Spatial2DView(origin="/image/0", name="0"),
                rrb.Spatial2DView(origin="/image/1", name="1"),
                rrb.Spatial2DView(origin="/image/2", name="2"),
                rrb.Spatial2DView(origin="/image/3", name="3"),
                rrb.Spatial2DView(origin="/image/4", name="4"),
                rrb.Spatial2DView(origin="/image/5", name="5"),
                rrb.Spatial2DView(origin="/image/6", name="6"),
                rrb.Spatial2DView(origin="/image/7", name="7"),
                rrb.Spatial2DView(origin="/image/8", name="8"),
                rrb.Spatial2DView(origin="/image/9", name="9"),
            ])]
        )
        rr.send_blueprint(my_blueprint)
        #rr.notebook_show()

        # Start a gRPC server and use it as log sink.
        server_uri = rr.serve_grpc()
        viwer_port = 16675
        print(server_uri)
        # Connect the web viewer to the gRPC server and open it in the browser
        rr.serve_web_viewer(connect_to=server_uri,web_port=viwer_port)
        return server_uri

    start_server()
    print("Started server")
    set_server(True)
    return


@app.cell
def _(mo, run_button, run_viewer):
    mo.stop(not run_button.value, mo.md("Waiting to start"))

    run_viewer(None)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Start Streaming")
    run_button
    return (run_button,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

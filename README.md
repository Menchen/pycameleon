# Pycameleon

[![CI](https://github.com/Menchen/pycameleon/actions/workflows/maturin.yml/badge.svg)](https://github.com/Menchen/pycameleon/actions/workflows/maturin.yml) [![PyPI](https://img.shields.io/pypi/v/pycameleon)](https://pypi.org/project/pycameleon/) [![PyPI Downloads](https://img.shields.io/pypi/dm/pycameleon)](https://pypi.org/project/pycameleon/) [![PyPI](https://img.shields.io/pypi/v/pycameleon)](https://pypi.org/project/pycameleon/) [![Python](https://img.shields.io/pypi/pyversions/pycameleon)](https://pypi.org/project/pycameleon/)

A python binding for [cameleon](https://github.com/cameleon-rs/cameleon), a rust crate to interact with U3V (USB 3 Vision) camera that implement GenICam.

⚠️ **Work in Progress**  
This project is currently under active development. Features are being added, and APIs may change as we improve the design.  

We welcome **new contributors**! If you want to help, here’s how you can get involved:

Your contributions, whether code, documentation, or ideas are highly appreciated!

## Instalation

### PyPI (Recommended)

This package is available in [PyPI](https://pypi.org/project/pycameleon).

```bash
pip install pycameleon
```




### Build from source

Build dependencies:

 - Rust/Cargo toolchain installed
 - maturin `pip install maturin`/`pipx install maturin`
 - libusb header (See [cameleon FAQ](https://github.com/cameleon-rs/cameleon#faq))

First install `maturin` using `pip` or `pipx` or other package manager.

```bash
pip install maturin
```

Then run the following to install the package into the current environment.

```bash
maturin develop
```

### Linux

By default, many devices are only accessible to root and need udev rules for non root users to access.

First use `lsusb` to list vendor and product id `ID vendor_id:product_id`.

Then, create an udev rule file at `/etc/udev/rules.d/`, for example `/dev/udev/rules.d/99-u3v.rules`

Choose and change vendor and product id with your own.
```
# Example: USB with vendor Id 2676 and product id 5678
SUBSYSTEM=="usb", ATTRS{idVendor}=="2676", ATTR{idProduct}=="5678", MODE:="0666", TAG+="uaccess", TAG+="udev-acl"
# Example: USB with vendor Id 2676 with any product id
SUBSYSTEM=="usb", ATTRS{idVendor}=="1234", MODE:="0666", TAG+="uaccess", TAG+="udev-acl"
```

If the framerate is low, you may need to increase usbfs_memory_mb limit. By default, USB-FS on Linux systems only allows 16 MB of buffer memory for all USB devices. This is quite low for high-resolution image streaming. We recommend you to set the value to 1000MB. You could set the value as following:

```bash
echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb
```

### Windows

For Windows, [driver supported by libusb](https://github.com/libusb/libusb/wiki/Windows#driver-installation) like `WinUSB` is required.

In most case, [Zadig](http://zadig.akeo.ie/) is recommended as an Automated Driver Installer GUI application.

NOTE: Make sure to install a driver to a composite device not to its child devices.
To do this, you need to list all devices connected to the host computer with zadig like below.


### MacOS

MacOS should have out of box support after installing the pip package, for building from source, the libusb package is available in homebrew.



## Quick Usage


```python

import pycameleon


# List all available cameras
available_camera = pycameleon.enumerate_cameras()

# Get the first U3V camera
cam = available_camera[0]

# Open the device in USB level
cam.open()

# Load the GenApi context and return it as XML string
gen_api_context = cam.load_context_from_camera()

# As a alternative, you can load gen api context dumped previously as `.load_context_from_camera()` can fail if already streaming or in invalid state.
# cam.load_context_from_xml(read_xml_to_string(str(cam)))


# After loading the context, we can read/write nodes and execute commands.
# Depending on the camera model, some nodes may not be available.
# Check the dumped GenApi XML for available nodes.

# Reset the device, need to enumerate it again and open from scratch after this.
# cam.execute("DeviceReset")

# Tell the camera to use Mono8 pixel format
cam.write_enum_as_str("PixelFormat", "Mono8")

# Start streaming and prepare a payload buffer for 1 channel.
payload = cam.start_streaming(1)

# Receive a single image from camera.
img, info = cam.receive_raw(payload)

# `img`` is an numpy array with type uint8, it's the raw byte array received from camera.

# `info` contain image information such as width, height and pixel format.


# At the end, close the camera to release USB handler.
cam.close()







```



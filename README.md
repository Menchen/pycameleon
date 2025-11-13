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

## Linux

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

## Windows

For Windows, [driver supported by libusb](https://github.com/libusb/libusb/wiki/Windows#driver-installation) like `WinUSB` is required.

In most case, [Zadig](http://zadig.akeo.ie/) is recommended as an Automated Driver Installer GUI application.

NOTE: Make sure to install a driver to a composite device not to its child devices.
To do this, you need to list all devices connected to the host computer with zadig like below.


## MacOS

MacOS should have out of box support after installing the pip package, for building from source, the libusb package is available in homebrew.

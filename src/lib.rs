mod errors;

use cameleon::genapi::{DefaultGenApiCtxt, GenApiError, Node, ParamsCtxt};
use cameleon::u3v::ControlHandle;
use cameleon::{payload::PayloadReceiver, CameraInfo};
use errors::PycameleonResult;
use numpy::PyArray;
use numpy::PyArrayMethods;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyNone},
    IntoPyObjectExt,
};

// use cameleon::u3v::enumerate_cameras;

#[pyclass]
pub struct PyCameleonCamera(
    pub cameleon::Camera<cameleon::u3v::ControlHandle, cameleon::u3v::StreamHandle>,
);

#[pyclass]
pub struct PyPayloadReceiver(pub PayloadReceiver);

pub struct PyCameraInfo<'a>(pub &'a CameraInfo);

impl<'py> IntoPyObject<'py> for PyCameraInfo<'_> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        dict.set_item("model_name", &self.0.model_name)?;
        dict.set_item("vendor_name", &self.0.vendor_name)?;
        dict.set_item("serial_number", &self.0.serial_number)?;
        dict.into_bound_py_any(py)
    }
}

fn read_node<T, DownCastNode, DownCastFn, IsReadableFn, ValueFn>(
    camera: &mut PyCameleonCamera,
    node_name: &str,
    downcast: DownCastFn,
    readable: IsReadableFn,
    extract: ValueFn,
) -> PycameleonResult<T>
where
    DownCastFn: for<'a> Fn(
        &'a Node,
        &'a ParamsCtxt<&'a mut ControlHandle, &'a mut DefaultGenApiCtxt>,
    ) -> Option<DownCastNode>,
    IsReadableFn: Fn(
        &DownCastNode,
        &mut ParamsCtxt<&mut ControlHandle, &mut DefaultGenApiCtxt>,
    ) -> Result<bool, GenApiError>,
    ValueFn: for<'a> Fn(
        &DownCastNode,
        &mut ParamsCtxt<&mut ControlHandle, &mut DefaultGenApiCtxt>,
    ) -> Result<T, GenApiError>,
{
    let mut params_ctxt = camera.0.params_ctxt()?;
    let node_result = params_ctxt
        .node(node_name)
        .ok_or(PyValueError::new_err(format!(
            "Node {} is not found.",
            node_name
        )));
    let node_option = downcast(&node_result?, &params_ctxt).ok_or(PyValueError::new_err(format!(
        "Node {} cannot be casted as string.",
        node_name
    )));
    let node = node_option?;
    // let params_ctxt_mut = &mut params_ctxt;
    let is_readable = readable(&node, &mut params_ctxt)?;
    if is_readable {
        let value = extract(&node, &mut params_ctxt)?;
        return Ok(value);
    }
    Err(PyValueError::new_err(format!("Node {} is not readable", node_name)).into())
}

fn write_node<T, DownCastNode, DownCastFn, IsReadableFn, SetterFn>(
    camera: &mut PyCameleonCamera,
    node_name: &str,
    value: T,
    downcast: DownCastFn,
    readable: IsReadableFn,
    setter: SetterFn,
) -> PycameleonResult<()>
where
    DownCastFn: for<'a> Fn(
        &'a Node,
        &'a ParamsCtxt<&'a mut ControlHandle, &'a mut DefaultGenApiCtxt>,
    ) -> Option<DownCastNode>,
    IsReadableFn: Fn(
        &DownCastNode,
        &mut ParamsCtxt<&mut ControlHandle, &mut DefaultGenApiCtxt>,
    ) -> Result<bool, GenApiError>,
    SetterFn: for<'a> Fn(
        &DownCastNode,
        &mut ParamsCtxt<&mut ControlHandle, &mut DefaultGenApiCtxt>,
        T,
    ) -> Result<(), GenApiError>,
{
    let mut params_ctxt = camera.0.params_ctxt()?;
    let node_result = params_ctxt
        .node(node_name)
        .ok_or(PyValueError::new_err(format!(
            "Node {} is not found.",
            node_name
        )));
    let node_option = downcast(&node_result?, &params_ctxt).ok_or(PyValueError::new_err(format!(
        "Node {} cannot be casted to the target type.",
        node_name
    )));
    let node = node_option?;
    // let params_ctxt_mut = &mut params_ctxt;
    let is_readable = readable(&node, &mut params_ctxt)?;
    if is_readable {
        setter(&node, &mut params_ctxt, value)?;
        return Ok(());
    }
    Err(PyValueError::new_err(format!("Node {} is not readable", node_name)).into())
}

macro_rules! impl_read_methods {
    (
        $cls:ty,
        [
            $(
                ($fn_name:ident, $type:ty, $as_fn:ident $(, $value_closure:expr)?)
            ),+ $(,)?
        ]
    ) => {
        #[pymethods]
        impl $cls {
            $(
                pub fn $fn_name(&mut self, node_name: &str) -> PycameleonResult<$type> {
                    read_node(
                        self,
                        node_name,
                        |n, _ctxt| n.$as_fn(_ctxt),
                        |n, _ctxt| n.is_readable(_ctxt),
                        impl_read_methods!(@value_closure $( $value_closure )?)
                    )
                }
            )+
        }
    };

    // if a custom closure is provided, use it
    (@value_closure $value_closure:expr) => { $value_closure };

    // default to |n, _ctxt| n.value(_ctxt)
    (@value_closure) => { |n, _ctxt| n.value(_ctxt) };
}

macro_rules! impl_write_methods {
    (
        $cls:ty,
        [
            $(
                // Optional custom setter closure
                ($fn_name:ident, $type:ty, $as_fn:ident $(, $setter_closure:expr)?)
            ),+ $(,)?
        ]
    ) => {
        #[pymethods]
        impl $cls {
            $(
                pub fn $fn_name(&mut self, node_name: &str, value: $type) -> PycameleonResult<()> {
                    write_node(
                        self,
                        node_name,
                        value,
                        |n, _ctxt| n.$as_fn(_ctxt),
                        |n, _ctxt| n.is_readable(_ctxt),
                        impl_write_methods!(@setter $( $setter_closure )?)
                    )
                }
            )+
        }
    };

    // if a custom setter closure is provided
    (@setter $setter_closure:expr) => { $setter_closure };

    // default setter: |n, _ctxt, v| n.set_value(_ctxt, v)
    (@setter) => { |n, _ctxt, v| n.set_value(_ctxt, v) };
}
impl_write_methods!(
    PyCameleonCamera,
    [
        (write_string, String, as_string),
        (write_integer, i64, as_integer),
        (write_float, f64, as_float),
        (write_bool, bool, as_boolean),
        (write_enum_as_int, i64, as_enumeration, |n, _ctxt, v| {
            n.set_entry_by_value(_ctxt, v)
        }),
        (write_enum_as_str, &str, as_enumeration, |n, _ctxt, v| {
            n.set_entry_by_symbolic(_ctxt, v)
        }),
    ]
);

impl_read_methods!(
    PyCameleonCamera,
    [
        (read_string, String, as_string),
        (read_integer, i64, as_integer),
        (read_float, f64, as_float),
        (read_bool, bool, as_boolean),
        (read_enum_as_int, i64, as_enumeration, |n, _ctxt| {
            Ok(n.current_entry(_ctxt)?.value(_ctxt))
        }),
        (read_enum_as_str, String, as_enumeration, |n, _ctxt| {
            Ok(n.current_entry(_ctxt)?.symbolic(_ctxt).to_owned())
        }),
    ]
);

#[pymethods]
impl PyCameleonCamera {
    pub fn open(&mut self) -> PycameleonResult<()> {
        self.0.open()?;
        Ok(())
    }
    pub fn load_context(&mut self) -> PycameleonResult<String> {
        Ok(self.0.load_context()?)
    }

    pub fn info<'a>(&'a mut self) -> PycameleonResult<PyCameraInfo<'a>> {
        Ok(PyCameraInfo(self.0.info()))
    }

    pub fn start_streaming(&mut self, cap: usize) -> PycameleonResult<PyPayloadReceiver> {
        Ok(PyPayloadReceiver(self.0.start_streaming(cap)?))
    }

    pub fn close(&mut self) -> PycameleonResult<()> {
        let cam = &mut self.0;
        cam.close().unwrap();
        Ok(())
    }

    pub fn execute(&mut self, node_name: &str) -> PycameleonResult<()> {
        let mut params_ctxt = self.0.params_ctxt()?;
        let node_option = params_ctxt
            .node(node_name)
            .ok_or(PyValueError::new_err(format!(
                "Node {} not found.",
                node_name
            )))?
            .as_command(&params_ctxt);
        let node = node_option.ok_or(PyValueError::new_err(format!(
            "Node {} cannot be casted as Command.",
            node_name
        )))?;
        if node.is_writable(&mut params_ctxt)? {
            node.execute(&mut params_ctxt)?;
            return Ok(());
        }
        Err(PyValueError::new_err(format!("Node {} is not writable", node_name)).into())
    }

    pub fn isdone_command(&mut self, node_name: &str) -> PycameleonResult<bool> {
        let mut params_ctxt = self.0.params_ctxt()?;
        let node_option = params_ctxt
            .node(node_name)
            .ok_or(PyValueError::new_err(format!(
                "Node {} not found.",
                node_name
            )))?
            .as_command(&params_ctxt);
        let node = node_option.ok_or(PyValueError::new_err(format!(
            "Node {} cannot be casted as Command.",
            node_name
        )))?;
        if node.is_writable(&mut params_ctxt)? {
            let result = node.is_done(&mut params_ctxt)?;
            return Ok(result);
        }
        Err(PyValueError::new_err(format!("Node {} is not writable", node_name)).into())
    }

    pub fn receive(
        &mut self,
        py: Python,
        payload_rx: &mut PyPayloadReceiver,
    ) -> PycameleonResult<Py<PyAny>> {
        let payload = match payload_rx.0.recv_blocking() {
            Ok(payload) => payload,
            Err(_) => {
                return Err(PyValueError::new_err("Failed to receive image").into());
            }
        };

        let result = match payload.image() {
            Some(img) => {
                let info = payload
                    .image_info()
                    .ok_or(PyValueError::new_err("Payload image empty"))?;
                let ndarray = PyArray::from_slice(py, img);
                ndarray
                    .reshape([info.height, info.width])
                    .unwrap()
                    .into_py_any(py)
            }
            None => PyNone::get(py).into_py_any(py),
        }?;

        payload_rx.0.send_back(payload);
        Ok(result)
    }

    pub fn __str__(&self) -> PycameleonResult<String> {
        Ok(format!("{:?}", self.0.info()))
    }

    pub fn __repr__(&self) -> PycameleonResult<String> {
        Ok(format!("{:?}", self.0.info()))
    }

    pub fn __enter__(&mut self) -> PycameleonResult<()> {
        self.0.open()?;
        Ok(())
    }

    pub fn __exit__(&mut self) -> PycameleonResult<()> {
        self.0.close()?;
        Ok(())
    }
}

#[pyfunction]
fn enumerate_cameras() -> PycameleonResult<Vec<PyCameleonCamera>> {
    let cameras = cameleon::u3v::enumerate_cameras()?;

    let pycameras = cameras.into_iter().map(PyCameleonCamera).collect();
    Ok(pycameras)
}

#[pymodule]
fn pycameleon(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enumerate_cameras, m)?)
}

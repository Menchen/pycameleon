use cameleon::{
    genapi::{DefaultGenApiCtxt, FromXml},
    payload::PayloadReceiver,
    CameraInfo,
};
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

// enum PyNodeType{
//     Float,
//     String,
//     Integer,
// }

// #[pyclass]
// pub struct PyNode{
//     node: Node,
//     node_type: PyNodeType,
// }

impl<'py> IntoPyObject<'py> for PyCameraInfo<'_> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        dict.set_item("model_name", &self.0.model_name).unwrap();
        dict.set_item("vendor_name", &self.0.vendor_name).unwrap();
        dict.set_item("serial_number", &self.0.serial_number)
            .unwrap();
        dict.into_bound_py_any(py)
    }
    // fn into(self, py: Python<'_>) -> PyObject {
    //     let dict = PyDict::new(py);
    //     dict.set_item("model_name", &self.0.model_name).unwrap();
    //     dict.set_item("vendor_name", &self.0.vendor_name).unwrap();
    //     dict.set_item("serial_number", &self.0.serial_number)
    //         .unwrap();
    //     dict.into_py_any(py)
    // }
}

#[pymethods]
impl PyCameleonCamera {
    pub fn open(&mut self) -> PyResult<()> {
        self.0.open().unwrap();
        Ok(())
    }
    pub fn load_context(&mut self) -> PyResult<String> {
        let result = self
            .0
            .load_context()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result)
    }
    pub fn load_context_xml(&mut self, genapi_xml: &str) -> PyResult<()> {
        let context = DefaultGenApiCtxt::from_xml(&genapi_xml)
            .map_err(|e| PyValueError::new_err(e.to_string()));
        self.0.ctxt = Some(context?);

        // let result = self
        //     .0
        //     .load_context()
        //     .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
    pub fn read_string(&mut self, node_name: &str) -> PyResult<Option<String>> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt.node(node_name).unwrap().as_string(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as string",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not readable",
            node_name
        )))
    }

    pub fn read_integer(&mut self, node_name: &str) -> PyResult<Option<i64>> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_integer(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as integer",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not readable",
            node_name
        )))
    }

    pub fn read_boolean(&mut self, node_name: &str) -> PyResult<Option<bool>> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_boolean(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as boolean",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not readable",
            node_name
        )))
    }

    pub fn read_float(&mut self, node_name: &str) -> PyResult<Option<f64>> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt.node(node_name).unwrap().as_float(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as float",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not readable",
            node_name
        )))
    }

    pub fn read_enum_as_int(&mut self, node_name: &str) -> PyResult<Option<i64>> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_enumeration(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as enumeration",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(
                node.current_entry(&mut params_ctxt)
                    .unwrap()
                    .value(&params_ctxt),
            ));
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not readable",
            node_name
        )))
    }

    pub fn read_enum_as_str(&mut self, node_name: &str) -> PyResult<Option<String>> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_enumeration(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as enumeration",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(
                node.current_entry(&mut params_ctxt)
                    .unwrap()
                    .symbolic(&params_ctxt)
                    .to_owned(),
            ));
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not readable",
            node_name
        )))
    }

    pub fn write_string(&mut self, node_name: &str, value: String) -> PyResult<()> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt.node(node_name).unwrap().as_string(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as string",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    pub fn write_integer(&mut self, node_name: &str, value: i64) -> PyResult<()> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_integer(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as integer",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    pub fn write_boolean(&mut self, node_name: &str, value: bool) -> PyResult<()> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_boolean(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as boolean",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    pub fn write_float(&mut self, node_name: &str, value: f64) -> PyResult<()> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt.node(node_name).unwrap().as_float(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as float",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    pub fn execute_command(&mut self, node_name: &str) -> PyResult<()> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_command(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as float",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_writable(&mut params_ctxt).unwrap() {
            node.execute(&mut params_ctxt).unwrap();
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    pub fn isdone_command(&mut self, node_name: &str) -> PyResult<bool> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_command(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as float",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_writable(&mut params_ctxt).unwrap() {
            let result = node.is_done(&mut params_ctxt).unwrap();
            return Ok(result);
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    pub fn write_enum_as_int(&mut self, node_name: &str, value: i64) -> PyResult<()> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_enumeration(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as enum",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_entry_by_value(&mut params_ctxt, value).unwrap();
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    pub fn write_enum_as_str(&mut self, node_name: &str, value: &str) -> PyResult<()> {
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node_option = params_ctxt
            .node(node_name)
            .unwrap()
            .as_enumeration(&params_ctxt);
        if node_option.is_none() {
            return Err(PyValueError::new_err(format!(
                "Node {} cannot be casted as enum",
                node_name
            )));
        }
        let node = node_option.unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_entry_by_symbolic(&mut params_ctxt, value).unwrap();
            return Ok(());
        }
        Err(PyValueError::new_err(format!(
            "Node {} is not writable",
            node_name
        )))
    }

    // pub fn node(&mut self,node_name : &str) -> PyResult<PyNode>{
    //     let params_ctxt = self.0.params_ctxt().unwrap();
    //     let node = params_ctxt.node(node_name).unwrap();
    //     Ok(PyNode(node))
    // }

    // pub fn as_float(&mut self, node: &PyNode) -> PyResult<PyNode>{
    //     let params_ctxt = self.0.params_ctxt().unwrap();
    //     let float_node = node.0.as_float(&params_ctxt).unwrap();
    // }

    // pub fn params_is_readable(&mut self,node_name : &str) -> PyResult<bool>{
    //     let mut params_ctxt = self.0.params_ctxt().unwrap();
    //     let node = params_ctxt.node(node_name).unwrap();
    //     Ok(true)
    // }

    // pub fn params_ctxt(&mut self) -> PyResult<PyParamsCtxt>{
    //     OK(PyParamsCtxt(self.0.params_ctxt().unwrap()))
    // }

    pub fn info(&mut self) -> PyResult<PyCameraInfo<'_>> {
        Ok(PyCameraInfo(self.0.info()))
    }

    pub fn start_streaming(&mut self, cap: usize) -> PyResult<PyPayloadReceiver> {
        Ok(PyPayloadReceiver(self.0.start_streaming(cap).unwrap()))
    }

    pub fn close(&mut self) -> PyResult<()> {
        let cam = &mut self.0;
        let result = cam.close();
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    pub fn receive(
        &mut self,
        py: Python,
        payload_rx: &mut PyPayloadReceiver,
    ) -> PyResult<Py<PyAny>> {
        let payload = match payload_rx.0.recv_blocking() {
            Ok(payload) => payload,
            Err(e) => {
                return Err(PyValueError::new_err(e.to_string()));
            }
        };

        let result = match payload.image() {
            Some(img) => {
                let info = payload.image_info().unwrap();
                let ndarray = PyArray::from_slice(py, img);
                ndarray
                    .reshape([info.height, info.width])
                    .unwrap()
                    .into_py_any(py)
            }
            None => PyNone::get(py).into_py_any(py),
        };

        payload_rx.0.send_back(payload);
        result
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.0.info()))
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.0.info()))
    }

    pub fn __enter__(&mut self) -> PyResult<()> {
        Ok(self.0.open().unwrap())
    }

    pub fn __exit__(&mut self) -> PyResult<()> {
        Ok(self.0.close().unwrap())
    }
}
#[pyfunction]
fn enumerate_cameras() -> PyResult<Vec<PyCameleonCamera>> {
    let cameras = cameleon::u3v::enumerate_cameras().unwrap();

    let pycameras = cameras.into_iter().map(|c| PyCameleonCamera(c)).collect();
    Ok(pycameras)
}

#[pymodule]
fn pycameleon(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enumerate_cameras, m)?)
}

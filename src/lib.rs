
use cameleon::{payload::PayloadReceiver, CameraInfo};
use pyo3::{exceptions::PyValueError, prelude::*, types::{PyBytes, PyNone, PyDict}};

// use cameleon::u3v::enumerate_cameras;

#[pyclass]
pub struct PyCameleonCamera(pub cameleon::Camera<cameleon::u3v::ControlHandle,cameleon::u3v::StreamHandle>);

#[pyclass]
pub struct PyPayloadReceiver(pub PayloadReceiver);

pub struct PyCameraInfo<'a>(pub &'a CameraInfo);

#[pyfunction]
fn enumerate_cameras() -> PyResult<Vec<PyCameleonCamera>>{
    let cameras = cameleon::u3v::enumerate_cameras().unwrap();

    if cameras.is_empty(){
        return Err(PyValueError::new_err("No camera found!"));
    }

    let pycameras = cameras.into_iter().map(|c| {PyCameleonCamera(c)}).collect();
    Ok(pycameras)
}

impl IntoPy<PyObject> for PyCameraInfo<'_>{
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("model_name",&self.0.model_name).unwrap();
        dict.set_item("vendor_name",&self.0.vendor_name).unwrap();
        dict.set_item("serial_number",&self.0.serial_number).unwrap();
        dict.into_py(py)
    }
}

#[pymethods]
impl PyCameleonCamera{
    pub fn open(&mut self) -> PyResult<()>{
        self.0.open().unwrap();
        Ok(())
    }
    pub fn load_context(&mut self) -> PyResult<String>{
        Ok(self.0.load_context().unwrap())

    }

    pub fn info(&mut self) -> PyResult<PyCameraInfo>
    {
        Ok(PyCameraInfo(self.0.info()))
    }

    pub fn start_streaming(&mut self,cap :usize) -> PyResult<PyPayloadReceiver>{
        Ok(PyPayloadReceiver(self.0.start_streaming(cap).unwrap()))
    }

    pub fn close(&mut self) -> PyResult<()>{
        let cam = &mut self.0;
        cam.close().unwrap();
        Ok(())
    }

    pub fn receive(&mut self, py: Python, payload_rx : &mut PyPayloadReceiver) -> PyResult<PyObject>{
        let payload = match payload_rx.0.recv_blocking(){
            Ok(payload) => payload,
            Err(_) => {
                return Err(PyValueError::new_err("Failed to receive image"));
            }
        };

        let result = match payload.image() {
            Some(img) => Ok(PyBytes::new(py, img).to_object(py)),
            None => Ok(PyNone::get(py).to_object(py))
        };

        payload_rx.0.send_back(payload);
        result
    }

    pub fn __str__(&self) -> PyResult<String>{
        Ok(format!("{:?}",self.0.info()))
    }

    pub fn __repr__(&self) -> PyResult<String>{
        Ok(format!("{:?}",self.0.info()))
    }

}

#[pymodule]
fn pycameleon(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(enumerate_cameras))?;
    Ok(())
}



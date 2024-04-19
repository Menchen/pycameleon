

use cameleon::{payload::PayloadReceiver, CameraInfo};
use pyo3::{exceptions::PyValueError, prelude::*, types::{PyDict, PyNone}};
use numpy::PyArray;

// use cameleon::u3v::enumerate_cameras;

#[pyclass]
pub struct PyCameleonCamera(pub cameleon::Camera<cameleon::u3v::ControlHandle,cameleon::u3v::StreamHandle>);

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
    pub fn read_string(&mut self, node_name : &str) -> PyResult<Option<String>>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_string(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Ok(None)
    }

    pub fn read_integer(&mut self, node_name : &str) -> PyResult<Option<i64>>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_integer(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Ok(None)
    }

    pub fn read_boolean(&mut self, node_name : &str) -> PyResult<Option<bool>>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_boolean(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Ok(None)
    }

    pub fn read_float(&mut self, node_name : &str) -> PyResult<Option<f64>>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_float(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            return Ok(Some(node.value(&mut params_ctxt).unwrap()));
        }
        Ok(None)
    }

    pub fn write_string(&mut self, node_name : &str, value : String) -> PyResult<()>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_string(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(())
        }
        Err(PyValueError::new_err(format!("Node {} is not writable",node_name)))
    }


    pub fn write_integer(&mut self, node_name : &str, value : i64) -> PyResult<()>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_integer(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(())
        }
        Err(PyValueError::new_err(format!("Node {} is not writable",node_name)))
    }



    pub fn write_boolean(&mut self, node_name : &str, value : bool) -> PyResult<()>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_boolean(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(())
        }
        Err(PyValueError::new_err(format!("Node {} is not writable",node_name)))
    }


    pub fn write_float(&mut self, node_name : &str, value : f64) -> PyResult<()>{
        let mut params_ctxt = self.0.params_ctxt().unwrap();
        let node = params_ctxt.node(node_name).unwrap().as_float(&params_ctxt).unwrap();
        if node.is_readable(&mut params_ctxt).unwrap() {
            node.set_value(&mut params_ctxt, value).unwrap();
            return Ok(())
        }
        Err(PyValueError::new_err(format!("Node {} is not writable",node_name)))
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
            Some(img) => {
                let info = payload.image_info().unwrap();
                let ndarray = PyArray::from_slice(py, img);
                ndarray.reshape([info.height,info.width]).unwrap().to_object(py)
            },
            None => PyNone::get(py).to_object(py)
        };

        payload_rx.0.send_back(payload);
        Ok(result)
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
    #[pyfn(m)]
    fn enumerate_cameras() -> PyResult<Vec<PyCameleonCamera>>{
        let cameras = cameleon::u3v::enumerate_cameras().unwrap();

        let pycameras = cameras.into_iter().map(|c| {PyCameleonCamera(c)}).collect();
        Ok(pycameras)
    }
    Ok(())
}



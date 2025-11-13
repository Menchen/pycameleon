use cameleon::{genapi::GenApiError, CameleonError};
use pyo3::{exceptions::*, PyErr};
use thiserror::Error;

pub type PycameleonResult<T> = Result<T, PycameleonError>;

#[derive(Error, Debug)]
pub enum PycameleonError {
    #[error(transparent)]
    Cameleon(#[from] CameleonError),

    #[error(transparent)]
    Py(#[from] PyErr),

    #[error(transparent)]
    GenApi(#[from] GenApiError),
    // PyValue(#[from] PyValueError),
}

// convert to PyErr
impl From<PycameleonError> for PyErr {
    fn from(err: PycameleonError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

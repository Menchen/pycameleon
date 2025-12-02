from typing import Any, NewType

# pyfunction
def enumerate_cameras() -> list[PyCameleonCamera]:
    pass

class PyPayloadReceiver:
    pass

class PyImageInfo:
    ##[getter]
    # pub fn width(&self) -> usize {
    #    self.0.width
    # }
    #
    ##[getter]
    # pub fn height(&self) -> usize {
    #    self.0.height
    # }
    #
    ##[getter]
    # pub fn pixel_format(&self) -> String {
    #    format!("{:?}", self.0.pixel_format)
    # }
    @property
    def width(self) -> int:
        pass
    @property
    def height(self) -> int:
        pass
    @property
    def pixel_format(self) -> str:
        pass

PyCameraInfo = NewType("PyCameraInfo", dict)

class PyCameleonCamera:
    pass
    def open(self) -> None:
        pass
    def load_context_from_camera(self) -> str:
        pass
    def load_context_from_xml(self, gen_api_context: str) -> None:
        pass
    def info(self) -> PyCameraInfo:
        pass
    def start_streaming(self, int) -> PyPayloadReceiver:
        pass
    def close(self) -> None:
        pass
    def execute(self, node_name: str) -> None:
        pass
    def is_command_done(self, node_name: str) -> bool:
        pass

    # TODO Narrow the return type
    def receive(self, payload_rx: PyPayloadReceiver) -> Any:
        pass

    def receive_raw(self, payload_rx: PyPayloadReceiver) -> tuple[Any, PyImageInfo]:
        pass

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

    def __enter__(self) -> None:
        pass

    def __exit__(self) -> None:
        pass

    ######## READ NODE MACROS ########

    #     (read_string, String, as_string),
    #     (read_integer, i64, as_integer),
    #     (read_float, f64, as_float),
    #     (read_bool, bool, as_boolean),
    #     (read_enum_as_int, i64, as_enumeration, |n, _ctxt| {
    #         Ok(n.current_entry(_ctxt)?.value(_ctxt))
    #     }),
    #     (read_enum_as_str, String, as_enumeration, |n, _ctxt| {
    #         Ok(n.current_entry(_ctxt)?.symbolic(_ctxt).to_owned())
    #     }),

    def read_string(self, node_name: str) -> str:
        pass
    def read_integer(self, node_name: str) -> int:
        pass
    def read_float(self, node_name: str) -> float:
        pass
    def read_bool(self, node_name: str) -> bool:
        pass
    def read_enum_as_int(self, node_name: str) -> int:
        pass
    def read_enum_as_str(self, node_name: str) -> str:
        pass

    ######## WRITE NODE MACROS ########

    # (write_string, String, as_string),
    # (write_integer, i64, as_integer),
    # (write_float, f64, as_float),
    # (write_bool, bool, as_boolean),
    # (write_enum_as_int, i64, as_enumeration, |n, _ctxt, v| {
    #     n.set_entry_by_value(_ctxt, v)
    # }),
    # (write_enum_as_str, &str, as_enumeration, |n, _ctxt, v| {
    #     n.set_entry_by_symbolic(_ctxt, v)
    # }),

    def write_string(self, node_name: str, value: str) -> None:
        pass
    def write_integer(self, node_name: str, value: int) -> None:
        pass
    def write_float(self, node_name: str, value: float) -> None:
        pass
    def write_bool(self, node_name: str, value: bool) -> None:
        pass
    def write_enum_as_int(self, node_name: str, value: int) -> None:
        pass
    def write_enum_as_str(self, node_name: str, value: str) -> None:
        pass

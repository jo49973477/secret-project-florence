"""Dependency-light ZeroMQ/msgpack client compatible with GR00T PolicyServer."""

from __future__ import annotations

import io
import json
from typing import Any

import msgpack
import numpy as np
import zmq


def _dict_get(mapping: dict[Any, Any], key: str, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping.get(key.encode(), default)


class CompatibleMsgSerializer:
    """Read msgpack-numpy replies without importing GR00T or msgpack-numpy."""

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                raise TypeError("Refusing to serialize an object-dtype ndarray.")
            payload = io.BytesIO()
            np.save(payload, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": payload.getvalue()}
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Unsupported msgpack object: {type(obj).__name__}")

    @staticmethod
    def _decode(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj

        if _dict_get(obj, "__ndarray_class__"):
            payload = _dict_get(obj, "as_npy")
            if payload is None:
                raise ValueError("Malformed ndarray payload: missing as_npy.")
            return np.load(io.BytesIO(payload), allow_pickle=False)

        nd_marker = _dict_get(obj, "nd")
        if nd_marker is not None:
            kind = _dict_get(obj, "kind")
            if kind in ("O", b"O"):
                raise ValueError("Refusing to decode an object-dtype ndarray payload.")
            dtype_value = _dict_get(obj, "type")
            data = _dict_get(obj, "data")
            if dtype_value is None or data is None:
                raise ValueError("Malformed msgpack-numpy ndarray payload.")
            dtype = np.dtype(dtype_value)
            if bool(nd_marker):
                shape = tuple(_dict_get(obj, "shape", ()))
                return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            return np.frombuffer(data, dtype=dtype, count=1)[0]

        has_modality_marker = bool(
            _dict_get(obj, "__ModalityConfig__") or _dict_get(obj, "__ModalityConfig_class__")
        )
        if has_modality_marker:
            payload = _dict_get(obj, "as_json")
            if payload is None:
                raise ValueError("Malformed ModalityConfig payload: missing as_json.")
            if isinstance(payload, bytes):
                payload = payload.decode()
            return json.loads(payload) if isinstance(payload, str) else payload
        return obj

    @classmethod
    def to_bytes(cls, data: Any) -> bytes:
        return msgpack.packb(data, default=cls._encode, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=cls._decode, raw=False, strict_map_key=False)


class MinimalPolicyClient:
    """Only the endpoints needed by UniVTAC, with explicit timeout recovery."""

    def __init__(self, host: str, port: int, *, timeout_ms: int = 300_000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._closed = False
        self.context = zmq.Context()
        self.socket: zmq.Socket
        self._init_socket()

    def _init_socket(self) -> None:
        old_socket = getattr(self, "socket", None)
        if old_socket is not None:
            old_socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(CompatibleMsgSerializer.to_bytes(request))
            response = CompatibleMsgSerializer.from_bytes(self.socket.recv())
        except zmq.error.Again as exc:
            self._init_socket()
            raise TimeoutError(
                f"Timed out calling GR00T server endpoint {endpoint!r} at "
                f"{self.host}:{self.port} after {self.timeout_ms} ms."
            ) from exc
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> bool:
        response = self.call("ping")
        return isinstance(response, dict) and response.get("status") == "ok"

    def get_action(self, observation: dict[str, Any]) -> Any:
        return self.call("get_action", {"observation": observation, "options": None})

    def reset(self) -> Any:
        return self.call("reset", {"options": None})

    def get_modality_config(self) -> dict[str, Any]:
        response = self.call("get_modality_config")
        if not isinstance(response, dict):
            raise TypeError(f"Expected modality config dict, got {type(response).__name__}.")
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.socket.close(linger=0)
        self.context.term()

    def __enter__(self) -> "MinimalPolicyClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

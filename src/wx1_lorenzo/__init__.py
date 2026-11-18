"""Lorenzo2D: a lossless Zarr v3 array-to-array codec for integer-quantised (GRIB-style) lattices.

Native zarr-python >= 3 ``ArrayArrayCodec`` (NOT a numcodecs.abc.Codec).  For every chunk the last two axes
are treated as (y, x); all leading axes are independent slices.

    d[y,x] = X[y,x] - X[y,x-1] - X[y-1,x] + X[y-1,x-1]   (modulo 2**bits)  followed by ZigZag.

Decoding is two modular prefix sums, so reconstruction of the integers is exact.
Chain it with a bytes codec, byte-shuffle and zstd (see README / examples).
Only uint16 and uint8 chunks are supported.  Regular (rectangular) grids only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

__version__ = "0.1.0"
CODEC_NAME = "wx1.lorenzo2d"
_SIGNED = {np.dtype("uint16"): np.int16, np.dtype("uint8"): np.int8}


def lorenzo2d_forward(a: np.ndarray) -> np.ndarray:
    """uint8/uint16 (..., ny, nx) -> ZigZag(2-D mixed difference), same dtype."""
    dt = a.dtype
    s = _SIGNED[dt]
    bits = dt.itemsize * 8
    dx = a.copy()
    dx[..., 1:] = a[..., 1:] - a[..., :-1]
    d = dx.copy()
    d[..., 1:, :] = dx[..., 1:, :] - dx[..., :-1, :]
    si = d.view(s).astype(np.int32)
    return ((si << 1) ^ (si >> (bits - 1))).astype(dt)


def lorenzo2d_inverse(z: np.ndarray) -> np.ndarray:
    dt = z.dtype
    s = _SIGNED[dt]
    zi = z.astype(np.int32)
    d = ((zi >> 1) ^ -(zi & 1)).astype(s).view(dt)
    return np.cumsum(np.cumsum(d, axis=-1, dtype=dt), axis=-2, dtype=dt)


@dataclass(frozen=True)
class Lorenzo2DCodec(ArrayArrayCodec):
    is_fixed_size = True

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        parse_named_configuration(data, CODEC_NAME, require_configuration=False)
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": CODEC_NAME}

    def validate(self, shape, dtype, chunk_grid) -> None:
        if len(shape) < 2:
            raise ValueError("wx1.lorenzo2d needs at least 2 dimensions")
        if np.dtype(dtype.to_native_dtype()) not in _SIGNED:
            raise ValueError("wx1.lorenzo2d supports uint8/uint16 only")

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer | None:
        out = lorenzo2d_forward(np.ascontiguousarray(chunk_array.as_numpy_array()))
        return chunk_spec.prototype.nd_buffer.from_numpy_array(out)

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        out = lorenzo2d_inverse(np.ascontiguousarray(chunk_array.as_numpy_array()))
        return chunk_spec.prototype.nd_buffer.from_numpy_array(out)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length


def register() -> None:
    register_codec(CODEC_NAME, Lorenzo2DCodec)


register()  # importing the package registers the codec (also done via the zarr.codecs entry point)


def lattice_params(gid) -> tuple[float, int, int]:
    """(R, E, D) of an open ecCodes GRIB handle."""
    import eccodes as ec
    return (float(ec.codes_get_double(gid, "referenceValue")), int(ec.codes_get(gid, "binaryScaleFactor")),
            int(ec.codes_get(gid, "decimalScaleFactor")))


def to_lattice(values: np.ndarray, R: float, E: int, D: int) -> np.ndarray:
    """float64 -> uint16 lattice.  Caller must check exactness with ``from_lattice``."""
    return np.rint((values * (10.0 ** D) - R) / (2.0 ** E)).astype(np.uint16)


def from_lattice(X: np.ndarray, R: float, E: int, D: int) -> np.ndarray:
    """Exactly the ecCodes decode formula (float64)."""
    y = R + X.astype(np.float64) * (2.0 ** E)
    return y * (10.0 ** (-D)) if D else y

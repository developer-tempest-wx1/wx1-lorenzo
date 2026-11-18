# wx1-lorenzo

Lossless 2-D Lorenzo spatial predictor as a native **Zarr v3 `ArrayArrayCodec`** (`wx1.lorenzo2d`) for integer-quantised meteorological grids (e.g. regular lat/lon GRIB-style `R, E, D` lattices).

- **Native Zarr v3 codec** (`zarr.abc.codec.ArrayArrayCodec`, *not* a `numcodecs.abc.Codec`; requires `zarr >= 3.1`).
- **Auto-registration:** Registered automatically via `import wx1_lorenzo` and the `[project.entry-points."zarr.codecs"]` Python entry point.
- **Bit-exact integer reconstruction:** Uses modular $2^b$ arithmetic for `uint8` and `uint16` chunks on regular rectangular `(y, x)` grids.
- **License:** Apache-2.0.

## Installation

```bash
pip install wx1-lorenzo
# With GRIB / shuffle example dependencies:
pip install "wx1-lorenzo[grib]"
```

Or from source:

```bash
pip install -e ".[grib]"
```

## How It Works

For each chunk of shape `(..., ny, nx)` (`uint8` or `uint16`), `Lorenzo2DCodec` treats the last two dimensions `(axis -2, axis -1)` as the 2-D spatial grid `(y, x)` while keeping any leading dimensions independent.

1. **Forward transform (encode):**
   $$d_{y,x} = X_{y,x} - X_{y,x-1} - X_{y-1,x} + X_{y-1,x-1} \pmod{2^b}$$
   with boundary conditions $d_{0,0} = X_{0,0}$, $d_{0,x} = X_{0,x} - X_{0,x-1} \pmod{2^b}$, and $d_{y,0} = X_{y,0} - X_{y-1,0} \pmod{2^b}$.
   Signed differences (reinterpreted as `int8` / `int16`) are ZigZag-mapped back to unsigned integers of the same dtype:
   $$z = (d_{\text{signed}} \ll 1) \oplus (d_{\text{signed}} \gg (b - 1))$$

2. **Inverse transform (decode):**
   Inverse ZigZag mapping followed by two exact modular prefix sums (`numpy.cumsum`) along `axis=-1` and `axis=-2` in the native unsigned dtype.

Chain `Lorenzo2DCodec` with `BytesCodec()`, byte-shuffle (`zarr.codecs.numcodecs.Shuffle(elementsize=2)`), and `ZstdCodec(level=3)` to match the benchmark pipeline.

## Quickstart (Zarr v3)

```python
import numpy as np
import zarr
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.codecs.numcodecs import Shuffle
import wx1_lorenzo as wl  # registers "wx1.lorenzo2d"

data = np.cumsum(np.cumsum(np.ones((2, 721, 1440), dtype=np.uint16), axis=1), axis=2).astype(np.uint16)

z = zarr.create_array(
    "sample.zarr",
    shape=data.shape,
    chunks=(1, 721, 1440),
    dtype="uint16",
    zarr_format=3,
    filters=[wl.Lorenzo2DCodec()],
    serializer=BytesCodec(),
    compressors=[Shuffle(elementsize=2), ZstdCodec(level=3)],
    overwrite=True,
)
z[:] = data

z_read = zarr.open_array("sample.zarr", mode="r")
assert np.array_equal(z_read[:], data)
```

## End-to-End GRIB2 Example

See [`examples/grib_to_zarr_e2e.py`](examples/grib_to_zarr_e2e.py) for extracting `(R, E, D)` affine lattice parameters from a regular lat/lon GRIB2 file, compressing with `wx1.lorenzo2d -> bytes -> numcodecs.shuffle -> zstd`, verifying double-precision bit-exactness against `eccodes`, and comparing disk usage against GRIB2-AEC:

```bash
python examples/grib_to_zarr_e2e.py FILE.grib2 gh out.zarr
```

## Limitations

- Supports `uint8` and `uint16` chunks with rank $\ge 2$.
- Regular rectangular grids only; does not directly apply to reduced Gaussian grids (e.g. O1280) or spectral coefficients.

## Development & Tests

```bash
pip install -e ".[grib]" pytest
pytest -v
```

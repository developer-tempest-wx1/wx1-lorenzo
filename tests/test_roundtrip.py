import numpy as np, zarr, pytest
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.codecs.numcodecs import Shuffle
import wx1_lorenzo as wl

@pytest.mark.parametrize("dtype", ["uint16", "uint8"])
@pytest.mark.parametrize("shape", [(721, 1440), (3, 64, 97), (2, 2, 5, 7)])
def test_numpy_roundtrip(dtype, shape):
    rng = np.random.default_rng(0)
    x = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
    assert np.array_equal(wl.lorenzo2d_inverse(wl.lorenzo2d_forward(x)), x)

def test_extremes():
    for v in (0, 65535):
        x = np.full((5, 6), v, np.uint16); x[2, 3] = 65535 - v
        assert np.array_equal(wl.lorenzo2d_inverse(wl.lorenzo2d_forward(x)), x)

def test_zarr_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    x = np.cumsum(np.cumsum(rng.integers(-3, 4, (4, 50, 60)), 1), 2).astype(np.uint16)
    z = zarr.create_array(str(tmp_path / "a.zarr"), shape=x.shape, chunks=(1, 50, 60), dtype="uint16", zarr_format=3,
                          filters=[wl.Lorenzo2DCodec()], serializer=BytesCodec(),
                          compressors=[Shuffle(elementsize=2), ZstdCodec(level=3)])
    z[:] = x
    assert np.array_equal(zarr.open_array(str(tmp_path / "a.zarr"), mode="r")[:], x)

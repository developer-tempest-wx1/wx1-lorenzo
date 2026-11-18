"""End-to-end: GRIB2 (AEC) -> Zarr v3 with Lorenzo2D -> read back -> bit-exact check -> size vs GRIB.
Usage: python examples/grib_to_zarr_e2e.py FILE.grib2 SHORTNAME OUT.zarr
"""
import shutil, sys, os
import eccodes as ec, numpy as np, zarr
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.codecs.numcodecs import Shuffle
import wx1_lorenzo as wl            # importing registers the codec "wx1.lorenzo2d"

path, short, out = sys.argv[1:4]
fields, sizes, params, refs = [], 0, [], []
with open(path, "rb") as f:
    while (gid := ec.codes_grib_new_from_file(f)) is not None:
        if ec.codes_get(gid, "shortName") == short:
            R, E, D = wl.lattice_params(gid)
            v = ec.codes_get_values(gid).reshape(int(ec.codes_get(gid, "Nj")), int(ec.codes_get(gid, "Ni")))
            X = wl.to_lattice(v, R, E, D)
            assert np.array_equal(wl.from_lattice(X, R, E, D), v), "field is not on its GRIB lattice"
            fields.append(X); refs.append(v); params.append([R, E, D])
            sizes += len(ec.codes_get_message(gid))
        ec.codes_release(gid)
X = np.stack(fields)
shutil.rmtree(out, ignore_errors=True)
z = zarr.create_array(out, shape=X.shape, chunks=(1, *X.shape[1:]), dtype="uint16", zarr_format=3,
                      filters=[wl.Lorenzo2DCodec()], serializer=BytesCodec(),
                      compressors=[Shuffle(elementsize=2), ZstdCodec(level=3)],
                      attributes={"grib_lattice_R_E_D": params})
z[:] = X
# --- read back through a fresh handle
z2 = zarr.open_array(out, mode="r")
R_E_D = z2.attrs["grib_lattice_R_E_D"]
ok = all(np.array_equal(wl.from_lattice(z2[i], *p), refs[i]) for i, p in enumerate(R_E_D))
nbytes = sum(os.path.getsize(os.path.join(r, n)) for r, _, ns in os.walk(out) for n in ns)
print(f"{short}: {len(fields)} fields  GRIB(AEC)={sizes/1e6:.2f} MB  Zarr+Lorenzo2D={nbytes/1e6:.2f} MB "
      f"({100*(nbytes/sizes-1):+.1f}%)  float64 bit-exact vs ecCodes: {ok}")

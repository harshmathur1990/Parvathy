# Data for Parvathy

Copy `export_simulations.py`, `plot_simulations.py` and `plot_halpha.py` together
onto the cluster. Use your Helita fork and install numpy, scipy and h5py.

Export every discovered simulation/snapshot H output into one file:

```bash
python3 export_simulations.py --output parvathy_simulations.h5
```

Optional filters: `--snap 385`, `--simulations en024048_hion`, or
`--manifest simulations.example.json`. Existing output files are protected:
use a new filename if you need to repeat the export.

The exporter uses the same grid-slice inference, coordinate checks and block
staggering as the plotting script. It writes no plots. A vertical emergent ray
is required. Numeric-resolution output folders map to an existing base Bifrost
run, as in the plotting script.

For each pair, the HDF5 layout is:

```
simulations/<output_simulation>/snapshots/<snapshot>/
    coordinates/x                 (nx,)          cm
    coordinates/y                 (ny,)          cm
    coordinates/wavelength        (nlambda,)     angstrom, increasing
    coordinates/ltau500            (4,)           [0, -1, -5.1, -5.7]
    magnetic_field_bz             (4, nx, ny)     kG, signed original Bifrost bz
    magnetic_field_valid          (4, nx, ny)     boolean
    halpha_intensity              (nx, ny, nlambda) original Multi3D ie
```

Bz is centered before taking the inferred vertical slice, then interpolated
onto each column's requested log10(tau500) surface. Its sign is the original
Bifrost sign, positive along original Bifrost +z. Coordinate axes are those of
Multi3D in cm. The sign convention is also recorded on the dataset. Use abs(Bz)
for the original unsigned-field plots. Missing/ambiguous depth surfaces contain
NaN and a false validity flag. Quiet-region continuum normalization is **not**
applied to the exported intensity.

Wavelengths and spectra are reordered together. No conversion between intensity
per frequency and intensity per wavelength is applied. To label the native
intensity units after verifying them for your Multi3D build, use e.g.:

```bash
python3 export_simulations.py --output parvathy_simulations.h5 \
  --intensity-unit 'erg s^-1 cm^-2 Hz^-1 sr^-1'
```

That example is only a label, not a conversion. The default label is explicitly
`unspecified native Multi3D ie units`; it does not guess the physical convention.

Each pair includes provenance, ray direction, transition level numbers and
coordinate transforms as attributes. Root `complete` is true only if every
discovered pair succeeded. `failures_json` lists failures and
`successful_snapshots` records the number exported. Failed groups are removed;
other snapshots are retained, and the process returns exit status 1 if any fail.
Check these attributes before handing over the file.

Example for Parvathy (no Helita dependency needed to read):

```python
import h5py
import numpy as np

with h5py.File('parvathy_simulations.h5', 'r') as f:
    print('Complete:', f.attrs['complete'])
    print('Failures:', f.attrs['failures_json'])
    g = f['simulations/en024048_hion/snapshots/385']
    ltau = g['coordinates/ltau500'][:]
    index = np.flatnonzero(np.isclose(ltau, -5.7))[0]
    bz = g['magnetic_field_bz'][index]       # (x, y), signed kG
    wavelength = g['coordinates/wavelength'][:]
    spectrum = g['halpha_intensity'][10, 20, :]  # one pixel
    # For very large cubes, slice this dataset rather than loading it all:
    intensity_plane = g['halpha_intensity'][:, :, 0]
```

The actual cluster export must be run where the simulation files are accessible.

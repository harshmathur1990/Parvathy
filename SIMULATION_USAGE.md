# Bifrost / Multi3D H-alpha plots

Copy `plot_simulations.py` and `plot_halpha.py` into the same directory on the
cluster. Use an environment with numpy, scipy, matplotlib and your helita fork
(the fork must provide `Multi3dOut.readtau500`).

Process every discovered simulation/snapshot pair:

```bash
python3 plot_simulations.py --no-show
```

The default roots are the two cluster paths supplied in the conversation.
The script finds `<output-directory>/<numeric-snapshot>/H` folders and generates
one 2x2 PNG per pair:

| Position | Full H-alpha core width | log10(tau500) for B_LOS |
| --- | --- | --- |
| Top left | 0.15 angstrom | -5.7 |
| Top right | 0.6 angstrom | -5.7 |
| Bottom left | 0.6 angstrom | -1 |
| Bottom right | 1.6 angstrom | -1 |

For the required vertical ray, the plotted field is the absolute line-of-sight
component, |B_LOS| = |B_z|, in kG. Core-width and ltau arguments are no longer
needed; these four combinations are fixed in the PANELS constant.

Optional filters:

```bash
python3 plot_simulations.py --snap 385 --no-show
python3 plot_simulations.py --simulations en024048_hion ch012012_hion --no-show
python3 plot_simulations.py --list-available
```

Base simulation filters also include their numeric resolution-suffix outputs.
For example, `en024048_hion_504` automatically maps to `en024048_hion` when
that base run directory exists. Other aliases require a manifest.

```bash
python3 plot_simulations.py --manifest simulations.example.json --no-show
```

The script processes each pair independently. Unreadable/incompatible pairs
are reported and skipped; exit status 1 signals any pair failure. A panel with
no valid optical-depth pixels is annotated, and the other panels are retained.

The example maps the `en024048_hion_504` Multi3D output to the
`en024048_hion` Bifrost run. Verify this association yourself before using it.
A manifest entry can override `bifrost_name`, `output_name`, `run_directory`,
`h_directory`, `coordinate_signs`, `coordinate_offsets_cm`, and `grid_slices`.
For explicit slices, use e.g. `"grid_slices": {"x": [0, 504, 2],
"y": [0, 504, 2], "z": [0, 1000, 4]}` (start, stop, step).

## Calculations and assumptions

- Read H-alpha using `set_transition(3, 2)` and `readvar('ie')`. Level numbers
  can be overridden for a different hydrogen atom model. Frequencies must be
  complete for this transition and the chosen ray must be vertical.
- Read positive tau500, take log10, and interpolate a height for the requested
  ltau in each column. No depth extrapolation is allowed. Columns with missing
  or ambiguous surfaces are excluded.
- Read bz as a horizontal slice of its memory map. Center in blocks of at most
  16 selected x columns at native vertical resolution, then apply the inferred
  z slice. This retains the native vertical neighbors required by the stagger
  operator without materializing the complete original cube. The stored field
  has the Multi3D grid shape.
- Center the staggered Bifrost bz with `do_stagger(..., 'zup')`, multiply by
  the snapshot's `u_b` (simulation field to gauss), and convert to kilogauss.
- Match grids using physical coordinates, not array sizes. The default assumes
  the conversion in BifrostData.write_multi3d: x stays positive, y and z change
  sign, and coordinates are in cm. Infer each origin-zero slice from the ratio
  of original to output dimensions: nearest integer stride (bounded so it fits),
  stop = output size times stride. Validate the selected physical coordinates
  against Multi3D before reading the field. Shapes alone cannot uniquely identify
  a crop; conflicting coordinates require explicit grid_slices or offsets.
  Matching bounds alone
  cannot establish that the selected outputs originate from the same snapshot.
- Average intensity by integrating the piecewise-linear wavelength profile
  over the requested full core width, including interpolated endpoints.
- Normalize with one quiet-pixel continuum median. Quiet pixels satisfy
  |Bz| < 0.01 kG on the ltau=0 surface. Continuum uses the nearest samples at
  6562.75 +/- 4 angstroms, preserving the original normalization method.
- Preserve Multi3D's emergent-intensity convention (`ie`); no conversion from
  intensity per frequency to intensity per wavelength is applied.
- Compute the mean line index in magnetic-field bins, including the largest
  field value in the last bin.

Outputs go into `halpha_plots/`: `<simulation>_snap<snapshot>_panels.png`
and a matching compressed NPZ. The NPZ stores three line-index maps, two
magnetic-field surface maps, normalization, snapshot, and per-panel parameters,
bin centers, means and counts. Multi3D/Bifrost are read once per pair and shared
calculations are reused across panels.

Numerical checks cover unit conversions, uneven wavelength integration,
optical-depth surfaces, slice inference, automatic discovery, snapshot filters,
resolution-suffix mapping and four-panel calculations with mock readers.
Actual cluster-file reading, Helita staggering on blocks, coordinate alignment
and real-data rendering remain unverified in this workspace.

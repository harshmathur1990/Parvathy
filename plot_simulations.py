#!/usr/bin/env python3
"""Batch H-alpha plots from Bifrost snapshots and Multi3D H outputs.

Requires numpy, scipy, matplotlib and the user's helita fork with readtau500.
The default coordinate conversion matches BifrostData.write_multi3d:
Multi3D (x,y,z) = Bifrost (x,-y,-z), in cm. Override via a JSON manifest.
All field analysis is in kG; quiet normalization is at log10(tau500)=0.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from plot_halpha import number, magnetic_to_kilogauss, wavelength_average

RUN_ROOT = Path('/mn/stornext/d21/RoCS/matsc/3d/run')
OUTPUT_ROOT = Path('/mn/stornext/d9/data/harshm/bifrost_data')
PANELS = ((0.15, -5.7), (0.6, -5.7), (0.6, -1.0), (1.6, -1.0))


def tau_surface_height(tau, z, target):
    """Interpolate height at log10(tau)=target; ambiguous/missing columns -> NaN.

    No extrapolation. Nonpositive/nonfinite tau samples cannot bracket a surface.
    Multiple distinct crossings are rejected rather than choosing arbitrarily.
    """
    tau = np.asarray(tau)
    z = np.asarray(z, dtype=float)
    if tau.ndim != 3 or tau.shape[-1] != z.size or z.size < 2:
        raise ValueError('tau500 must have shape (x,y,z) matching the depth coordinates.')
    if not np.all(np.isfinite(z)) or not (np.all(np.diff(z) > 0) or np.all(np.diff(z) < 0)):
        raise ValueError('Multi3D z coordinates must be finite and strictly monotonic.')
    heights = np.full(tau.shape[:2], np.nan)
    hits = np.zeros(tau.shape[:2], dtype=int)
    for k in range(z.size - 1):
        with np.errstate(divide='ignore', invalid='ignore'):
            left = np.log10(np.where(tau[..., k] > 0, tau[..., k], np.nan))
            right = np.log10(np.where(tau[..., k+1] > 0, tau[..., k+1], np.nan))
        # Strict brackets; exact sample matches are handled separately.
        crossing = np.isfinite(left) & np.isfinite(right) & (
            ((left < target) & (target < right)) |
            ((right < target) & (target < left)))
        with np.errstate(divide='ignore', invalid='ignore'):
            candidate = z[k] + (target-left) / (right-left) * (z[k+1]-z[k])
        heights[crossing] = candidate[crossing]
        hits[crossing] += 1
    # More than one exact match (including a plateau) is ambiguous.
    for k in range(z.size):
        with np.errstate(divide='ignore', invalid='ignore'):
            equal = (tau[..., k] > 0) & (np.log10(tau[..., k]) == target)
        heights[equal] = z[k]
        hits[equal] += 1
    heights[hits != 1] = np.nan
    return heights


def infer_grid_slice(original_size, output_size):
    """Infer an origin-zero crop/downsample from sizes using the user's rule.

    Sizes alone are ambiguous; physical coordinate validation is also required.
    """
    if not 0 < output_size <= original_size:
        raise ValueError(f'Cannot slice {original_size} samples to {output_size}.')
    step = max(1, int(np.floor(original_size / output_size + 0.5)))
    step = min(step, original_size // output_size)
    return slice(0, output_size * step, step)


def make_field_sampler(data, geometry, config):
    """Center B_z, convert to kG, and align by physical coordinates in cm."""
    from scipy.interpolate import RegularGridInterpolator
    from helita.sim.bifrost import do_stagger
    # Read metadata/coordinates first; no field cube is materialized here.
    full_coords = data.get_coords(units='cgs', axes='xyz')
    target_coords = [np.asarray(getattr(geometry, axis)) for axis in 'xyz']
    slices = [infer_grid_slice(len(original), len(target))
              for original, target in zip(full_coords, target_coords)]
    overrides = config.get('grid_slices', {})
    for axis, name in enumerate('xyz'):
        if name in overrides:
            slices[axis] = slice(*overrides[name])
        if len(range(*slices[axis].indices(len(full_coords[axis])))) != len(target_coords[axis]):
            raise ValueError(f'{name} slice does not match the Multi3D dimension.')
    signs = config.get('coordinate_signs', [1, -1, -1])
    offsets = config.get('coordinate_offsets_cm', [0, 0, 0])
    if len(signs) != 3 or any(s not in (-1, 1) for s in signs) or len(offsets) != 3:
        raise ValueError('coordinate_signs requires three +/-1 values; offsets requires three cm values.')
    coords = [np.asarray(values)[selection] for values, selection in zip(full_coords, slices)]
    for name, values, target, sign, offset in zip('xyz', coords, target_coords, signs, offsets):
        converted = values * sign + float(offset)
        tolerance = max(np.max(np.abs(converted)), 1.) * 1e-5
        if not np.allclose(converted, target, rtol=0, atol=tolerance):
            raise ValueError(f'Inferred {name} slice disagrees with Multi3D coordinates. Set grid_slices/signs/offsets in the manifest.')
    print('Inferred Bifrost slices: ' + ', '.join(f'{name}={selection}' for name, selection in zip('xyz', slices)))
    # get_var returns a memory-mapped view for bz. Keep native z spacing until
    # zup has centered the field; staggering an already decimated z is incorrect.
    raw = data.get_var('bz', iix=slices[0], iiy=slices[1],
                       iiz=slice(None), printing_stats=False)
    gauss_factor = float(data.params['u_b'][data.snapInd])
    if not np.isfinite(gauss_factor) or gauss_factor <= 0:
        raise ValueError('Bifrost u_b must be a finite positive conversion to gauss.')
    field = np.empty(tuple(len(c) for c in coords), dtype=float)
    # Only a small horizontal block at native vertical resolution is copied.
    # The final resident field has exactly the Multi3D grid shape.
    for start in range(0, raw.shape[0], 16):
        block = np.array(raw[start:start+16], copy=True, order='F')
        centered = do_stagger(block, 'zup', obj=data)
        field[start:start+16] = magnetic_to_kilogauss(
            centered[..., slices[2]] * gauss_factor, 'gauss')
    source_axes = []
    for axis, (values, sign, offset) in enumerate(zip(coords, signs, offsets)):
        values = np.asarray(values, dtype=float) * sign + float(offset)
        if values.size != field.shape[axis] or not np.all(np.isfinite(values)):
            raise ValueError('Centered B_z dimensions do not match finite Bifrost coordinates.')
        if np.all(np.diff(values) < 0):
            values = values[::-1]
            field = np.flip(field, axis=axis)
        if values.size < 2 or not np.all(np.diff(values) > 0):
            raise ValueError('Bifrost coordinates must be strictly monotonic.')
        source_axes.append(values)
    sampler = RegularGridInterpolator(source_axes, field, bounds_error=True)
    x, y = np.asarray(geometry.x), np.asarray(geometry.y)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    def sample(height):
        result = np.full(height.shape, np.nan)
        valid = np.isfinite(height)
        if np.any(valid):
            points = np.column_stack((xx[valid], yy[valid], height[valid]))
            # Mesh files are printed with limited precision: tolerate rounding only.
            for axis, grid in enumerate(source_axes):
                tolerance = max(np.max(np.abs(grid)), 1.) * 1e-5
                if np.any(points[:, axis] < grid[0]-tolerance) or np.any(points[:, axis] > grid[-1]+tolerance):
                    raise ValueError('Multi3D surface lies outside Bifrost coordinates. Check coordinate signs, offsets, and simulation mapping.')
                points[:, axis] = np.clip(points[:, axis], grid[0], grid[-1])
            result[valid] = sampler(points)
        return result
    return sample


def process(simulation, snap, args, config, axes):
    from helita.sim.bifrost import BifrostData
    from helita.sim.multi3d import Multi3dOut
    root_name = config.get('bifrost_name', simulation)
    run_directory = Path(config.get('run_directory', args.run_root / root_name))
    output_name = config.get('output_name', simulation)
    h_directory = Path(config.get('h_directory', args.outputs_root / output_name / str(snap) / 'H'))
    if not h_directory.is_dir():
        raise ValueError(f'Missing H output directory: {h_directory}')
    multi = Multi3dOut(directory=str(h_directory), printinfo=False)
    # Load only the metadata needed for tau and emergent intensity.
    multi.readinput()
    multi.readpar()
    multi.readnu()
    tau = multi.readtau500()
    multi.set_transition(args.upper_level, args.lower_level, ang=args.angle)
    # The reader assumes all transition frequencies were written contiguously.
    expected = np.arange(multi.d.ired, multi.d.ired + multi.d.nnu)
    positions = np.flatnonzero(np.isin(multi.outff, expected))
    if positions.size != expected.size or not np.array_equal(multi.outff[positions], expected) or not np.all(np.diff(positions) == 1):
        raise ValueError('H-alpha output frequencies are incomplete or noncontiguous; cannot use readvar safely.')
    if not np.isclose(multi.theinput['muxout'][args.angle], 0) or not np.isclose(multi.theinput['muyout'][args.angle], 0) or not np.isclose(abs(multi.theinput['muzout'][args.angle]), 1):
        raise ValueError('This pixel-column comparison requires a vertical emergent-intensity ray.')
    intensity = multi.readvar('ie')
    wave = np.asarray(multi.d.l.to_value('Angstrom'))
    order = np.argsort(wave)
    wave = wave[order]
    if intensity.shape != (multi.geometry.nx, multi.geometry.ny, wave.size):
        raise ValueError('Emergent intensity shape does not match the Multi3D grid.')
    def plane(index):
        return intensity[..., int(order[index])]
    if wave[0] > args.center-4 or wave[-1] < args.center+4:
        raise ValueError('H-alpha output must cover the continuum references at center +/-4 angstroms.')
    # Preserve the original nearest-sample continuum reference definition.
    continuum = (np.asarray(plane(np.argmin(abs(wave-(args.center-4)))), dtype=float)
                 + np.asarray(plane(np.argmin(abs(wave-(args.center+4)))), dtype=float))/2
    data = BifrostData(root_name, snap=snap, fdir=str(run_directory), units_output='simu')
    sample = make_field_sampler(data, multi.geometry, config)
    quiet_field = sample(tau_surface_height(tau, multi.geometry.z, 0.0))
    quiet = np.isfinite(quiet_field) & (abs(quiet_field) < 0.01) & np.isfinite(continuum)
    if not np.any(quiet):
        raise ValueError('No valid quiet pixels at log10(tau500)=0 with |B_z| <0.01 kG.')
    normalization = np.median(continuum[quiet])
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError('Quiet continuum normalization is not positive and finite.')
    fields = {ltau: sample(tau_surface_height(tau, multi.geometry.z, ltau))
              for ltau in sorted({ltau for _, ltau in PANELS})}
    indices = {width: wavelength_average(wave, plane,
               args.center-width/2, args.center+width/2) / normalization
               for width in sorted({width for width, _ in PANELS})}
    results = {'normalization': normalization, 'snapshot': snap}
    for panel, (ax, (width, ltau)) in enumerate(zip(np.asarray(axes).flat, PANELS), 1):
        field, line_index = fields[ltau], indices[width]
        valid = np.isfinite(field) & np.isfinite(line_index)
        ax.set_xlabel(r'$|B_\mathrm{LOS}|$ (kG)')
        ax.set_ylabel('Hα Line Index')
        ax.set_title(f'Core width {width:g} Å; log₁₀(τ₅₀₀)={ltau:g}')
        b, li = abs(field[valid]), line_index[valid]
        if not b.size:
            centers = means = np.array([])
            counts = np.array([], dtype=int)
            ax.text(0.5, 0.5, 'No valid pixels at this optical depth',
                    ha='center', va='center', transform=ax.transAxes)
        elif b.max() == 0:
            centers, means, counts = np.array([0.]), np.array([li.mean()]), np.array([li.size])
        else:
            edges = np.linspace(0, b.max(), args.bins+1)
            counts, _ = np.histogram(b, bins=edges)
            sums, _ = np.histogram(b, bins=edges, weights=li)
            populated = counts > 0
            centers = ((edges[:-1]+edges[1:])/2)[populated]
            means = sums[populated]/counts[populated]
            counts = counts[populated]
        ax.plot(centers, means, '.', markersize=4)
        prefix = f'panel{panel}_'
        results.update({prefix+'bin_centers_kG': centers, prefix+'bin_mean': means,
                        prefix+'bin_count': counts, prefix+'ltau': ltau,
                        prefix+'core_width_angstrom': width})
        print(f'{simulation} snap={snap}: panel {panel}, width={width:g}, ltau={ltau:g}: {valid.sum()}/{valid.size} valid pixels')
    for ltau, field in fields.items():
        results[f'field_kG_ltau{ltau:g}'] = field
    for width, line_index in indices.items():
        results[f'line_index_width{width:g}'] = line_index
    np.savez_compressed(args.output_dir / f'{simulation}_snap{snap}_panels.npz', **results)


def discover_outputs(args, manifest):
    """Discover numeric snapshot/H directories and match them to Bifrost runs."""
    if not args.outputs_root.is_dir():
        raise ValueError(f'Output root not found: {args.outputs_root}')
    runs = {path.name for path in args.run_root.iterdir() if path.is_dir()} if args.run_root.is_dir() else set()
    tasks = []
    for folder in sorted(args.outputs_root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        config = dict(manifest.get(name, {}))
        root_name = config.get('bifrost_name', name)
        # Match numeric resolution suffixes to an existing base simulation.
        if root_name not in runs and 'bifrost_name' not in config:
            match = re.fullmatch(r'(.+)_([0-9]+)', name)
            if match and match.group(1) in runs:
                root_name = match.group(1)
        config['bifrost_name'] = root_name
        if args.simulations and name not in args.simulations and root_name not in args.simulations:
            continue
        output_folder = args.outputs_root / config.get('output_name', name)
        for snapshot_dir in sorted(output_folder.iterdir(), key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name)):
            if not snapshot_dir.is_dir() or not snapshot_dir.name.isdigit():
                continue
            snap = int(snapshot_dir.name)
            if args.snap is not None and snap != args.snap:
                continue
            h_directory = Path(config.get('h_directory', snapshot_dir / 'H'))
            if not h_directory.is_dir():
                continue
            entry = dict(config, h_directory=str(h_directory))
            tasks.append((name, snap, entry))
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulations', nargs='+', help='Optional simulation filter; default discovers all outputs')
    p.add_argument('--snap', type=int, help='Optional snapshot filter; default processes all available snapshots')
    p.add_argument('--run-root', type=Path, default=RUN_ROOT)
    p.add_argument('--outputs-root', type=Path, default=OUTPUT_ROOT)
    p.add_argument('--output-dir', type=Path, default=Path('halpha_plots'))
    p.add_argument('--manifest', type=Path, help='JSON per-simulation directory and coordinate overrides')
    p.add_argument('--list-available', action='store_true', help='List discovered simulation/snapshot pairs and exit')
    p.add_argument('--center', type=number, default=6562.75)
    p.add_argument('--upper-level', type=int, default=3)
    p.add_argument('--lower-level', type=int, default=2)
    p.add_argument('--angle', type=int, default=0)
    p.add_argument('--bins', type=int, default=199)
    p.add_argument('--no-show', action='store_true')
    args = p.parse_args()
    if args.bins < 1 or args.angle < 0:
        p.error('Bins must be positive; angle must be nonnegative')
    try:
        manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
        tasks = discover_outputs(args, manifest)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if not tasks:
        p.error('No simulation/snapshot H directories match the selected filters.')
    if args.list_available:
        for name, snap, config in tasks:
            print(f'{name}  snap={snap}  Bifrost={config["bifrost_name"]}  H={config["h_directory"]}')
        return
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    successes, failures = [], []
    for name, snap, config in tasks:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        try:
            process(name, snap, args, config, axes)
            fig.suptitle(f'{name} — snapshot {snap}')
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            image = args.output_dir / f'{name}_snap{snap}_panels.png'
            fig.savefig(image, dpi=200)
            print(f'Saved {image.resolve()}')
            successes.append((name, snap))
            if not args.no_show:
                plt.show()
        except (OSError, ValueError, KeyError, IndexError, RuntimeError, ImportError) as exc:
            print(f'SKIPPED {name} snap={snap}: {exc}')
            failures.append((name, snap))
        finally:
            plt.close(fig)
    print(f'Finished: {len(successes)} figures saved; {len(failures)} pairs failed.')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

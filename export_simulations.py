#!/usr/bin/env python3
"""Export all discovered Bifrost/Multi3D snapshots to a single HDF5 file.

Keep plot_simulations.py and plot_halpha.py beside this script.
Requires the user's helita fork, numpy, scipy and h5py. No plots are generated.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from plot_simulations import (RUN_ROOT, OUTPUT_ROOT, discover_outputs,
                              make_field_sampler, tau_surface_height)

LTAU = (0.0, -1.0, -5.1, -5.7)


def export_snapshot(destination, name, snap, config, args):
    from helita.sim.bifrost import BifrostData
    from helita.sim.multi3d import Multi3dOut
    multi = Multi3dOut(directory=config['h_directory'], printinfo=False)
    multi.readinput()
    multi.readpar()
    multi.readnu()
    tau = multi.readtau500()
    multi.set_transition(args.upper_level, args.lower_level, ang=args.angle)
    expected = np.arange(multi.d.ired, multi.d.ired + multi.d.nnu)
    positions = np.flatnonzero(np.isin(multi.outff, expected))
    if (positions.size != expected.size or
        not np.array_equal(multi.outff[positions], expected) or
        not np.all(np.diff(positions) == 1)):
        raise ValueError('H-alpha frequencies are incomplete or noncontiguous.')
    ray = np.array([multi.theinput[key][args.angle]
                    for key in ('muxout', 'muyout', 'muzout')])
    if not np.allclose(ray[:2], 0) or not np.isclose(abs(ray[2]), 1):
        raise ValueError('Export requires a vertical ray for matching intensity pixels to field columns.')
    intensity = multi.readvar('ie')
    wavelength = np.asarray(multi.d.l.to_value('Angstrom'))
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    if not np.all(np.isfinite(wavelength)) or not np.all(np.diff(wavelength) > 0):
        raise ValueError('Wavelengths must be finite and distinct.')
    shape = (multi.geometry.nx, multi.geometry.ny, wavelength.size)
    if intensity.shape != shape:
        raise ValueError('Intensity dimensions do not match geometry/wavelengths.')
    root_name = config['bifrost_name']
    run_directory = Path(config.get('run_directory', args.run_root / root_name))
    data = BifrostData(root_name, snap=snap, fdir=str(run_directory), units_output='simu')
    sample = make_field_sampler(data, multi.geometry, config)
    group = destination.create_group(f'simulations/{name}/snapshots/{snap}')
    group.attrs.update(snapshot=snap, bifrost_name=root_name,
                       bifrost_directory=str(run_directory), h_directory=config['h_directory'],
                       hydrogen_upper_level=args.upper_level, hydrogen_lower_level=args.lower_level,
                       intensity_normalization='none; original Multi3D ie values',
                       ray_direction_cosines=ray,
                       coordinate_signs=config.get('coordinate_signs', [1, -1, -1]),
                       coordinate_offsets_cm=config.get('coordinate_offsets_cm', [0, 0, 0]))
    coords = group.create_group('coordinates')
    scales = []
    for axis in 'xy':
        ds = coords.create_dataset(axis, data=np.asarray(getattr(multi.geometry, axis)))
        ds.attrs['units'] = 'cm'
        ds.make_scale(axis)
        scales.append(ds)
    wave = coords.create_dataset('wavelength', data=wavelength)
    wave.attrs['units'] = 'angstrom'
    wave.make_scale('wavelength')
    depth = coords.create_dataset('ltau500', data=LTAU)
    depth.attrs.update(units='dimensionless', definition='log10(tau500)')
    depth.make_scale('ltau500')
    intensity_ds = group.create_dataset('halpha_intensity', shape=shape, dtype=intensity.dtype,
                          chunks=(min(shape[0],64), min(shape[1],64), 1),
                          compression='gzip', compression_opts=4, shuffle=True)
    intensity_ds.attrs.update(units=args.intensity_unit,
                              definition='Unnormalized emergent intensity ie; no frequency/wavelength density conversion',
                              axes='x,y,wavelength')
    for axis, scale in enumerate(scales+[wave]):
        intensity_ds.dims[axis].attach_scale(scale)
        intensity_ds.dims[axis].label = ('x','y','wavelength')[axis]
    # Stream one wavelength plane to avoid loading the whole intensity cube.
    for index, original_index in enumerate(order):
        intensity_ds[..., index] = intensity[..., int(original_index)]
    maps = group.create_dataset('magnetic_field_bz', shape=(len(LTAU),shape[0],shape[1]),
                               dtype='f4', compression='gzip', shuffle=True)
    maps.attrs.update(units='kG', axes='ltau500,x,y',
                      definition='Signed, cell-centered Bifrost B_z interpolated at each log10(tau500) surface',
                      sign_convention='Original Bifrost bz sign; positive along original Bifrost +z, not Multi3D +z',
                      missing_values='NaN for absent or ambiguous optical-depth surfaces')
    valid = group.create_dataset('magnetic_field_valid', shape=maps.shape, dtype='bool', compression='gzip')
    for ds in (maps, valid):
        for axis, scale in enumerate([depth]+scales):
            ds.dims[axis].attach_scale(scale)
            ds.dims[axis].label = ('ltau500','x','y')[axis]
    for index, ltau in enumerate(LTAU):
        field = sample(tau_surface_height(tau, multi.geometry.z, ltau))
        maps[index] = field
        valid[index] = np.isfinite(field)
        print(f'{name} snap={snap} ltau={ltau:g}: {np.isfinite(field).sum()}/{field.size} valid pixels')
    group.attrs['complete'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('parvathy_simulations.h5'))
    p.add_argument('--run-root', type=Path, default=RUN_ROOT)
    p.add_argument('--outputs-root', type=Path, default=OUTPUT_ROOT)
    p.add_argument('--simulations', nargs='+')
    p.add_argument('--snap', type=int)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--upper-level', type=int, default=3)
    p.add_argument('--lower-level', type=int, default=2)
    p.add_argument('--angle', type=int, default=0)
    p.add_argument('--intensity-unit', default='unspecified native Multi3D ie units',
                   help='Unit label from your Multi3D build; values are preserved without conversion')
    args = p.parse_args()
    if args.angle < 0:
        p.error('--angle must be nonnegative')
    try:
        import h5py
        manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
        tasks = discover_outputs(args, manifest)
        if not tasks:
            p.error('No simulation/snapshot H directories found.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects an existing exported dataset from overwrite.
        failures = []
        with h5py.File(args.output, 'x') as f:
            f.attrs.update(schema_version='1.0', created_utc=datetime.now(timezone.utc).isoformat(),
                           description='H-alpha spectral cubes and optical-depth magnetic field maps',
                           ltau500_values=LTAU, complete=False)
            for name, snap, config in tasks:
                try:
                    export_snapshot(f, name, snap, config, args)
                except (OSError, ValueError, KeyError, IndexError, RuntimeError, ImportError) as exc:
                    path = f'simulations/{name}/snapshots/{snap}'
                    if path in f:
                        del f[path]
                    failures.append({'simulation':name,'snapshot':snap,'error':str(exc)})
                    print(f'FAILED {name} snap={snap}: {exc}')
                f.flush()
            f.attrs['failures_json'] = json.dumps(failures)
            f.attrs['successful_snapshots'] = len(tasks)-len(failures)
            f.attrs['complete'] = not failures
        print(f'Saved {args.output.resolve()}: {len(tasks)-len(failures)} snapshots; {len(failures)} failures.')
        if failures:
            raise SystemExit(1)
    except (ImportError, OSError, ValueError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()

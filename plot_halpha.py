#!/usr/bin/env python3
"""Plot H-alpha line index versus |B_z| for a chosen core width and log(tau).

Core width is the FULL width in angstroms. Quiet-pixel continuum normalization
uses the nearest log(tau)=0 layer, matching the supplied notebook.
The line index is the wavelength integral divided by the full core width.
Magnetic inputs are converted to kG using file units or --field-unit.
Requires: numpy, h5py, matplotlib (pip install numpy h5py matplotlib).
"""
import argparse
from pathlib import Path
import math

MAGNETIC_FILE = 'BIFROST_en024048_hion_0_504_0_504_ltau_mag_field.nc'
PROFILES_FILE = 'BIFROST_en024048_hion_0_504_0_504_supplementary_outputs_multi3d_pops_pb_rates.nc'


def number(value):
    """Accept decimal points or decimal commas, e.g. -4,5 means -4.5."""
    result = float(str(value).strip().replace(',', '.'))
    if not math.isfinite(result):
        raise ValueError('Enter a finite number.')
    return result


def magnetic_to_kilogauss(values, unit):
    """Convert a field with an explicit supported unit to internal kG units."""
    import numpy as np
    if isinstance(unit, bytes):
        unit = unit.decode('utf-8')
    name = str(unit).strip().lower().replace(' ', '')
    factors = {'g': 0.001, 'gauss': 0.001, 'kg': 1.0, 'kilogauss': 1.0,
               't': 10.0, 'tesla': 10.0, 'mt': 0.01, 'millitesla': 0.01}
    if name not in factors:
        raise ValueError(f'Unknown magnetic-field unit {unit!r}. Use --field-unit gauss, kilogauss, tesla, or millitesla.')
    return np.asarray(values, dtype=float) * factors[name]


def wavelength_average(wavelength, read_plane, low, high):
    """Integrate the piecewise-linear spectrum, including exact band endpoints.

    read_plane(index) returns one intensity map at the given wavelength index.
    Only two maps need to be retained during integration.
    """
    import numpy as np
    wave = np.asarray(wavelength, dtype=float)
    if wave.ndim != 1 or wave.size < 2 or not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        raise ValueError('Wavelength grid must contain at least two finite, strictly increasing values.')
    if not wave[0] <= low < high <= wave[-1]:
        raise ValueError('The requested core interval extends beyond the wavelength grid.')
    def interpolate(x):
        right = int(np.searchsorted(wave, x, side='left'))
        if wave[right] == x:
            return np.asarray(read_plane(right), dtype=float)
        left = right - 1
        fraction = (x - wave[left]) / (wave[right] - wave[left])
        return ((1 - fraction) * np.asarray(read_plane(left), dtype=float)
                + fraction * np.asarray(read_plane(right), dtype=float))
    previous_x, previous_y = low, interpolate(low)
    integral = np.zeros_like(previous_y)
    for index in np.flatnonzero((wave > low) & (wave < high)):
        current_x = wave[index]
        current_y = np.asarray(read_plane(int(index)), dtype=float)
        integral += (current_x - previous_x) * (previous_y + current_y) / 2
        previous_x, previous_y = current_x, current_y
    integral += (high - previous_x) * (previous_y + interpolate(high)) / 2
    return integral / (high - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-width', type=number, help='Full core width in angstroms, e.g. 0.6')
    parser.add_argument('--ltau', type=number, help='Requested log(tau500), e.g. -2 or -4.5')
    parser.add_argument('--magnetic-file', type=Path, default=Path(MAGNETIC_FILE))
    parser.add_argument('--field-unit', default='auto', help='Input B_z unit: auto reads its units attribute; or specify gauss, kilogauss, tesla, millitesla (G, kG, T, mT also accepted)')
    parser.add_argument('--profiles-file', type=Path, default=Path(PROFILES_FILE))
    parser.add_argument('--output', type=Path, help='Output image path (default: parameter-based PNG name)')
    parser.add_argument('--no-show', action='store_true', help='Save the plot without opening a window')
    args = parser.parse_args()
    try:
        width = args.core_width if args.core_width is not None else number(input('Core width in angstroms (e.g. 0.6): '))
        requested_tau = args.ltau if args.ltau is not None else number(input('ltau500 (e.g. -1, -2, -4.5): '))
        if width <= 0:
            raise ValueError('Core width must be greater than zero.')
        for path in (args.magnetic_file, args.profiles_file):
            if not path.is_file():
                raise ValueError(f'Data file not found: {path}. Supply its path with --magnetic-file or --profiles-file.')
    except (ValueError, EOFError) as exc:
        parser.error(str(exc))

    try:
        import numpy as np
        import h5py
        import matplotlib
        if args.no_show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        parser.error(f'{exc}. Install dependencies: pip install numpy h5py matplotlib')

    try:
        with h5py.File(args.magnetic_file, 'r') as f:
            tau = np.asarray(f['ltau500'][:]).squeeze()
            bz = f['B_z']
            if tau.ndim != 1 or not np.all(np.isfinite(tau)):
                raise ValueError('ltau500 must be a finite one-dimensional depth grid.')
            if bz.ndim != 3 or bz.shape[-1] != tau.size:
                raise ValueError('Expected B_z shape (x, y, depth), matching ltau500.')
            if not tau.min() <= requested_tau <= tau.max():
                raise ValueError(f'ltau must lie between {tau.min():g} and {tau.max():g}.')
            layer = int(np.argmin(np.abs(tau - requested_tau)))
            quiet_layer = int(np.argmin(np.abs(tau)))
            field_unit = bz.attrs.get('units') if args.field_unit.lower() == 'auto' else args.field_unit
            if field_unit is None:
                raise ValueError('B_z has no units attribute. Specify --field-unit explicitly (e.g. gauss or tesla).')
            b = magnetic_to_kilogauss(bz[:, :, layer], field_unit)
            quiet_b = magnetic_to_kilogauss(bz[:, :, quiet_layer], field_unit)
            selected_tau = float(tau[layer])
        quiet_mask = np.isfinite(quiet_b) & (np.abs(quiet_b) < 0.01)
        if not np.any(quiet_mask):
            raise ValueError('No quiet pixels with |B_z| < 10 G at the normalization layer.')

        with h5py.File(args.profiles_file, 'r') as f:
            wavelength = np.asarray(f['wave_H'][:]).squeeze()
            profiles = f['profiles_H']
            if wavelength.ndim != 1 or not np.all(np.isfinite(wavelength)):
                raise ValueError('wave_H must be a finite one-dimensional wavelength grid in angstroms.')
            if profiles.ndim != 5 or profiles.shape[1:3] != b.shape or profiles.shape[3] != wavelength.size:
                raise ValueError('Expected profiles_H shape (time, x, y, wavelength, Stokes), matching B_z and wave_H.')
            if wavelength.size < 2 or not np.all(np.diff(wavelength) > 0):
                raise ValueError('wave_H must be strictly increasing with at least two samples.')
            center = 6562.75
            low, high = center - width / 2, center + width / 2
            if low < wavelength.min() or high > wavelength.max():
                raise ValueError('The requested core interval extends beyond the wavelength grid.')
            indices = np.flatnonzero((wavelength >= low) & (wavelength <= high))
            if wavelength.min() > center - 4 or wavelength.max() < center + 4:
                raise ValueError('The grid must include the continuum wavelengths at H-alpha +/- 4 angstroms.')
            minus = int(np.argmin(np.abs(wavelength - (center - 4))))
            plus = int(np.argmin(np.abs(wavelength - (center + 4))))
            continuum = (np.asarray(profiles[0, :, :, minus, 0], dtype=float)
                         + np.asarray(profiles[0, :, :, plus, 0], dtype=float)) / 2
            quiet_continuum = continuum[quiet_mask]
            quiet_continuum = quiet_continuum[np.isfinite(quiet_continuum)]
            normalization = np.median(quiet_continuum) if quiet_continuum.size else np.nan
            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError('Quiet-pixel continuum median must be finite and positive.')
            # Interpolate band endpoints and weight by actual wavelength spacing.
            line_index = wavelength_average(
                wavelength, lambda index: profiles[0, :, :, index, 0], low, high
            ) / normalization

        valid = np.isfinite(b) & np.isfinite(line_index)
        field, intensity = np.abs(b[valid]), line_index[valid]
        if not field.size:
            raise ValueError('No finite magnetic-field and line-index pairs.')
        if field.max() == 0:
            centers, means = np.array([0.0]), np.array([intensity.mean()])
        else:
            edges = np.linspace(0, field.max(), 200)
            counts, _ = np.histogram(field, bins=edges)
            sums, _ = np.histogram(field, bins=edges, weights=intensity)
            populated = counts > 0
            centers = ((edges[:-1] + edges[1:]) / 2)[populated]
            means = sums[populated] / counts[populated]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(centers, means, s=20)
        ax.set_xlabel(r'$|B_z|$ (kG)')
        ax.set_ylabel('Hα Line Index')
        ax.set_title(f'Hα Line Index vs Magnetic Field\nCore width = {width:g} Å; ltau500 = {selected_tau:g}')
        fig.tight_layout()
        output = args.output or Path(f'halpha_core_{width:g}_ltau_{requested_tau:g}.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f'Requested ltau={requested_tau:g}; selected ltau={selected_tau:g} (layer {layer}).')
        print(f'Input magnetic unit: {field_unit!r}; internal and plotted unit: kG.')
        print(f'Core interval: {low:g}–{high:g} Å; {indices.size} wavelength samples.')
        print(f'Quiet normalization layer: ltau={tau[quiet_layer]:g}; median continuum={normalization:g}.')
        print(f'Saved plot: {output.resolve()}')
        if not args.no_show:
            plt.show()
        plt.close(fig)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

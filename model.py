"""Leaky integrate-and-fire dynamics. Units: mV, ms, nA, megaohms."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Parameters:
    rest_mv: float = -65.
    reset_mv: float = -65.
    threshold_mv: float = -50.
    tau_ms: float = 20.
    resistance_mohm: float = 100.
    refractory_ms: float = 2.

def simulate(current_na, p=Parameters(), duration_ms=1000., dt_ms=.1):
    """Forward Euler; events at right grid edges; hold reset for full refractory steps."""
    values = [current_na, duration_ms, dt_ms, *p.__dict__.values()]
    if not np.isfinite(values).all() or dt_ms <= 0 or duration_ms <= 0:
        raise ValueError('Finite parameters and positive time settings required')
    if p.tau_ms <= dt_ms or p.resistance_mohm <= 0 or p.refractory_ms < 0:
        raise ValueError('Require tau > dt, positive resistance and nonnegative refractory time')
    if p.reset_mv >= p.threshold_mv or p.rest_mv >= p.threshold_mv:
        raise ValueError('Rest and reset must be below threshold')
    steps = int(round(duration_ms / dt_ms))
    hold = int(round(p.refractory_ms / dt_ms))
    if not np.isclose(steps * dt_ms, duration_ms) or not np.isclose(hold * dt_ms, p.refractory_ms):
        raise ValueError('Duration and refractory time must be grid-aligned')
    t = np.arange(steps+1) * dt_ms
    voltage = np.full(steps+1, p.reset_mv)
    spikes, remaining = [], 0
    for k in range(steps):
        if remaining:
            remaining -= 1
            continue
        candidate = voltage[k] + dt_ms/p.tau_ms * (-(voltage[k]-p.rest_mv) + p.resistance_mohm*current_na)
        if candidate >= p.threshold_mv:
            spikes.append(t[k+1])
            remaining = hold
        else:
            voltage[k+1] = candidate
    return t, voltage, np.asarray(spikes)

def analytic_rate(current_na, p=Parameters()):
    vinf = p.rest_mv + p.resistance_mohm * current_na
    if vinf <= p.threshold_mv:
        return 0.
    crossing = p.tau_ms * np.log((vinf-p.reset_mv)/(vinf-p.threshold_mv))
    return 1000. / (p.refractory_ms + crossing)

def measured_rate(spikes, start_ms=100., end_ms=1000.):
    """Count events in (start, end]; denominator in seconds."""
    return float(np.sum((spikes > start_ms) & (spikes <= end_ms))*1000/(end_ms-start_ms))

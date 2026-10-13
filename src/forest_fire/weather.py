"""Weather: how drought and wind change the chance that fire spreads.

Both effects act only through the spread probability q from Phase 2, so the
fire rules themselves do not change.

Dryness (drought, the El Niño lever)
    El Niño years bring prolonged drought to Indonesia, much of Southeast Asia
    and eastern Australia. Dry fuel catches more easily. We describe this with
    a dryness d between 0 and 1 that shrinks the chance that fuel *fails* to
    catch:

        q = 1 - (1 - q_normal) * (1 - d)

    d = 0 is a normal year (q = q_normal) and d = 1 is bone-dry fuel (q = 1).
    For example, q_normal = 0.6 with d = 0.5 gives q = 0.8. This is an
    illustrative mapping, not a calibrated fuel-moisture model.

Wind
    Wind tilts flames forward and pushes hot gas downwind, so fire spreads more
    easily downwind and less easily upwind. Each neighbour gets its own per-step
    probability:

        p_direction = min(1, p * exp(C2 * V * cos(angle)))

    V is the wind speed in m/s, `angle` is between the wind and the direction
    from the burning cell to that neighbour, and C2 = 0.131 s/m is the angular
    wind coefficient of the cellular-automaton fire model of Alexandridis et al.
    (2008), fitted to a real wildfire (Spetses, Greece, 1990). At 10 m/s this
    multiplies the chance of spreading downwind by 3.7, leaves crosswind
    unchanged and cuts upwind spread to 0.27.

    Their model also has an overall factor exp((C1 - C2) * V) that lowers spread
    in every direction as wind rises. It was fitted together with their other
    parameters; in this model it smothers wind-blown fires (stronger wind giving
    smaller fires), so it is left out. Wind works best with multi-step burning
    (spread.py: burn_steps), where per-step chances are small enough that the
    downwind boost is not cut off at probability 1.

    Directions: the wind direction is where the wind blows TO, in degrees,
    0 = towards the right edge (east), 90 = towards the top (north).
"""

import numpy as np

C2 = 0.131  # angular wind coefficient, s/m (Alexandridis et al. 2008)


def dry_spread_probability(q_normal: float, dryness: float) -> float:
    """Spread probability after drought: the failure-to-catch chance shrinks by (1 - dryness)."""
    if not (0.0 <= dryness <= 1.0):
        raise ValueError("dryness must be between 0 and 1")
    return 1.0 - (1.0 - q_normal) * (1.0 - dryness)


def wind_unit_vector(direction_deg: float) -> np.ndarray:
    """(row, column) unit vector the wind blows towards. Rows grow downwards, so north is row -1."""
    a = np.deg2rad(direction_deg)
    return np.array([-np.sin(a), np.cos(a)])


def wind_factor(cos_angle, wind_speed: float):
    """Multiplier on the spread probability for a neighbour at `cos_angle` to the wind (1 = downwind)."""
    return np.exp(C2 * wind_speed * np.asarray(cos_angle))


def direction_probabilities(offsets: np.ndarray, q: float, wind_direction: float = 0.0,
                            wind_speed: float = 0.0) -> np.ndarray:
    """Spread probability towards each neighbour offset, given q and the wind (speed in m/s).

    offsets : (n_neighbours, 2) array of (row, column) steps
    returns : (n_neighbours,) array of probabilities, each capped at 1
    """
    probs = np.full(len(offsets), float(q))
    if wind_speed > 0:
        unit = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)   # diagonals have length sqrt(2)
        probs = np.minimum(1.0, probs * wind_factor(unit @ wind_unit_vector(wind_direction), wind_speed))
    return probs

"""Net 2-D shear extraction from a marker-based GelSight sensor.

Core idea: under linear isotropic elasticity the in-plane displacement caused by
normal load is an odd (radial) field around each contact, so it integrates to
zero over the gel. Shear produces an even field whose integral is (h/G) * F_t.
Summing the whole marker displacement field therefore gives net shear with the
normal-load contribution cancelled, for any number or shape of contacts, as long
as each contact's deformation stays inside the sensor window.
"""
from .config import Config
from .pipeline import Pipeline, Result

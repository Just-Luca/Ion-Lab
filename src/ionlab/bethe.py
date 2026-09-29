"""Bethe stopping power and CSDA range for heavy charged particles in water.

Units: energies in MeV, lengths in cm, mass stopping power in MeV cm^2/g.
Water density is taken as 1 g/cm^3, so mass and linear quantities coincide.
"""
import numpy as np
from scipy.integrate import quad

K = 0.307075          # MeV cm^2 / mol
ME = 0.51099895       # electron mass, MeV
MP = 938.272          # proton mass, MeV
Z_OVER_A = 0.55508    # water, mol/g
I_WATER = 75e-6       # mean excitation energy, MeV (ICRU 49; ICRU 90 uses 78 eV)


def stopping_power(T, M=MP, z=1, I=I_WATER):
    """Mass stopping power in MeV cm^2/g (Bethe, no shell/density corrections)."""
    gamma = 1.0 + T / M
    beta2 = 1.0 - 1.0 / gamma**2
    bg2 = gamma**2 - 1.0                      # beta^2 * gamma^2
    tmax = 2 * ME * bg2 / (1 + 2 * gamma * ME / M + (ME / M) ** 2)
    arg = 2 * ME * bg2 * tmax / I**2
    return K * z**2 * Z_OVER_A / beta2 * (0.5 * np.log(arg) - beta2)


def csda_range(T0, T_min=1.0, **kw):
    """CSDA range in cm, integrating 1/S from T_min to T0 (below T_min ignored)."""
    r, _ = quad(lambda T: 1.0 / stopping_power(T, **kw), T_min, T0)
    return r


if __name__ == "__main__":
    print(f"S(100 MeV) = {stopping_power(100.0):.3f} MeV cm2/g  (PSTAR: 7.289)")
    print(f"R(150 MeV) = {csda_range(150.0):.2f} cm            (PSTAR: ~15.77)")
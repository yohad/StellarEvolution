import astropy.units as u
import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.constants import G, c, e, hbar, k_B, m_e, m_p, sigma_sb
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from pathlib import Path
from tqdm import tqdm

e = e.esu

mu_thermal = m_p / 2
a = 4 * sigma_sb / c


def opacity(X):
    return 0.2 * (1 + X) * u.cm**2 / u.g


def density(M, R):
    return M / (4 / 3 * np.pi * R**3)


def thermal_pressure(rho, T):
    result = rho * k_B * T / mu_thermal
    return result


def radiation_pressure(T):
    result = a / 3 * T**4
    return result


def degeneracy_pressure(rho):
    result = (
        (2 * np.pi**2) ** (2 / 3)
        * hbar**2
        / 5
        / m_e
        * (rho / mu_thermal) ** (5 / 3)
    )
    return result


def total_pressure(X, rho, T):
    return (
        thermal_pressure(rho, T)
        + radiation_pressure(T)
        + degeneracy_pressure(rho)
    )


def calculate_temperature(X, rho, R, M):
    def foo(x):
        x *= u.K
        result = total_pressure(X, rho, x) - G * M * rho / R
        return result.cgs.value

    T = root_scalar(foo, x0=1000)
    return T.root * u.K


def luminosity(T, R, rho, X):
    kappa = opacity(X)
    result = sigma_sb * T**4 * R / (kappa * rho)
    return result


def epsilon(rho, T):
    WEAK = 1e-24
    E0 = ((mu_thermal / 2) ** (1 / 2) * np.pi * e**2 * k_B * T / hbar) ** (
        2 / 3
    )
    sigmav = (
        WEAK
        * 4
        * E0
        * np.exp(-3 * E0 / (k_B * T))
        * np.sqrt(2 / 3 * k_B * T * E0)
        / (np.sqrt(np.pi * mu_thermal * (k_B * T) ** 3))
        * u.fm**2
    )

    n_H = rho / mu_thermal
    result = n_H * sigmav * u.MeV / m_p
    return result.cgs


def simulation(t, y, M):
    R, X = y
    R *= u.Rsun
    M *= u.Msun

    rho = density(M, R)
    T = calculate_temperature(X, rho, R, M)
    L = luminosity(T, R, rho, X)
    eps = epsilon(rho, T)
    dR = R**2 / G / M**2 * (M * eps - L)
    dX = -eps * 4 * m_p / (2.5 * u.MeV)
    diff = [dR.to("Rsun / yr").value, dX.to("1/yr").value]
    return diff


@click.command()
@click.option(
    "--mass", default=1, help="Mass of the Star in [Msun]", type=click.FLOAT
)
@click.option(
    "--output",
    default="output",
    type=click.Path(exists=True),
    help="Mass of the Star in [Msun]",
)
def main(mass, output):
    # Calculations
    print("Integrating in time...")
    result = solve_ivp(
        simulation,
        [0.1, 6e8],
        y0=[100 * mass, 1],
        args=[mass],
        first_step=10,
        dense_output=True,
    )
    R, X = result.y
    R *= u.Rsun

    # Get other quantities
    M = mass * u.Msun
    L = np.zeros_like(X) * u.Lsun
    T = np.zeros_like(X) * u.K
    rho = np.zeros_like(X) * u.g / u.cm**3
    print("Calculating other quantities...")
    for i, _ in tqdm(enumerate(R)):
        rho[i] = density(M, R[i])
        T[i] = calculate_temperature(X[i], rho[i], R[i], M)
        L[i] = luminosity(T[i], R[i], rho[i], X[i])

    # Save data
    output = Path(output)
    output_path = output / f"M{mass}.csv"
    df = pd.DataFrame({"T": T, "L": L, "rho": rho, "R": R, "X": X})
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

import astropy.units as u
import click
import numpy as np
import pandas as pd
from astropy.constants import G, c, e, hbar, k_B, m_e, m_p, sigma_sb
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from pathlib import Path
from tqdm import tqdm

e = e.esu

mu_thermal = m_p / 2
a = 4 * sigma_sb / c
Z = 0.02  # Assume constant heavy elements fraction


def opacity(X, rho, T):
    Y = 1 - X - Z
    k_T = 0.4 * u.cm**2 / u.g
    k_bf = (4e25 * Z * (1 + X) * rho * T ** (-3.5)).cgs.value * u.cm**2 / u.g
    k_ff = (
        (4e22 * (X + Y) * (1 + X) * rho * T ** (-3.5)).cgs.value * u.cm**2 / u.g
    )
    k_H = (
        (1.3e-29 * Z * (1 + X) * rho ** (1 / 2) * T**9).cgs.value
        * u.cm**2
        / u.g
    )
    k_thomson = 0.2 * (1 + X) * u.cm**2 / u.g
    k_H = 0 * u.cm**2 / u.g  # ignoring for now because its so large
    k_total = k_T + k_bf + k_ff + k_H + k_thomson
    return k_total


def density(M, R):
    return M / (4 / 3 * np.pi * R**3)


def thermal_pressure(rho, T, X):
    result = (1 / 2 + 3 * X / 2) * rho * k_B * T / m_p
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
        thermal_pressure(rho, T, X)
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
    kappa = opacity(X, rho, T)
    result = sigma_sb * T**4 * R / (kappa * rho)
    return result


def epsilon_PP(rho, T, X):
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

    n_H = rho * X / mu_thermal
    result = n_H * sigmav * u.MeV / m_p
    return result.cgs


def epsilon_CNO(rho, T, X):
    EM = 1e-7
    mu_thermal = 14 * m_p
    E0 = ((mu_thermal / 2) ** (1 / 2) * np.pi * e**2 * k_B * T / hbar) ** (
        2 / 3
    )
    sigmav = (
        EM
        * 4
        * E0
        * np.exp(-3 * E0 / (k_B * T))
        * np.sqrt(2 / 3 * k_B * T * E0)
        / (np.sqrt(np.pi * mu_thermal * (k_B * T) ** 3))
        * u.fm**2
    )

    n_Z = rho * Z / mu_thermal
    result = n_Z * sigmav * u.MeV / m_p
    return result.cgs


def epsilon(rho, T, X):
    PP = epsilon_PP(rho, T, X)
    CNO = epsilon_CNO(rho, T, X)
    return PP + CNO


def simulation(
    t,
    y,
    M,
    t_arr,
    R_arr,
    X_arr,
    rho_arr,
    T_arr,
    L_arr,
    eps_arr,
    T_eff_arr,
    pbar,
):
    R, X = y
    X = max(X, 0)  # Need to make sure X is nonnegative

    if len(t_arr) > 0:
        pbar.update(t - t_arr[-1])

    t_arr.append(t)
    R_arr.append(R)
    X_arr.append(X)

    R *= u.Rsun
    M *= u.Msun

    rho = density(M, R)
    rho_arr.append(rho.to("g/cm3").value)

    T = calculate_temperature(X, rho, R, M)
    T_arr.append(T.value)

    L = luminosity(T, R, rho, X)
    L_arr.append(L.to("Lsun").value)

    kappa = opacity(X, rho, T)
    T_eff = T / (4 * np.pi * kappa * rho * R) ** (1 / 4)
    T_eff_arr.append(T_eff.to("K").value)

    eps = epsilon(rho, T, X)
    eps_arr.append(eps.to("cm2 / s3").value)

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
@click.option(
    "--tmax",
    default=1e9,
    type=click.FLOAT,
    help="Time to run the simulation in years",
)
def main(mass, output, tmax):
    # Calculations
    t_arr = []
    R_arr = []
    X_arr = []
    rho_arr = []
    T_arr = []
    L_arr = []
    eps_arr = []
    T_eff_arr = []

    with tqdm(total=tmax) as pbar:
        result = solve_ivp(
            simulation,
            [0.1, tmax],
            y0=[100 * mass, 0.98],
            args=[
                mass,
                t_arr,
                R_arr,
                X_arr,
                rho_arr,
                T_arr,
                L_arr,
                eps_arr,
                T_eff_arr,
                pbar,
            ],
            first_step=10,
            method="RK23",
        )
    R, X = result.y
    R *= u.Rsun

    # Save data
    output = Path(output)
    output_path = output / f"M{mass}.csv"
    df = pd.DataFrame(
        {
            "t": t_arr,
            "T": T_arr,
            "T_eff": T_eff_arr,
            "L": L_arr,
            "rho": rho_arr,
            "R": R_arr,
            "X": X_arr,
        }
    )
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

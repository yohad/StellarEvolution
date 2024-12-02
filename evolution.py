# %%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import astropy.units as u
from astropy.constants import G, k_B, sigma_sb, m_e, m_p, c, hbar, e
import plotly.graph_objects as go

e = e.esu

mu_thermal = m_p / 2
a = 4 * sigma_sb / c
# %%


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
        (2 * np.pi**2) ** (2 / 3) * hbar**2 / 5 / m_e * (rho / mu_thermal) ** (5 / 3)
    )
    return result


def total_pressure(X, rho, T):
    return thermal_pressure(rho, T) + radiation_pressure(T) + degeneracy_pressure(rho)


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
    E0 = ((mu_thermal / 2) ** (1 / 2) * np.pi * e**2 * k_B * T / hbar) ** (2 / 3)
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


# %%
masses = [0.01, 0.5, 1, 30]
results = []
for M in masses:
    print(f"Calculating for {M} [M_sun]")
    results.append(
        solve_ivp(simulation, [0.1, 6e8], y0=[10000 * M, 1], args=[M], first_step=10)
    )

# %%
luminosities = []
temperatures = []
T_eff = []
for sim, M in zip(results, masses):
    print(f"M = {M} [Msun]")
    M *= u.Msun
    t = sim.t * u.yr
    R, X = sim.y
    R *= u.Rsun
    rho = density(1 * u.Msun, R)

    T = np.zeros_like(X) * u.K
    L = np.zeros_like(X) * u.Lsun
    kappa = np.zeros_like(X) * u.cm**2 / u.g
    for i in range(len(X)):
        T[i] = calculate_temperature(X[i], rho[i], R[i], M)
        L[i] = luminosity(T[i], R[i], rho[i], X[i])
        kappa[i] = opacity(X[i])

    temperatures.append(T)
    luminosities.append(L)
    l = 1 / 3 / kappa / rho
    tau = R / l
    T_eff.append((T / tau ** (1 / 4)).cgs)

# %%
fig = go.Figure()

for i in range(len(masses)):
    fig.add_trace(
        go.Scatter(x=T_eff[i], y=luminosities[i], name=f"M = {masses[i]} [Msun]")
    )

fig.update_xaxes(type="log", autorange="reversed", title="T_eff [K]")
fig.update_yaxes(type="log", title="L [Lsun]")
fig.update_layout(title=dict(text="Stellar Evolution"))
fig.show()

# %%
fig = go.Figure()

for i in range(len(masses)):
    fig.add_trace(
        go.Scatter(x=results[i].t, y=T_eff[i], name=f"M = {masses[i]} [Msun]")
    )
fig.update_xaxes(type="log", title="t [yr]")
fig.update_yaxes(type="log", title="T [K]")
fig.update_layout(title=dict(text="T_eff(t)"))

fig.show()
# %%
fig = go.Figure()

for i in range(len(masses)):
    fig.add_trace(
        go.Scatter(x=results[i].t, y=results[i].y[0], name=f"M = {masses[i]} [Msun]")
    )
fig.update_xaxes(type="log", title="t [yr]")
fig.update_yaxes(type="log", title="R [Rsun]")
fig.update_layout(title=dict(text="R(t)"))

fig.show()
# %%
fig = go.Figure()

for i in range(len(masses)):
    fig.add_trace(
        go.Scatter(x=results[i].t, y=luminosities[i], name=f"M = {masses[i]} [Msun]")
    )
fig.update_xaxes(type="log", title="t [yr]")
fig.update_yaxes(type="log", title="L [Lsun]")
fig.update_layout(title=dict(text="L(t)"))

fig.show()

# simapp.py
# Launch: streamlit run simapp.py

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


# ============================================================
# i18n
# ============================================================
I18N = {
    "fr": {
        "lang_name": "Français",
        "title": "Simulateur thermo hydraulique - Darcy Radial",
        "tab_sim": "Simulation",
        "tab_interp": "Interprétation",
        "tab_params": "Paramètres expliqués",
        "sidebar": "Paramètres",
        "mode_label": "Mode d'écoulement",
        "mode_map": {
            "matrix": "Milieu intact (Darcy limité par Δp)",
            "eff_auto": "Réservoir stimulé (Q imposé → k requis)",
            "eff_manual": "Réservoir fracturé (k effectif imposé)",
        },
        "run_btn": "Lancer",
        "reset_btn": "Vider le cache",
        "generate_gif": "Générer le GIF",
        "download_gif": "Télécharger le GIF",
        "summary": "Résumé",
        "summary_table_title": "Résumé de la simulation (tableau)",
        "diag": "Diagnostics",
        "plot_Tmean": "Température moyenne (°C)",
        "plot_PV": "PV injecté (Qt / Vp)",
        "plot_plume": "Rayon de plume (m)",
        "plot_profiles": "Profils radiaux ΔT(r) (sans interpolation)",
        "hydraulics": "Hydraulique",
        "geom": "Géométrie",
        "thermal": "Thermique",
        "losses": "Pertes verticales",
        "time_num": "Temps & numérique",
        "stop_pv": "Arrêt PV",
        "display": "Affichage",
        "gif_opts": "Options GIF",
        "ui_hint": "Régle les paramètres à gauche puis clique **Lancer**.",
        "interp_title": "Comment lire les 4 graphiques",
        "interp_bullets": [
            "**Température moyenne** : indique si le réservoir se réchauffe globalement (énergie stockée) ou si le chauffage reste local. Un plateau suggère que pertes/limites hydrauliques dominent.",
            "**PV injecté (Qt/Vp)** : mesure à quel point le débit “remplit” le volume poreux. PV=1 signifie que tu as injecté l’équivalent du volume poreux total (agressif si atteint vite).",
            "**Rayon de plume** : distance atteinte par un seuil de réchauffement (ΔT≥0.5°C et ΔT≥1°C). Si ça plafonne, chauffer plus loin devient physiquement limité.",
            "**Profils ΔT(r)** : montre directement la forme du gradient thermique à des instants donnés. Courbe raide = advection dominante; courbe étalée = diffusion/pertes dominantes.",
        ],
        "params_title": "À quoi servent les paramètres ?",
        "params_intro": "Cet onglet décrit les réglages disponibles dans la barre latérale (sans math).",
        "params_sections": {
            "Hydraulique": [
                ("Mode d'écoulement", "Choisit si le débit est limité par Δp (milieu intact), si k effectif est imposé (fracturé), ou si on calcule k requis pour atteindre un débit cible (stimulé)."),
                ("Δp max (MPa)", "Différence de pression maximale entre le puits et le bord externe. Plus Δp est grand, plus le débit tend à augmenter."),
                ("μ (Pa·s)", "Viscosité du fluide. Plus μ est grand, plus l’écoulement est freiné."),
                ("k matrice (m²)", "Perméabilité de la matrice (milieu intact). Plus k est grand, plus l’eau circule facilement."),
                ("Q cible (L/s)", "Débit souhaité (mode stimulé). Le modèle calcule la perméabilité effective minimale nécessaire pour l’atteindre (dans la limite du Δp)."),
                ("k effectif (m²)", "Perméabilité effective imposée (mode fracturé). Représente l’effet des fractures/stimulation."),
            ],
            "Géométrie": [
                ("R (m)", "Rayon externe du réservoir simulé (frontière à T0)."),
                ("rw (m)", "Rayon du puits (point d’injection)."),
                ("b (m)", "Épaisseur (hauteur) du réservoir. Joue sur le débit et le volume poreux."),
                ("φ (-)", "Porosité : fraction du volume occupée par le fluide (impacte volume poreux et vitesses)."),
            ],
            "Thermique": [
                ("T0 (°C)", "Température initiale du réservoir (et température imposée au bord externe)."),
                ("Tin (°C)", "Température du fluide injecté au niveau du puits (pendant l’injection)."),
                ("Activer pertes", "Ajoute une perte de chaleur vers le haut/bas (simplifiée) : le réservoir revient vers T0 au cours du temps."),
                ("tau (ans)", "Échelle de temps des pertes : tau petit = pertes fortes; tau grand = pertes faibles."),
            ],
            "Temps & numérique": [
                ("Durée totale (ans)", "Durée de la simulation."),
                ("Durée injection (ans)", "Durée pendant laquelle Tin est imposée au puits. Après, on coupe l’injection (front se stabilise/refroidit)."),
                ("Nr", "Nombre de cellules radiales (plus grand = plus fin près du puits, plus lent)."),
                ("dt (jours)", "Pas de temps. Plus petit = plus stable/précis, mais plus long à calculer."),
                ("Snapshot (jours)", "Fréquence d’enregistrement des profils (courbes et GIF)."),
            ],
            "Arrêt PV": [
                ("Stop quand PV>=seuil", "Arrête la simulation quand le volume injecté atteint un multiple du volume poreux (PV)."),
                ("Seuil PV", "Valeur PV d’arrêt (ex : 1.0 = volume injecté = volume poreux)."),
            ],
            "Affichage": [
                ("Palette", "Palette de couleurs du GIF."),
                ("Tmin_plot / Tmax_plot", "Bornes de couleur pour l’affichage (ne change pas la physique)."),
                ("Halfwidth plan (m)", "Demi-largeur de l’image 2D (vue en plan) du GIF."),
                ("nxy", "Résolution (pixels) de l’image du GIF. Plus grand = plus lourd."),
            ],
            "Options GIF": [
                ("fps", "Images par seconde du GIF."),
                ("Contour step (°C)", "Pas entre lignes de contour (plus petit = plus de lignes)."),
                ("Contour lw", "Épaisseur des lignes de contour."),
                ("Seuil plume (°C > T0)", "Seuil (au-dessus de T0) utilisé pour afficher la plume en contours dans le GIF."),
                ("Qualité GIF (DPI)", "Qualité/poids du GIF."),
                ("Max frames GIF", "Limite le nombre d’images du GIF (pour éviter un fichier trop lourd)."),
            ],
        },
    },

    "en": {
        "lang_name": "English",
        "title": "Thermo-hydraulic simulator - Radial Darcy",
        "tab_sim": "Simulation",
        "tab_interp": "Interpretation",
        "tab_params": "Parameter guide",
        "sidebar": "Parameters",
        "mode_label": "Flow mode",
        "mode_map": {
            "matrix": "Intact medium (Darcy limited by Δp)",
            "eff_auto": "Stimulated reservoir (Q imposed → k required)",
            "eff_manual": "Fractured reservoir (imposed effective k)",
        },
        "run_btn": "Run",
        "reset_btn": "Clear cache",
        "generate_gif": "Generate GIF",
        "download_gif": "Download GIF",
        "summary": "Summary",
        "summary_table_title": "Simulation summary (table)",
        "diag": "Diagnostics",
        "plot_Tmean": "Mean temperature (°C)",
        "plot_PV": "Injected PV (Qt / Vp)",
        "plot_plume": "Plume radius (m)",
        "plot_profiles": "Radial ΔT(r) profiles (no interpolation)",
        "hydraulics": "Hydraulics",
        "geom": "Geometry",
        "thermal": "Thermal",
        "losses": "Vertical losses",
        "time_num": "Time & numerics",
        "stop_pv": "Stop PV",
        "display": "Display",
        "gif_opts": "GIF options",
        "ui_hint": "Set parameters on the left then click **Run**.",
        "interp_title": "How to read the 4 plots",
        "interp_bullets": [
            "**Mean temperature**: shows whether the reservoir heats globally (stored energy) or only locally. A plateau suggests losses/hydraulic limits dominate.",
            "**Injected PV (Qt/Vp)**: indicates how fast you “fill” the pore volume. PV=1 means injected volume equals total pore volume (aggressive if early).",
            "**Plume radius**: distance reached by warming thresholds (ΔT≥0.5°C and ΔT≥1°C). Saturation indicates physical limits.",
            "**ΔT(r) profiles**: direct thermal gradients at selected times. Steep front = advection-dominated; smeared front = diffusion/loss-dominated.",
        ],
        "params_title": "What do the parameters mean?",
        "params_intro": "This tab explains the sidebar settings (no math).",
        "params_sections": {
            "Hydraulics": [
                ("Flow mode", "Choose whether flow is Δp-limited (intact), whether effective k is imposed (fractured), or whether k is computed to reach a target flow rate (stimulated)."),
                ("Δp max (MPa)", "Maximum pressure drop from the well to the outer boundary. Higher Δp generally increases flow."),
                ("μ (Pa·s)", "Fluid viscosity. Higher μ reduces flow."),
                ("k matrix (m²)", "Matrix permeability (intact medium). Higher k means easier flow."),
                ("Q target (L/s)", "Desired flow rate (stimulated mode). The model computes the minimum effective k to reach it (within Δp)."),
                ("k effective (m²)", "Imposed effective permeability (fractured mode). Represents fractures/stimulation."),
            ],
            "Geometry": [
                ("R (m)", "Outer reservoir radius (boundary held at T0)."),
                ("rw (m)", "Well radius (injection point)."),
                ("b (m)", "Reservoir thickness. Affects flow and pore volume."),
                ("φ (-)", "Porosity: fraction of volume occupied by fluid (affects pore volume and velocities)."),
            ],
            "Thermal": [
                ("T0 (°C)", "Initial reservoir temperature (also outer boundary temperature)."),
                ("Tin (°C)", "Injected fluid temperature at the well (during injection)."),
                ("Enable losses", "Adds simplified vertical heat losses: reservoir relaxes back toward T0 over time."),
                ("tau (years)", "Loss time scale: smaller tau = stronger losses; larger tau = weaker losses."),
            ],
            "Time & numerics": [
                ("Total duration (years)", "Simulation duration."),
                ("Injection duration (years)", "Time during which Tin is imposed at the well. After that, injection stops."),
                ("Nr", "Number of radial cells (higher = finer near well, slower)."),
                ("dt (days)", "Time step. Smaller = more stable/accurate but slower."),
                ("Snapshot (days)", "How often profiles are saved (for plots and GIF)."),
            ],
            "Stop PV": [
                ("Stop when PV>=threshold", "Stops the run when injected volume reaches a multiple of pore volume (PV)."),
                ("PV threshold", "Stopping PV value (e.g., 1.0 = injected volume equals pore volume)."),
            ],
            "Display": [
                ("Palette", "GIF colormap."),
                ("Tmin_plot / Tmax_plot", "Color scale bounds (does not change physics)."),
                ("Halfwidth plan (m)", "Half-width of the 2D plan view for the GIF."),
                ("nxy", "GIF image resolution. Higher = heavier."),
            ],
            "GIF options": [
                ("fps", "Frames per second."),
                ("Contour step (°C)", "Contour interval (smaller = more contour lines)."),
                ("Contour lw", "Contour line width."),
                ("Plume threshold (°C > T0)", "Threshold above T0 used to draw plume contours in the GIF."),
                ("GIF quality (DPI)", "GIF quality/size."),
                ("Max frames GIF", "Caps frames to avoid overly large GIFs."),
            ],
        },
    },

    "it": {
        "lang_name": "Italiano",
        "title": "Simulatore termo-idraulico - Darcy Radiale",
        "tab_sim": "Simulazione",
        "tab_interp": "Interpretazione",
        "tab_params": "Guida parametri",
        "sidebar": "Parametri",
        "mode_label": "Modalità di flusso",
        "mode_map": {
            "matrix": "Mezzo intatto (Darcy limitato da Δp)",
            "eff_auto": "Serbatoio stimolato (Q imposto → k richiesto)",
            "eff_manual": "Serbatoio fratturato (k efficace imposto)",
        },
        "run_btn": "Avvia",
        "reset_btn": "Svuota cache",
        "generate_gif": "Genera GIF",
        "download_gif": "Scarica GIF",
        "summary": "Riepilogo",
        "summary_table_title": "Riepilogo simulazione (tabella)",
        "diag": "Diagnostica",
        "plot_Tmean": "Temperatura media (°C)",
        "plot_PV": "PV iniettato (Qt / Vp)",
        "plot_plume": "Raggio plume (m)",
        "plot_profiles": "Profili radiali ΔT(r) (senza interpolazione)",
        "hydraulics": "Idraulica",
        "geom": "Geometria",
        "thermal": "Termico",
        "losses": "Perdite verticali",
        "time_num": "Tempo & numerica",
        "stop_pv": "Stop PV",
        "display": "Display",
        "gif_opts": "Opzioni GIF",
        "ui_hint": "Imposta i parametri a sinistra poi clicca **Avvia**.",
        "interp_title": "Come leggere i 4 grafici",
        "interp_bullets": [
            "**Temperatura media**: indica se il serbatoio si riscalda globalmente o solo localmente. Un plateau suggerisce perdite/limiti idraulici.",
            "**PV iniettato (Qt/Vp)**: mostra quanto rapidamente “riempi” il volume dei pori. PV=1 = volume iniettato pari al volume dei pori (aggressivo se rapido).",
            "**Raggio plume**: distanza raggiunta da soglie (ΔT≥0.5°C e ΔT≥1°C). Saturazione = limite fisico.",
            "**Profili ΔT(r)**: gradienti termici diretti a tempi selezionati. Fronte ripido = advezione; fronte diffuso = diffusione/perdite.",
        ],
        "params_title": "A cosa servono i parametri?",
        "params_intro": "Questa scheda spiega le impostazioni della barra laterale (senza formule).",
        "params_sections": {
            "Idraulica": [
                ("Modalità di flusso", "Scegli se il flusso è limitato da Δp (intatto), se k efficace è imposto (fratturato) o se k viene calcolato per raggiungere un Q target (stimolato)."),
                ("Δp max (MPa)", "Caduta di pressione massima. Aumentandola tende ad aumentare il flusso."),
                ("μ (Pa·s)", "Viscosità del fluido. Più è alta, più il flusso è frenato."),
                ("k matrice (m²)", "Permeabilità della matrice. Più è alta, più il fluido scorre facilmente."),
                ("Q target (L/s)", "Portata desiderata (modo stimolato). Il modello calcola k efficace minima per raggiungerla."),
                ("k efficace (m²)", "Permeabilità efficace imposta (modo fratturato). Rappresenta fratture/stimolazione."),
            ],
            "Geometria": [
                ("R (m)", "Raggio esterno del serbatoio (bordo a T0)."),
                ("rw (m)", "Raggio del pozzo (punto di iniezione)."),
                ("b (m)", "Spessore del serbatoio. Influenza flusso e volume poroso."),
                ("φ (-)", "Porosità: frazione di volume occupata dal fluido."),
            ],
            "Termico": [
                ("T0 (°C)", "Temperatura iniziale (e temperatura al bordo esterno)."),
                ("Tin (°C)", "Temperatura del fluido iniettato al pozzo (durante iniezione)."),
                ("Attivare perdite", "Aggiunge perdite verticali semplificate: il serbatoio tende a tornare verso T0."),
                ("tau (anni)", "Scala di tempo delle perdite: tau piccolo = perdite forti; tau grande = perdite deboli."),
            ],
            "Tempo & numerica": [
                ("Durata totale (anni)", "Durata della simulazione."),
                ("Durata iniezione (anni)", "Tempo in cui Tin è imposto al pozzo. Poi l’iniezione si ferma."),
                ("Nr", "Numero di celle radiali (più alto = più fine, più lento)."),
                ("dt (giorni)", "Passo di tempo. Più piccolo = più accurato ma più lento."),
                ("Snapshot (giorni)", "Frequenza di salvataggio dei profili (grafici e GIF)."),
            ],
            "Stop PV": [
                ("Stop quando PV>=soglia", "Ferma quando il volume iniettato raggiunge un multiplo del volume poroso (PV)."),
                ("Soglia PV", "Valore PV di stop (es. 1.0 = volume iniettato = volume poroso)."),
            ],
            "Display": [
                ("Palette", "Mappa colori del GIF."),
                ("Tmin_plot / Tmax_plot", "Limiti della scala colori (non cambia la fisica)."),
                ("Halfwidth plan (m)", "Mezza larghezza della vista in pianta per il GIF."),
                ("nxy", "Risoluzione dell’immagine GIF."),
            ],
            "Opzioni GIF": [
                ("fps", "Frame al secondo."),
                ("Contour step (°C)", "Intervallo tra isolinee."),
                ("Contour lw", "Spessore isolinee."),
                ("Soglia plume (°C > T0)", "Soglia sopra T0 usata per disegnare la plume nel GIF."),
                ("Qualità GIF (DPI)", "Qualità/peso del GIF."),
                ("Max frames GIF", "Limita i frame per evitare un GIF troppo pesante."),
            ],
        },
    },
}


# ============================================================
# Model helpers
# ============================================================
def build_log_grid(rw, R, Nr):
    r_faces = np.geomspace(rw, R, Nr + 1)
    r_cent = np.sqrt(r_faces[:-1] * r_faces[1:])
    dr_cell = r_faces[1:] - r_faces[:-1]
    return r_faces, r_cent, dr_cell


def effective_alpha(phi, k_rock, rho_rock, cp_rock, k_w, rho_w, cp_w):
    rhoCp = (1 - phi) * rho_rock * cp_rock + phi * rho_w * cp_w
    k_eff = (1 - phi) * k_rock + phi * k_w
    return k_eff / rhoCp


def Q_from_dp_radial(k_m2, R, rw, b, mu, dp_Pa):
    L = np.log(R / rw)
    return (2.0 * np.pi * k_m2 * b / mu) * (dp_Pa / L)


def k_required_for_Q_dp(Q_m3s, R, rw, b, mu, dp_Pa):
    L = np.log(R / rw)
    return (Q_m3s * mu * L) / (2.0 * np.pi * b * dp_Pa)


def pore_velocity(phi, r_cent, Q_m3s, b):
    return (Q_m3s / (2.0 * np.pi * r_cent * b)) / phi


def assemble_matrix(r_faces, r_cent, dr_cell, v, alpha, dt, lambda_loss):
    N = len(r_cent)

    dW = np.empty(N)
    dE = np.empty(N)
    dW[0] = r_cent[0] - r_faces[0]
    dE[0] = r_cent[1] - r_cent[0]
    for i in range(1, N - 1):
        dW[i] = r_cent[i] - r_cent[i - 1]
        dE[i] = r_cent[i + 1] - r_cent[i]
    dW[N - 1] = r_cent[N - 1] - r_cent[N - 2]
    dE[N - 1] = r_faces[N] - r_cent[N - 1]

    # diffusion
    aW = np.zeros(N)
    aP = np.zeros(N)
    aE = np.zeros(N)
    for i in range(1, N - 1):
        rwf = r_faces[i]
        ref = r_faces[i + 1]
        vol = r_cent[i] * dr_cell[i]
        aE[i] = alpha * ref / (vol * dE[i])
        aW[i] = alpha * rwf / (vol * dW[i])
        aP[i] = -(aE[i] + aW[i])

    # advection upwind outward
    advW = np.zeros(N)
    advP = np.zeros(N)
    for i in range(1, N):
        advW[i] = -v[i] / dW[i]
        advP[i] = +v[i] / dW[i]

    diag = np.ones(N) + dt * advP - dt * aP + dt * lambda_loss
    lower = dt * advW - dt * aW
    upper = -dt * aE

    return diags([lower[1:], diag, upper[:-1]], [-1, 0, 1], format="csc")


def apply_outer_dirichlet(A, b, T0):
    A = A.tolil()
    n = A.shape[0]
    A[n - 1, :] = 0.0
    A[n - 1, n - 1] = 1.0
    b[n - 1] = T0
    return A.tocsc(), b


def apply_inner_dirichlet(A, b, Tin):
    A = A.tolil()
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = Tin
    return A.tocsc(), b


def volume_average_T(r_cent, dr_cell, Tprof):
    w = r_cent * dr_cell
    return float(np.sum(w * Tprof) / np.sum(w))


def plume_radius(r_cent, Tprof, T0, dT):
    thr = T0 + dT
    idx = np.where(Tprof >= thr)[0]
    return float(r_cent[idx[-1]]) if len(idx) else 0.0


# ============================================================
# Simulation
# ============================================================
@st.cache_data(show_spinner=False)
def run_sim(params):
    R = params["R"]; rw = params["rw"]; b = params["b"]; phi = params["phi"]
    T0 = params["T0"]; Tin = params["Tin"]
    years_total = params["years_total"]; injection_years = params["injection_years"]
    Nr = params["Nr"]; dt_days = params["dt_days"]; save_every_days = params["save_every_days"]

    mu = params["mu_Pa_s"]; dp_max_MPa = params["dp_max_MPa"]
    mode = params["mode"]
    k_matrix = params["k_matrix_m2"]
    k_eff_manual = params["k_eff_manual_m2"]
    Q_target_lps = params["Q_target_lps"]

    enable_losses = params["enable_losses"]; tau_years = params["tau_years"]

    k_rock = params["k_rock"]; rho_rock = params["rho_rock"]; cp_rock = params["cp_rock"]
    k_w = params["k_w"]; rho_w = params["rho_w"]; cp_w = params["cp_w"]

    stop_at_PV = params["stop_at_PV"]
    PV_stop_value = params["PV_stop_value"]

    r_faces, r_cent, dr_cell = build_log_grid(rw, R, Nr)
    alpha = effective_alpha(phi, k_rock, rho_rock, cp_rock, k_w, rho_w, cp_w)

    dt = dt_days * SECONDS_PER_DAY
    t_end = years_total * SECONDS_PER_YEAR
    inj_end = injection_years * SECONDS_PER_YEAR
    save_every = save_every_days * SECONDS_PER_DAY

    lambda_loss = 0.0
    if enable_losses and tau_years > 0:
        lambda_loss = 1.0 / (tau_years * SECONDS_PER_YEAR)

    Vp = np.pi * (R**2 - rw**2) * b * phi
    dp_eff = dp_max_MPa * 1e6

    if mode == "matrix":
        k_used = k_matrix
    elif mode == "eff_manual":
        k_used = k_eff_manual
    elif mode == "eff_auto":
        Q_target = Q_target_lps / 1000.0
        k_req = k_required_for_Q_dp(Q_target, R, rw, b, mu, dp_eff)
        k_used = max(k_req, k_matrix)
    else:
        raise ValueError("Invalid mode")

    Q_eff = Q_from_dp_radial(k_used, R, rw, b, mu, dp_eff)
    v_on = pore_velocity(phi, r_cent, Q_eff, b)
    A_on = assemble_matrix(r_faces, r_cent, dr_cell, v_on, alpha, dt, lambda_loss)

    Tprof = np.full(Nr, T0, dtype=float)

    times, snaps = [], []
    Tmean, PV, r05, r10 = [], [], [], []

    t = 0.0
    next_save = 0.0
    nsteps = int(np.ceil(t_end / dt))

    for _ in range(nsteps):
        inj = (t <= inj_end)

        bvec = Tprof.copy() + (dt * lambda_loss * T0)

        A2, b2 = apply_outer_dirichlet(A_on, bvec, T0)
        if inj:
            A2, b2 = apply_inner_dirichlet(A2, b2, Tin)
        else:
            A2 = A2.tolil()
            A2[0, :] = 0.0
            A2[0, 0] = 1.0
            A2[0, 1] = -1.0
            b2[0] = 0.0
            A2 = A2.tocsc()

        Tprof = spsolve(A2, b2)
        t += dt

        if t >= next_save - 1e-9:
            ty = t / SECONDS_PER_YEAR
            times.append(ty)
            snaps.append(Tprof.copy())

            Tmean.append(volume_average_T(r_cent, dr_cell, Tprof))

            PVt = (Q_eff * min(t, inj_end)) / Vp if Vp > 0 else 0.0
            PV.append(PVt)

            r05.append(plume_radius(r_cent, Tprof, T0, 0.5))
            r10.append(plume_radius(r_cent, Tprof, T0, 1.0))

            if stop_at_PV and PVt >= PV_stop_value - 1e-12:
                break

            next_save += save_every

    info = {
        "mode": mode,
        "k_used_m2": float(k_used),
        "Q_eff_lps": float(Q_eff * 1000.0),
        "dp_eff_MPa": float(dp_eff / 1e6),
        "Vp_m3": float(Vp),
        "tPV_years": float((Vp / Q_eff) / SECONDS_PER_YEAR) if Q_eff > 0 else np.inf,
        "t_end_years": float(times[-1]) if times else 0.0,
        "PV_end": float(PV[-1]) if PV else 0.0,
        "lambda_loss_1s": float(lambda_loss),
    }

    return r_cent, np.array(times), np.array(snaps), info, np.array(Tmean), np.array(PV), np.array(r05), np.array(r10)


# ============================================================
# GIF maker (uses temporary file, Windows-safe)
# ============================================================
def make_gif_bytes(r_cent, times, snaps, params, title_mode):
    max_frames = params["max_frames"]
    if len(times) > max_frames:
        idx = np.linspace(0, len(times) - 1, max_frames).astype(int)
        times = times[idx]
        snaps = snaps[idx]

    R = params["R"]
    T0 = params["T0"]
    halfwidth = params["halfwidth_plan_m"]
    nxy = params["nxy"]

    Tmin_plot = params["Tmin_plot"]
    Tmax_plot = params["Tmax_plot"]
    cmap = params["cmap"]
    fps = params["fps"]
    dpi = params["gif_dpi"]

    contour_step = params["contour_step"]
    contour_lw = params["contour_lw"]
    plume_thr = params["plume_threshold"]

    x = np.linspace(-halfwidth, halfwidth, nxy)
    X, Y = np.meshgrid(x, x)
    Rxy = np.sqrt(X**2 + Y**2)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    im = ax.imshow(
        np.full_like(Rxy, T0),
        extent=[x.min(), x.max(), x.min(), x.max()],
        origin="lower",
        vmin=Tmin_plot,
        vmax=Tmax_plot,
        cmap=cmap,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Temp (°C)")
    ax.plot(0, 0, "ko", ms=4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    all_levels = np.arange(Tmin_plot, Tmax_plot + contour_step, contour_step)
    cs = None

    def update(k):
        nonlocal cs
        Tprof = snaps[k]
        field = np.interp(Rxy, r_cent, Tprof, left=Tprof[0], right=T0)
        im.set_data(field)

        if cs is not None:
            for c in cs.collections:
                c.remove()
            cs = None

        if k != 0:
            thr = T0 + plume_thr
            idxs = np.where(Tprof >= thr)[0]
            rpl = float(r_cent[idxs[-1]]) if len(idxs) else 0.0
            progress = max(0.0, min(1.0, rpl / R))
            nlev = int(np.floor(progress * len(all_levels)))
            if nlev > 0:
                levels_to_plot = all_levels[-nlev:]
                cs = ax.contour(
                    X, Y, field,
                    levels=np.sort(levels_to_plot),
                    colors="k",
                    linewidths=contour_lw,
                    linestyles="dashed",
                )

        ax.set_title(f"{title_mode} | t={times[k]:.2f} y")
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(times),
        interval=int(1000 / max(1, fps)), blit=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = Path(tmpdir) / "anim.gif"
        ani.save(str(gif_path), writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return gif_path.read_bytes()


# ============================================================
# Non-interpolated profiles ΔT(r) at selected times
# ============================================================
def plot_radial_profiles(r_cent, times, snaps, T0, lang_key):
    tmax = float(times[-1]) if len(times) else 0.0
    candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
    targets = [t for t in candidates if t <= tmax + 1e-12]
    if len(targets) < 3 and tmax > 0:
        targets = list(np.linspace(0.0, tmax, 4)[1:])

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)

    for ttar in targets:
        k = int(np.argmin(np.abs(times - ttar)))
        dT = snaps[k] - T0
        ax.plot(
            r_cent, dT, marker="o", markersize=2.2, linewidth=1.0,
            label=f"t≈{times[k]:.2f} y"
        )

    ax.grid(True)
    ax.set_xlabel("Radius r (m)")
    ax.set_ylabel("ΔT (°C)")
    ax.set_title("ΔT(r) profiles")
    ax.legend()
    return fig


# ============================================================
# Formatting helpers (table-friendly)
# ============================================================
def fmt_2dec_or_sci(x):
    """2 decimals for normal numbers; scientific for very small/very large."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return str(x)
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-2 or ax >= 1e6):
        return f"{x:.2e}"
    return f"{x:.2f}"


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Simulateur Darcy Radial", layout="wide")

lang_key = st.sidebar.selectbox(
    "Language / Langue / Lingua",
    options=["fr", "en", "it"],
    format_func=lambda k: I18N[k]["lang_name"],
    index=0,
)
T = I18N[lang_key]

st.title(T["title"])
tab_sim, tab_interp, tab_params = st.tabs([T["tab_sim"], T["tab_interp"], T["tab_params"]])

palettes = ["turbo", "inferno", "plasma", "viridis", "magma",
            "cividis", "hot", "coolwarm", "Spectral", "rainbow"]

with st.sidebar:
    st.header(T["sidebar"])

    mode = st.selectbox(
        T["mode_label"],
        options=["matrix", "eff_auto", "eff_manual"],
        format_func=lambda k: T["mode_map"][k],
        index=0,
    )

    st.header(T["hydraulics"])
    dp_max_MPa = st.slider("Δp max (MPa)", 0.1, 50.0, 5.0, 0.1)
    mu = st.number_input("μ (Pa·s)", value=1.0e-3, format="%.2e")
    k_matrix = st.number_input("k matrice (m²)", value=1.0e-16, format="%.2e")
    Q_target_lps = st.slider("Q cible (L/s) (mode stimulé)", 0.1, 50.0, 10.0, 0.1)
    k_eff_manual = st.number_input("k effectif (m²) (mode fracturé)", value=3.0e-13, format="%.2e")

    st.header(T["geom"])
    R = st.slider("R (m)", 10.0, 2000.0, 250.0, 10.0)
    rw = st.number_input("rw (m)", value=0.075, format="%.3f")
    b = st.slider("b (m)", 1.0, 200.0, 10.0, 1.0)
    phi = st.slider("φ (-)", 0.01, 0.40, 0.15, 0.01)

    st.header(T["thermal"])
    T0 = st.slider("T0 (°C)", -5.0, 40.0, 15.0, 0.5)
    Tin = st.slider("Tin (°C)", 0.0, 120.0, 40.0, 0.5)

    st.header(T["losses"])
    enable_losses = st.checkbox("Activer pertes", value=True)
    tau_years = st.slider("tau (ans)", 0.1, 50.0, 5.0, 0.1)

    st.header(T["time_num"])
    years_total = st.slider("Durée totale (ans)", 0.1, 100.0, 10.0, 0.5)
    injection_years = st.slider("Durée injection (ans)", 0.0, 100.0, 10.0, 0.5)
    Nr = st.slider("Nr (maillage radial)", 80, 800, 260, 10)
    dt_days = st.slider("dt (jours)", 0.5, 30.0, 5.0, 0.5)
    save_every_days = st.slider("Snapshot (jours)", 5.0, 365.0, 60.0, 5.0)

    st.header(T["stop_pv"])
    stop_at_PV = st.checkbox("Stop quand PV>=seuil", value=True)
    PV_stop_value = st.slider("Seuil PV", 0.1, 3.0, 1.0, 0.05)

    st.header(T["display"])
    cmap = st.selectbox("Palette", palettes, index=0)
    Tmin_plot = st.number_input("Tmin_plot", value=float(T0))
    Tmax_plot = st.number_input("Tmax_plot", value=float(Tin))
    halfwidth_plan_m = st.slider("Halfwidth plan (m)", 50.0, 2000.0, 260.0, 10.0)
    nxy = st.slider("nxy (image)", 120, 520, 280, 20)

    st.header(T["gif_opts"])
    generate_gif = st.checkbox(T["generate_gif"], value=True)
    fps = st.slider("fps", 1, 12, 5, 1)
    contour_step = st.slider("Contour step (°C)", 0.5, 5.0, 2.0, 0.5)
    contour_lw = st.slider("Contour lw", 0.1, 1.5, 0.35, 0.05)
    plume_threshold = st.slider("Seuil plume (°C > T0)", 0.1, 5.0, 0.5, 0.1)
    gif_dpi = st.slider("Qualité GIF (DPI)", 80, 260, 160, 10)
    max_frames = st.slider("Max frames GIF", 20, 140, 60, 5)

    run_btn = st.button(T["run_btn"], type="primary")
    if st.button(T["reset_btn"]):
        st.cache_data.clear()
        st.success("Cache vidé.")


params = dict(
    R=float(R), rw=float(rw), b=float(b), phi=float(phi),
    T0=float(T0), Tin=float(Tin),
    years_total=float(years_total), injection_years=float(injection_years),
    Nr=int(Nr), dt_days=float(dt_days), save_every_days=float(save_every_days),

    mu_Pa_s=float(mu), dp_max_MPa=float(dp_max_MPa),
    mode=str(mode), k_matrix_m2=float(k_matrix),
    Q_target_lps=float(Q_target_lps), k_eff_manual_m2=float(k_eff_manual),

    enable_losses=bool(enable_losses), tau_years=float(tau_years),

    k_rock=1.5, rho_rock=2300.0, cp_rock=900.0,
    k_w=0.6, rho_w=1000.0, cp_w=4180.0,

    stop_at_PV=bool(stop_at_PV), PV_stop_value=float(PV_stop_value),

    cmap=str(cmap), Tmin_plot=float(Tmin_plot), Tmax_plot=float(Tmax_plot),
    halfwidth_plan_m=float(halfwidth_plan_m), nxy=int(nxy),
    fps=int(fps), contour_step=float(contour_step), contour_lw=float(contour_lw),
    plume_threshold=float(plume_threshold),
    gif_dpi=int(gif_dpi), max_frames=int(max_frames),
)

with tab_sim:
    st.info(T["ui_hint"])

    if run_btn:
        try:
            with st.spinner("Simulation..."):
                r_cent, times, snaps, info, Tmean, PV, r05, r10 = run_sim(params)

            readable_mode = T["mode_map"].get(info["mode"], info["mode"])

            # GIF first
            if generate_gif:
                st.subheader("GIF")
                with st.spinner("GIF..."):
                    gif_bytes = make_gif_bytes(r_cent, times, snaps, params, readable_mode)
                st.image(gif_bytes, use_container_width=True)
                st.download_button(
                    T["download_gif"],
                    data=gif_bytes,
                    file_name="animation.gif",
                    mime="image/gif"
                )

            # Summary as a TABLE (rounded)
            st.subheader(T["summary_table_title"])
            summary_df = pd.DataFrame(
                [
                    ("Mode", readable_mode),
                    ("k utilisé (m²)", fmt_2dec_or_sci(info["k_used_m2"])),
                    ("Q effectif (L/s)", fmt_2dec_or_sci(info["Q_eff_lps"])),
                    ("Δp effectif (MPa)", fmt_2dec_or_sci(info["dp_eff_MPa"])),
                    ("Volume poreux Vp (m³)", fmt_2dec_or_sci(info["Vp_m3"])),
                    ("Temps pour PV=1 (ans)", fmt_2dec_or_sci(info["tPV_years"])),
                    ("Temps final simulé (ans)", fmt_2dec_or_sci(info["t_end_years"])),
                    ("PV final", fmt_2dec_or_sci(info["PV_end"])),
                    ("Pertes λ (1/s)", fmt_2dec_or_sci(info["lambda_loss_1s"])),
                ],
                columns=["Paramètre", "Valeur"]
            )
            st.table(summary_df)

            st.subheader(T["diag"])
            col1, col2 = st.columns(2)

            with col1:
                fig = plt.figure()
                plt.plot(times, Tmean)
                plt.grid(True)
                plt.xlabel("Time (years)")
                plt.ylabel(T["plot_Tmean"])
                st.pyplot(fig, clear_figure=True)

                fig = plt.figure()
                plt.plot(times, PV)
                plt.axhline(1.0, ls="--")
                plt.grid(True)
                plt.xlabel("Time (years)")
                plt.ylabel(T["plot_PV"])
                st.pyplot(fig, clear_figure=True)

            with col2:
                fig = plt.figure()
                plt.plot(times, r05, label="ΔT ≥ 0.5°C")
                plt.plot(times, r10, label="ΔT ≥ 1.0°C")
                plt.grid(True)
                plt.legend()
                plt.xlabel("Time (years)")
                plt.ylabel(T["plot_plume"])
                st.pyplot(fig, clear_figure=True)

            st.subheader(T["plot_profiles"])
            fig = plot_radial_profiles(r_cent, times, snaps, params["T0"], lang_key)
            st.pyplot(fig, clear_figure=True)

            st.success("OK ✅")

        except Exception as e:
            st.error("Erreur. Détails :")
            st.exception(e)

with tab_interp:
    st.subheader(T["interp_title"])
    for btxt in T["interp_bullets"]:
        st.markdown(f"- {btxt}")

with tab_params:
    st.subheader(T["params_title"])
    st.write(T["params_intro"])

    for section_title, items in T["params_sections"].items():
        st.markdown(f"### {section_title}")
        for name, desc in items:
            st.markdown(f"**{name}** : {desc}")

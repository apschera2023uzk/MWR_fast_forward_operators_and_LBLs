#!/usr/bin/env python3
"""
plot_iwv_scatter.py
-------------------
Recreates ARMS-gb / RTTOV-gb departure-vs-IWV scatter plots from the
pre-processed NetCDF that already contains Deviations_* and cloud_flag.

New feature: in addition to the elevation-averaged and channel-averaged plots,
one plot per (elevation × channel) pair is written to  .../iwv_per_ele_chan/
"""

##############################################################################
# 1  Imports
##############################################################################

import argparse
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

##############################################################################
# 2  Parameters
##############################################################################

fs = 20
plt.rc("font", size=fs)
plt.style.use("seaborn-poster")

elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 4.2])
n_chans    = 14

##############################################################################
# 3  Argument parsing
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--NetCDF", "-nc", type=str,
        default=os.path.expanduser(
            "~/PhD_data/TB_preproc_and_proc_results/"
            "3campaigns_3models_all_results_and_stats.nc"))
    parser.add_argument("--output", "-o", type=str,
        default=os.path.expanduser("~/PhD_plots/2026/"))
    return parser.parse_args()

##############################################################################
# 4  Helpers
##############################################################################

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

##############################################################################

def select_ds_camp_loc(ds, campaign, location):
    mask   = (ds["Campaign"] == campaign) & (ds["Location"] == location)
    ds_sel = ds.sel(time=ds["time"].values[mask.values])
    return ds_sel

##############################################################################

def apply_sky_mask(ds_sel, sky):
    if sky == "all_sky":
        return ds_sel
    cf       = ds_sel["cloud_flag"]
    bad_mask = (cf == 1) if sky == "clear" else (cf == 0)
    ds_cf    = ds_sel.copy()
    for var in ds_cf.data_vars:
        if "elevation" not in ds_cf[var].dims:
            continue
        ds_cf[var] = ds_cf[var].where(~bad_mask.broadcast_like(ds_cf[var]))
    return ds_cf

##############################################################################

def get_iwv(ds_sel, campaign, location):
    """Return IWV DataArray for the given campaign/location."""
    mapping = {
        ("FESSTVaL", "RAO_Lindenberg"): "Dwdhat_IWV",
        ("Vital I",  "JOYCE"):          "Joyhat_IWV",
        ("FESSTVaL", "Falkenberg"):     "Sunhat_IWV",
        ("Socles",   "JOYCE"):          "Tophat_IWV",
    }
    key = (campaign, location)
    if key not in mapping:
        return None
    var = mapping[key]
    return ds_sel[var] if var in ds_sel else None

##############################################################################

def get_departures(ds_cf, var_name):
    """Return (time, N_Channels, elevation) array, azimuth-averaged if needed."""
    da = ds_cf[var_name]
    if "azimuth" in da.dims:
        da = da.mean(dim="azimuth", keep_attrs=True)
    # ensure shape (time, N_Channels, elevation)
    if da.dims[1] != "N_Channels":
        da = da.transpose("time", "N_Channels", "elevation")
    return da.values   # numpy array

##############################################################################

def safe_ylim(ax, arr, limit=13):
    if np.nanmax(np.abs(arr)) < limit + 1:
        ax.set_ylim(-limit, limit)

##############################################################################

def safe_xlim(ax, arr, limit=13):
    if np.nanmax(np.abs(arr)) < limit + 1:
        ax.set_xlim(-limit, limit)

##############################################################################
# 5  Plotting functions
##############################################################################

# ── 5a  All channels × all elevations ─────────────────────────────────────

def plot_model_vs_iwv_all(dep_rttov, dep_arms, iwv_flat,
                          campaign, location, sky, outpath):
    """2 plots: RTTOV vs IWV and ARMS vs IWV, all chans all elevs flattened."""
    for model, dep, color in [("RTTOV-gb", dep_rttov, "red"),
                               ("ARMS-gb",  dep_arms,  "orange")]:
        y_all = dep.flatten()
        x_all = np.repeat(iwv_flat, dep.shape[1] * dep.shape[2])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x_all, y_all, marker="X", color=color, s=8, alpha=0.4)
        ax.axhline(0, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("IWV [kg m⁻²]")
        ax.set_ylabel(f"{model} deviations from R24 [K]")
        ax.set_xlim(0, 45)
        safe_ylim(ax, y_all)
        ax.set_title(f"{model} vs. IWV — all chans/elevs\n"
                     f"[{campaign}, {location}, {sky}]")
        plt.tight_layout()
        fname = f"{sky}_allchans_allelev_{model}_IWV_{campaign}_{location}.png"
        plt.savefig(os.path.join(outpath, fname), dpi=300, bbox_inches="tight")
        plt.close()

##############################################################################
# ── 5b  Per channel (all elevations stacked) ─────────────────────────────

def plot_model_vs_iwv_per_channel(dep_rttov, dep_arms, iwv_flat,
                                   campaign, location, sky, outpath):
    """14 × 2 plots: one per channel for RTTOV and ARMS."""
    for ch_idx in range(n_chans):
        for model, dep, color in [("RTTOV-gb", dep_rttov, "red"),
                                   ("ARMS-gb",  dep_arms,  "orange")]:
            # dep shape: (time, n_chans, n_elevs) — stack over elevations
            y_all = dep[:, ch_idx, :].flatten()
            x_all = np.repeat(iwv_flat, dep.shape[2])
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(x_all, y_all, marker="X", color=color, s=8, alpha=0.5)
            ax.axhline(0, color="black", linestyle="--", linewidth=2)
            ax.set_xlabel("IWV [kg m⁻²]")
            ax.set_ylabel(f"{model} deviations from R24 [K]")
            ax.set_xlim(0, 45)
            axis_len = 4 if ch_idx > 6 else 13
            if np.nanmax(np.abs(y_all)) <= axis_len:
                ax.set_ylim(-axis_len, axis_len)
            ax.set_title(f"Ch{ch_idx+1}  {model} vs. IWV — all elevs\n"
                         f"[{campaign}, {location}, {sky}]")
            plt.tight_layout()
            fname = (f"{sky}_ch{ch_idx+1}_{model}_IWV_allelev_"
                     f"{campaign}_{location}.png")
            plt.savefig(os.path.join(outpath, fname), dpi=300,
                        bbox_inches="tight")
            plt.close()

##############################################################################
# ── 5c  Per elevation (all channels stacked) ─────────────────────────────

def plot_model_vs_iwv_per_elevation(dep_rttov, dep_arms, iwv_flat,
                                     campaign, location, sky, outpath,
                                     elevations=elevations):
    """n_elev × 2 plots: one per elevation for RTTOV and ARMS."""
    n_elev = dep_rttov.shape[2]
    for el_idx in range(n_elev):
        elev_val = elevations[el_idx] if el_idx < len(elevations) else el_idx
        for model, dep, color in [("RTTOV-gb", dep_rttov, "red"),
                                   ("ARMS-gb",  dep_arms,  "orange")]:
            y_all = dep[:, :, el_idx].flatten()
            x_all = np.repeat(iwv_flat, dep.shape[1])
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(x_all, y_all, marker="X", color=color, s=8, alpha=0.5)
            ax.axhline(0, color="black", linestyle="--", linewidth=2)
            ax.set_xlabel("IWV [kg m⁻²]")
            ax.set_ylabel(f"{model} deviations from R24 [K]")
            ax.set_xlim(0, 45)
            safe_ylim(ax, y_all)
            ax.set_title(f"{model} vs. IWV — {elev_val:.1f}° — all chans\n"
                         f"[{campaign}, {location}, {sky}]")
            plt.tight_layout()
            fname = (f"{sky}_elev{elev_val:.1f}_{model}_IWV_allchans_"
                     f"{campaign}_{location}.png")
            plt.savefig(os.path.join(outpath, fname), dpi=300,
                        bbox_inches="tight")
            plt.close()

##############################################################################
# ── 5d  NEW: Per (elevation × channel) ───────────────────────────────────

def plot_model_vs_iwv_per_ele_chan(dep_rttov, dep_arms, iwv_flat,
                                   campaign, location, sky, outpath,
                                   elevations=elevations):
    """n_elev × n_chans × 2 plots — one per (elevation, channel) pair."""
    n_elev = dep_rttov.shape[2]
    for el_idx in range(n_elev):
        elev_val = elevations[el_idx] if el_idx < len(elevations) else el_idx
        for ch_idx in range(n_chans):
            for model, dep, color in [("RTTOV-gb", dep_rttov, "red"),
                                       ("ARMS-gb",  dep_arms,  "orange")]:
                y = dep[:, ch_idx, el_idx]
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(iwv_flat, y, marker="X", color=color,
                           s=15, alpha=0.6)
                ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
                ax.set_xlabel("IWV [kg m⁻²]")
                ax.set_ylabel(f"{model} deviation from R24 [K]")
                ax.set_xlim(0, 45)
                axis_len = 4 if ch_idx > 6 else 13
                if np.nanmax(np.abs(y[~np.isnan(y)]), initial=0) <= axis_len:
                    ax.set_ylim(-axis_len, axis_len)
                ax.set_title(f"{model}  Ch{ch_idx+1}  {elev_val:.1f}°\n"
                             f"[{campaign}, {location}, {sky}]")
                plt.tight_layout()
                fname = (f"{sky}_elev{elev_val:.1f}_ch{ch_idx+1}_{model}_IWV_"
                         f"{campaign}_{location}.png")
                plt.savefig(os.path.join(outpath, fname), dpi=150,
                            bbox_inches="tight")
                plt.close()

##############################################################################
# ── 5e  ARMS vs RTTOV scatter coloured by IWV ────────────────────────────

def plot_arms_vs_rttov(dep_rttov, dep_arms, iwv_flat,
                       campaign, location, sky, outpath,
                       elevations=elevations):
    """All-elevation all-channel + per-elevation + per-channel ARMS vs RTTOV."""

    # All combined
    x_all = dep_arms.flatten()
    y_all = dep_rttov.flatten()
    c_all = np.repeat(iwv_flat, dep_arms.shape[1] * dep_arms.shape[2])
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(x_all, y_all, c=c_all, cmap="viridis", s=10, alpha=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("ARMS-gb deviations from R24 [K]")
    ax.set_ylabel("RTTOV-gb deviations from R24 [K]")
    safe_ylim(ax, y_all)
    safe_xlim(ax, x_all)
    ax.set_title(f"RTTOV vs. ARMS — all chans/elevs\n"
                 f"[{campaign}, {location}, {sky}]")
    plt.colorbar(sc, ax=ax, label="IWV [kg m⁻²]")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath,
        f"{sky}_allchans_allelev_ARMS_RTTOV_{campaign}_{location}.png"),
        dpi=300, bbox_inches="tight")
    plt.close()

    # Per elevation
    n_elev = dep_rttov.shape[2]
    for el_idx in range(n_elev):
        elev_val = elevations[el_idx] if el_idx < len(elevations) else el_idx
        x = dep_arms[:, :, el_idx].flatten()
        y = dep_rttov[:, :, el_idx].flatten()
        c = np.repeat(iwv_flat, dep_arms.shape[1])
        fig, ax = plt.subplots(figsize=(12, 10))
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=15, alpha=0.6)
        ax.axhline(0, color="black", linestyle="--", linewidth=2)
        ax.axvline(0, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("ARMS-gb deviations [K]")
        ax.set_ylabel("RTTOV-gb deviations [K]")
        safe_ylim(ax, y)
        safe_xlim(ax, x)
        ax.set_title(f"RTTOV vs. ARMS  {elev_val:.1f}° — all chans\n"
                     f"[{campaign}, {location}, {sky}]")
        plt.colorbar(sc, ax=ax, label="IWV [kg m⁻²]")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath,
            f"{sky}_elev{elev_val:.1f}_allchans_ARMS_RTTOV_"
            f"{campaign}_{location}.png"),
            dpi=300, bbox_inches="tight")
        plt.close()

    # Per channel
    for ch_idx in range(n_chans):
        x = dep_arms[:, ch_idx, :].flatten()
        y = dep_rttov[:, ch_idx, :].flatten()
        c = np.repeat(iwv_flat, dep_arms.shape[2])
        axis_len = 4 if ch_idx > 6 else 13
        fig, ax = plt.subplots(figsize=(12, 10))
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=15, alpha=0.6)
        ax.axhline(0, color="black", linestyle="--", linewidth=2)
        ax.axvline(0, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("ARMS-gb deviations [K]")
        ax.set_ylabel("RTTOV-gb deviations [K]")
        if np.nanmax(np.abs(y[~np.isnan(y)]), initial=0) < axis_len:
            ax.set_ylim(-axis_len, axis_len)
        if np.nanmax(np.abs(x[~np.isnan(x)]), initial=0) < axis_len:
            ax.set_xlim(-axis_len, axis_len)
        ax.set_title(f"RTTOV vs. ARMS  Ch{ch_idx+1} — all elevs\n"
                     f"[{campaign}, {location}, {sky}]")
        plt.colorbar(sc, ax=ax, label="IWV [kg m⁻²]")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath,
            f"{sky}_ch{ch_idx+1}_allelev_ARMS_RTTOV_"
            f"{campaign}_{location}.png"),
            dpi=300, bbox_inches="tight")
        plt.close()

##############################################################################
# ── 5f  NEW: ARMS vs RTTOV per (elevation × channel) ─────────────────────

def plot_arms_vs_rttov_per_ele_chan(dep_rttov, dep_arms, iwv_flat,
                                    campaign, location, sky, outpath,
                                    elevations=elevations):
    """n_elev × n_chans scatter of ARMS vs RTTOV coloured by IWV."""
    n_elev = dep_rttov.shape[2]
    for el_idx in range(n_elev):
        elev_val = elevations[el_idx] if el_idx < len(elevations) else el_idx
        for ch_idx in range(n_chans):
            x = dep_arms[:, ch_idx, el_idx]
            y = dep_rttov[:, ch_idx, el_idx]
            axis_len = 4 if ch_idx > 6 else 13
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(x, y, c=iwv_flat, cmap="viridis", s=20, alpha=0.7)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
            ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
            ax.set_xlabel("ARMS-gb deviation [K]")
            ax.set_ylabel("RTTOV-gb deviation [K]")
            valid_x = x[~np.isnan(x)]
            valid_y = y[~np.isnan(y)]
            if len(valid_y) and np.nanmax(np.abs(valid_y)) < axis_len:
                ax.set_ylim(-axis_len, axis_len)
            if len(valid_x) and np.nanmax(np.abs(valid_x)) < axis_len:
                ax.set_xlim(-axis_len, axis_len)
            ax.set_title(f"RTTOV vs. ARMS  Ch{ch_idx+1}  {elev_val:.1f}°\n"
                         f"[{campaign}, {location}, {sky}]")
            plt.colorbar(sc, ax=ax, label="IWV [kg m⁻²]")
            plt.tight_layout()
            fname = (f"{sky}_elev{elev_val:.1f}_ch{ch_idx+1}_ARMS_RTTOV_IWV_"
                     f"{campaign}_{location}.png")
            plt.savefig(os.path.join(outpath, fname), dpi=150,
                        bbox_inches="tight")
            plt.close()

##############################################################################
# 5g  MWR deviations vs IWV — all instruments combined per plot
##############################################################################

MWR_INSTRUMENTS = {
    "dwdhat": ("Deviations_dwdhat_R24", "blue"),
    "joyhat": ("Deviations_joyhat_R24", "green"),
    "foghat": ("Deviations_foghat_R24", "purple"),
    "hamhat": ("Deviations_hamhat_R24", "brown"),
}

def get_mwr_departures(ds_cf):
    """Return dict of {instrument: (time, N_Channels, elevation) array}."""
    result = {}
    for name, (var, _) in MWR_INSTRUMENTS.items():
        if var not in ds_cf:
            continue
        da = ds_cf[var]
        if "azimuth" in da.dims:
            da = da.mean(dim="azimuth", keep_attrs=True)
        if da.dims[1] != "N_Channels":
            da = da.transpose("time", "N_Channels", "elevation")
        result[name] = da.values
    return result


def plot_mwr_vs_iwv_all(mwr_deps, iwv_flat, campaign, location, sky, outpath):
    """One plot per instrument: all chans × all elevs vs IWV."""
    for name, dep in mwr_deps.items():
        color = MWR_INSTRUMENTS[name][1]
        y_all = dep.flatten()
        x_all = np.repeat(iwv_flat, dep.shape[1] * dep.shape[2])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x_all, y_all, marker="o", color=color, s=8, alpha=0.4)
        ax.axhline(0, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("IWV [kg m⁻²]")
        ax.set_ylabel(f"{name} deviations from R24 [K]")
        ax.set_xlim(0, 45)
        safe_ylim(ax, y_all)
        ax.set_title(f"{name} vs. IWV — all chans/elevs\n"
                     f"[{campaign}, {location}, {sky}]")
        plt.tight_layout()
        fname = f"{sky}_allchans_allelev_{name}_IWV_{campaign}_{location}.png"
        plt.savefig(os.path.join(outpath, fname), dpi=300, bbox_inches="tight")
        plt.close()


def plot_mwr_vs_iwv_per_channel(mwr_deps, iwv_flat,
                                 campaign, location, sky, outpath):
    """Per channel: all instruments on one plot, all elevations stacked."""
    for ch_idx in range(n_chans):
        fig, ax = plt.subplots(figsize=(10, 10))
        for name, dep in mwr_deps.items():
            color = MWR_INSTRUMENTS[name][1]
            y = dep[:, ch_idx, :].flatten()
            x = np.repeat(iwv_flat, dep.shape[2])
            ax.scatter(x, y, marker="o", color=color, s=8,
                       alpha=0.5, label=name)
        ax.axhline(0, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("IWV [kg m⁻²]")
        ax.set_ylabel("MWR deviation from R24 [K]")
        ax.set_xlim(0, 45)
        axis_len = 4 if ch_idx > 6 else 13
        ax.set_ylim(-axis_len, axis_len)
        ax.legend()
        ax.set_title(f"Ch{ch_idx+1} — all MWRs vs. IWV — all elevs\n"
                     f"[{campaign}, {location}, {sky}]")
        plt.tight_layout()
        fname = f"{sky}_ch{ch_idx+1}_MWRs_IWV_allelev_{campaign}_{location}.png"
        plt.savefig(os.path.join(outpath, fname), dpi=300, bbox_inches="tight")
        plt.close()


def plot_mwr_vs_iwv_per_elevation(mwr_deps, iwv_flat,
                                   campaign, location, sky, outpath,
                                   elevations=elevations):
    """Per elevation: all instruments on one plot, all channels stacked."""
    n_elev = next(iter(mwr_deps.values())).shape[2]
    for el_idx in range(n_elev):
        elev_val = elevations[el_idx] if el_idx < len(elevations) else el_idx
        fig, ax = plt.subplots(figsize=(10, 10))
        for name, dep in mwr_deps.items():
            color = MWR_INSTRUMENTS[name][1]
            y = dep[:, :, el_idx].flatten()
            x = np.repeat(iwv_flat, dep.shape[1])
            ax.scatter(x, y, marker="o", color=color, s=8,
                       alpha=0.5, label=name)
        ax.axhline(0, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("IWV [kg m⁻²]")
        ax.set_ylabel("MWR deviation from R24 [K]")
        ax.set_xlim(0, 45)
        safe_ylim(ax, np.concatenate([d[:, :, el_idx].flatten()
                                       for d in mwr_deps.values()]))
        ax.legend()
        ax.set_title(f"All MWRs vs. IWV — {elev_val:.1f}° — all chans\n"
                     f"[{campaign}, {location}, {sky}]")
        plt.tight_layout()
        fname = (f"{sky}_elev{elev_val:.1f}_MWRs_IWV_allchans_"
                 f"{campaign}_{location}.png")
        plt.savefig(os.path.join(outpath, fname), dpi=300, bbox_inches="tight")
        plt.close()


def plot_mwr_vs_iwv_per_ele_chan(mwr_deps, iwv_flat,
                                  campaign, location, sky, outpath,
                                  elevations=elevations):
    """Per (elevation × channel): all instruments on one plot."""
    n_elev = next(iter(mwr_deps.values())).shape[2]
    for el_idx in range(n_elev):
        elev_val = elevations[el_idx] if el_idx < len(elevations) else el_idx
        for ch_idx in range(n_chans):
            axis_len = 4 if ch_idx > 6 else 13
            fig, ax = plt.subplots(figsize=(8, 8))
            for name, dep in mwr_deps.items():
                color = MWR_INSTRUMENTS[name][1]
                y = dep[:, ch_idx, el_idx]
                ax.scatter(iwv_flat, y, marker="o", color=color,
                           s=15, alpha=0.6, label=name)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
            ax.set_xlabel("IWV [kg m⁻²]")
            ax.set_ylabel("MWR deviation from R24 [K]")
            ax.set_xlim(0, 45)
            ax.set_ylim(-axis_len, axis_len)
            ax.legend(fontsize=10)
            ax.set_title(f"MWRs vs. IWV  Ch{ch_idx+1}  {elev_val:.1f}°\n"
                         f"[{campaign}, {location}, {sky}]")
            plt.tight_layout()
            fname = (f"{sky}_elev{elev_val:.1f}_ch{ch_idx+1}_MWRs_IWV_"
                     f"{campaign}_{location}.png")
            plt.savefig(os.path.join(outpath, fname), dpi=150,
                        bbox_inches="tight")
            plt.close()

##############################################################################
# 6  Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()

    print("Loading dataset …")
    ds0 = xr.open_dataset(args.NetCDF)

    # Keep only what we need
    keep = [v for v in ds0.data_vars if
            v.startswith("Deviations_") or v.startswith("TBs_") or
            v in ("cloud_flag", "Campaign", "Location") or
            "IWV" in v or "LWP" in v]
    ds0 = ds0[keep]

    # Average out azimuth dimension where present
    ds0 = ds0.mean(dim="azimuth", keep_attrs=True) if "azimuth" in ds0.dims \
        else ds0

    skies     = ["clear", "cloudy", "all_sky"]
    campaigns = np.unique(ds0["Campaign"].values)
    locations = np.unique(ds0["Location"].values)

    for campaign in campaigns:
        for location in locations:
            ds_sel = select_ds_camp_loc(ds0, campaign, location)
            if len(ds_sel["time"]) == 0:
                continue
            print(f"\n{campaign} / {location}")

            iwv_da = get_iwv(ds_sel, campaign, location)
            if iwv_da is None:
                print("  No IWV variable found — skipping")
                continue

            for sky in skies:
                print(f"  sky: {sky}")
                ds_cf = apply_sky_mask(ds_sel, sky)

                # ── Check required deviation variables ─────────────────────
                if "Deviations_RTTOV_R24" not in ds_cf or \
                   "Deviations_ARMS_R24"  not in ds_cf:
                    print("  Missing RTTOV/ARMS deviations — skipping")
                    continue

                dep_rttov = get_departures(ds_cf, "Deviations_RTTOV_R24")
                dep_arms  = get_departures(ds_cf, "Deviations_ARMS_R24")
                # shape: (time, N_Channels, elevation)

                iwv_vals = iwv_da.values   # (time,)

                # ── Output directories ────────────────────────────────────
                base = ensure_folder(
                    os.path.join(args.output,
                                 f"{campaign}_{location}", "IWV_scatter"))
                dir_all      = ensure_folder(os.path.join(base, "all"))
                dir_per_chan = ensure_folder(os.path.join(base, "per_channel"))
                dir_per_elev = ensure_folder(os.path.join(base, "per_elevation"))
                dir_per_ec   = ensure_folder(
                    os.path.join(base, "per_elevation_and_channel"))  # NEW
                dir_arms_rttov = ensure_folder(
                    os.path.join(base, "ARMS_vs_RTTOV"))
                dir_arms_rttov_ec = ensure_folder(
                    os.path.join(base, "ARMS_vs_RTTOV_per_ele_chan"))  # NEW

                # ── Plots ─────────────────────────────────────────────────
                plot_model_vs_iwv_all(dep_rttov, dep_arms, iwv_vals,
                                      campaign, location, sky, dir_all)

                plot_model_vs_iwv_per_channel(dep_rttov, dep_arms, iwv_vals,
                                              campaign, location, sky,
                                              dir_per_chan)

                plot_model_vs_iwv_per_elevation(dep_rttov, dep_arms, iwv_vals,
                                                campaign, location, sky,
                                                dir_per_elev)

                plot_model_vs_iwv_per_ele_chan(dep_rttov, dep_arms, iwv_vals,
                                               campaign, location, sky,
                                               dir_per_ec)   # NEW

                plot_arms_vs_rttov(dep_rttov, dep_arms, iwv_vals,
                                   campaign, location, sky, dir_arms_rttov)

                plot_arms_vs_rttov_per_ele_chan(dep_rttov, dep_arms, iwv_vals,
                                                campaign, location, sky,
                                                dir_arms_rttov_ec)  # NEW

                # ── MWR-Deviationen laden ─────────────────────────────────────────────────
                mwr_deps = get_mwr_departures(ds_cf)
                if mwr_deps:
                    dir_mwr_all      = ensure_folder(os.path.join(base, "MWR_all"))
                    dir_mwr_chan     = ensure_folder(os.path.join(base, "MWR_per_channel"))
                    dir_mwr_elev     = ensure_folder(os.path.join(base, "MWR_per_elevation"))
                    dir_mwr_ele_chan = ensure_folder(os.path.join(base, "MWR_per_ele_chan"))

                    plot_mwr_vs_iwv_all(mwr_deps, iwv_vals,
                                        campaign, location, sky, dir_mwr_all)
                    plot_mwr_vs_iwv_per_channel(mwr_deps, iwv_vals,
                                                campaign, location, sky, dir_mwr_chan)
                    plot_mwr_vs_iwv_per_elevation(mwr_deps, iwv_vals,
                                                  campaign, location, sky, dir_mwr_elev)
                    plot_mwr_vs_iwv_per_ele_chan(mwr_deps, iwv_vals,
                                                 campaign, location, sky, dir_mwr_ele_chan)

                plt.close("all")

    print("\nDone.")



#!/usr/bin/env python3
"""
Antenna Fault Heat Map (Az/El) — GUI

Features
- Load METRICS CSV (time series) and EVENTS TXT (log lines)
- Auto-detect azimuth/elevation/time columns when possible
- Choose/override column mappings via dropdowns
- Plot:
  - Motion track (from metrics)
  - Fault points (from events and/or metrics)
  - Heat map of fault density over az/el bins
- Filter by time range and by fault keyword(s)
- Export plot as PNG

Install
  pip install numpy pandas matplotlib

Run
  python antenna_fault_heatmap_gui.py
"""

from __future__ import annotations

import os
import re
import sys
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ----------------------------
# Helpers
# ----------------------------

def _safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def _normalize_az(az_deg: float) -> float:
    # Keep [0, 360)
    return az_deg % 360.0

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _try_parse_datetime_series(s: pd.Series) -> Optional[pd.Series]:
    # Try a few common datetime parses
    for infer in (True, False):
        try:
            out = pd.to_datetime(s, errors="coerce", infer_datetime_format=infer, utc=False)
            if out.notna().sum() >= max(3, int(0.2 * len(s))):
                return out
        except Exception:
            continue
    return None

def _auto_pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(cols_l):
            if cand in c:
                return cols[i]
    return None

def _parse_events_txt(path: str) -> pd.DataFrame:
    """
    Parse a TXT log into a DataFrame with columns:
      time (datetime64[ns] or NaT),
      az (float or NaN),
      el (float or NaN),
      label (str)

    It attempts to parse lines that look like:
      2026-01-23 10:35:12 ... AZ=123.4 EL=56.7 FAULT=OVERLOAD
      2026/01/23 10:35:12 ... az: 123.4 elev: 56.7 alarm: X
      2026-01-23T10:35:12Z ... Azimuth 123.4 Elevation 56.7 ...
    """
    # Date/time patterns (loose)
    dt_patterns = [
        r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})",
        r"(?P<ts>\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}:\d{2})",
        r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})",
        r"(?P<ts>\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2})",
    ]

    # Az/El patterns (very loose)
    az_patterns = [
        r"(?:\bAZ\b|\bazimuth\b)\s*[:=]?\s*(?P<az>-?\d+(?:\.\d+)?)",
        r"\baz\s*[:=]\s*(?P<az>-?\d+(?:\.\d+)?)",
    ]
    el_patterns = [
        r"(?:\bEL\b|\belev(?:ation)?\b)\s*[:=]?\s*(?P<el>-?\d+(?:\.\d+)?)",
        r"\bel\s*[:=]\s*(?P<el>-?\d+(?:\.\d+)?)",
    ]

    # Fault label patterns
    label_patterns = [
        r"(?:\bFAULT\b|\bALARM\b|\bERROR\b)\s*[:=]?\s*(?P<label>[A-Za-z0-9_\-./ ]+)",
        r"(?:\bcode\b|\bid\b)\s*[:=]?\s*(?P<label>[A-Za-z0-9_\-./ ]+)",
    ]

    compiled_dt = [re.compile(p) for p in dt_patterns]
    compiled_az = [re.compile(p, flags=re.IGNORECASE) for p in az_patterns]
    compiled_el = [re.compile(p, flags=re.IGNORECASE) for p in el_patterns]
    compiled_label = [re.compile(p, flags=re.IGNORECASE) for p in label_patterns]

    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            ts = None
            for rdt in compiled_dt:
                m = rdt.search(raw)
                if m:
                    ts = m.group("ts")
                    break

            az = None
            for raz in compiled_az:
                m = raz.search(raw)
                if m:
                    az = _safe_float(m.group("az"))
                    break

            el = None
            for rel in compiled_el:
                m = rel.search(raw)
                if m:
                    el = _safe_float(m.group("el"))
                    break

            label = None
            for rlb in compiled_label:
                m = rlb.search(raw)
                if m:
                    label = m.group("label").strip()
                    break

            if label is None:
                # If no explicit label, use a short snippet (still useful for filtering)
                label = raw[:80]

            rows.append({"time_raw": ts, "az": az, "el": el, "label": label, "line": raw})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "time_raw" in df.columns:
        df["time"] = pd.to_datetime(df["time_raw"], errors="coerce", infer_datetime_format=True)
    else:
        df["time"] = pd.NaT

    # Normalize
    df["az"] = df["az"].astype(float)
    df["el"] = df["el"].astype(float)

    return df


def _to_polar_r(el_deg: float) -> float:
    """
    Convert elevation degrees to polar radius for a 'sky plot' style:
      - Center = zenith (el=90) -> r=0
      - Outer ring = horizon (el=0) -> r=90
    """
    return 90.0 - el_deg


# ----------------------------
# GUI App
# ----------------------------

@dataclass
class DataState:
    metrics_path: Optional[str] = None
    events_path: Optional[str] = None
    df_metrics: Optional[pd.DataFrame] = None
    df_events: Optional[pd.DataFrame] = None

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Antenna Fault Heat Map (Az/El) — Sky Plot")
        self.geometry("1200x780")

        self.state = DataState()

        # Selected column mappings
        self.metrics_time_col = tk.StringVar(value="")
        self.metrics_az_col = tk.StringVar(value="")
        self.metrics_el_col = tk.StringVar(value="")
        self.metrics_fault_col = tk.StringVar(value="")  # optional (e.g., status/fault code)

        self.events_use = tk.BooleanVar(value=True)
        self.metrics_fault_use = tk.BooleanVar(value=False)

        # Filters
        self.filter_keywords = tk.StringVar(value="")  # comma-separated
        self.tmin_str = tk.StringVar(value="")
        self.tmax_str = tk.StringVar(value="")

        # Binning
        self.az_bin_deg = tk.DoubleVar(value=5.0)
        self.el_bin_deg = tk.DoubleVar(value=5.0)

        # Plot toggles
        self.show_track = tk.BooleanVar(value=True)
        self.show_fault_points = tk.BooleanVar(value=True)
        self.show_heatmap = tk.BooleanVar(value=True)

        self._build_ui()
        self._build_plot()

    # ---------- UI ----------
    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        # Left controls
        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")

        # Right plot
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # File loaders
        lf_files = ttk.LabelFrame(left, text="1) Load Data", padding=10)
        lf_files.pack(fill="x", pady=(0, 10))

        ttk.Button(lf_files, text="Load Metrics CSV", command=self.load_metrics).pack(fill="x")
        self.lbl_metrics = ttk.Label(lf_files, text="(no metrics loaded)")
        self.lbl_metrics.pack(fill="x", pady=(4, 8))

        ttk.Button(lf_files, text="Load Events TXT", command=self.load_events).pack(fill="x")
        self.lbl_events = ttk.Label(lf_files, text="(no events loaded)")
        self.lbl_events.pack(fill="x", pady=(4, 0))

        # Column mapping
        lf_cols = ttk.LabelFrame(left, text="2) Column Mapping (Metrics CSV)", padding=10)
        lf_cols.pack(fill="x", pady=(0, 10))

        self.cmb_time = ttk.Combobox(lf_cols, textvariable=self.metrics_time_col, state="readonly", values=[])
        self.cmb_az = ttk.Combobox(lf_cols, textvariable=self.metrics_az_col, state="readonly", values=[])
        self.cmb_el = ttk.Combobox(lf_cols, textvariable=self.metrics_el_col, state="readonly", values=[])
        self.cmb_fault = ttk.Combobox(lf_cols, textvariable=self.metrics_fault_col, state="readonly", values=[])

        ttk.Label(lf_cols, text="Time column:").pack(anchor="w")
        self.cmb_time.pack(fill="x", pady=(0, 6))

        ttk.Label(lf_cols, text="Azimuth column (deg):").pack(anchor="w")
        self.cmb_az.pack(fill="x", pady=(0, 6))

        ttk.Label(lf_cols, text="Elevation column (deg):").pack(anchor="w")
        self.cmb_el.pack(fill="x", pady=(0, 6))

        ttk.Label(lf_cols, text="Fault/status column (optional):").pack(anchor="w")
        self.cmb_fault.pack(fill="x", pady=(0, 2))

        # Data source toggles
        lf_src = ttk.LabelFrame(left, text="3) Fault Sources", padding=10)
        lf_src.pack(fill="x", pady=(0, 10))

        ttk.Checkbutton(lf_src, text="Use Events TXT (fault points)", variable=self.events_use).pack(anchor="w")
        ttk.Checkbutton(lf_src, text="Use Metrics fault column (points where fault!=0/blank)", variable=self.metrics_fault_use).pack(anchor="w")

        # Filters
        lf_filter = ttk.LabelFrame(left, text="4) Filters", padding=10)
        lf_filter.pack(fill="x", pady=(0, 10))

        ttk.Label(lf_filter, text="Fault keywords (comma-separated, optional):").pack(anchor="w")
        ttk.Entry(lf_filter, textvariable=self.filter_keywords).pack(fill="x", pady=(0, 6))

        ttk.Label(lf_filter, text="Time min (optional, e.g., 2026-01-23 10:00):").pack(anchor="w")
        ttk.Entry(lf_filter, textvariable=self.tmin_str).pack(fill="x", pady=(0, 6))

        ttk.Label(lf_filter, text="Time max (optional):").pack(anchor="w")
        ttk.Entry(lf_filter, textvariable=self.tmax_str).pack(fill="x", pady=(0, 2))

        # Binning & plot options
        lf_plot = ttk.LabelFrame(left, text="5) Plot Options", padding=10)
        lf_plot.pack(fill="x", pady=(0, 10))

        row1 = ttk.Frame(lf_plot)
        row1.pack(fill="x")
        ttk.Label(row1, text="Az bin (deg):").pack(side="left")
        ttk.Entry(row1, textvariable=self.az_bin_deg, width=8).pack(side="left", padx=6)
        ttk.Label(row1, text="El bin (deg):").pack(side="left")
        ttk.Entry(row1, textvariable=self.el_bin_deg, width=8).pack(side="left", padx=6)

        ttk.Checkbutton(lf_plot, text="Show motion track", variable=self.show_track).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(lf_plot, text="Show fault points", variable=self.show_fault_points).pack(anchor="w")
        ttk.Checkbutton(lf_plot, text="Show heat map", variable=self.show_heatmap).pack(anchor="w")

        # Actions
        lf_actions = ttk.LabelFrame(left, text="6) Actions", padding=10)
        lf_actions.pack(fill="x", pady=(0, 10))

        ttk.Button(lf_actions, text="Update Plot", command=self.update_plot).pack(fill="x")
        ttk.Button(lf_actions, text="Export PNG", command=self.export_png).pack(fill="x", pady=(6, 0))

        # Status
        self.status = ttk.Label(left, text="Load metrics and/or events, then Update Plot.", wraplength=340)
        self.status.pack(fill="x", pady=(6, 0))

        # Plot container
        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill="both", expand=True)

    def _build_plot(self):
        self.fig = plt.Figure(figsize=(7.5, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="polar")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._style_polar_axes()
        self.canvas.draw()

    def _style_polar_axes(self):
        ax = self.ax
        ax.clear()

        # Polar: theta=azimuth, r=90-el
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise: E at right

        # Elevation rings: r = 90-el
        # r=0 => el=90, r=90 => el=0
        ax.set_rlim(0, 90)

        # Ring labels in elevation degrees (convert r->el)
        rings = [0, 15, 30, 45, 60, 75, 90]
        ax.set_yticks(rings)
        ax.set_yticklabels([f"{int(90-r)}°" for r in rings])  # label as elevation

        ax.set_title("Antenna Az/El Fault Heat Map (Sky Plot)", pad=18)

        # Light grid
        ax.grid(True, alpha=0.35)

    # ---------- Data loading ----------
    def load_metrics(self):
        path = filedialog.askopenfilename(
            title="Select metrics CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not read CSV:\n{e}")
            return

        if df.empty:
            messagebox.showwarning("Empty", "CSV loaded but it is empty.")
            return

        self.state.metrics_path = path
        self.state.df_metrics = df

        self.lbl_metrics.config(text=os.path.basename(path))

        cols = list(df.columns)
        self.cmb_time["values"] = cols
        self.cmb_az["values"] = cols
        self.cmb_el["values"] = cols
        self.cmb_fault["values"] = [""] + cols

        # Auto-detect
        time_guess = _auto_pick_col(cols, ["time", "timestamp", "datetime", "date"])
        az_guess = _auto_pick_col(cols, ["az", "azimuth"])
        el_guess = _auto_pick_col(cols, ["el", "elev", "elevation"])

        fault_guess = _auto_pick_col(cols, ["fault", "alarm", "error", "status", "code"])

        self.metrics_time_col.set(time_guess or cols[0])
        self.metrics_az_col.set(az_guess or "")
        self.metrics_el_col.set(el_guess or "")
        self.metrics_fault_col.set(fault_guess or "")

        self.status.config(text="Metrics loaded. Verify column mappings, then load events or Update Plot.")

    def load_events(self):
        path = filedialog.askopenfilename(
            title="Select events TXT",
            filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            df = _parse_events_txt(path)
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not parse TXT:\n{e}")
            return

        self.state.events_path = path
        self.state.df_events = df
        self.lbl_events.config(text=os.path.basename(path))
        self.status.config(text="Events loaded. Update Plot when ready.")

    # ---------- Filtering ----------
    def _parse_tmin_tmax(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        tmin = self.tmin_str.get().strip()
        tmax = self.tmax_str.get().strip()
        tmin_ts = None
        tmax_ts = None

        if tmin:
            try:
                tmin_ts = pd.to_datetime(tmin, errors="raise")
            except Exception:
                messagebox.showwarning("Time parse", f"Could not parse time min: {tmin}")

        if tmax:
            try:
                tmax_ts = pd.to_datetime(tmax, errors="raise")
            except Exception:
                messagebox.showwarning("Time parse", f"Could not parse time max: {tmax}")

        return tmin_ts, tmax_ts

    def _keyword_list(self) -> List[str]:
        raw = self.filter_keywords.get().strip()
        if not raw:
            return []
        return [k.strip().lower() for k in raw.split(",") if k.strip()]

    def _filter_fault_df(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        if df is None or df.empty:
            return df

        out = df.copy()

        # Time filter (if time exists)
        tmin_ts, tmax_ts = self._parse_tmin_tmax()
        if "time" in out.columns:
            if tmin_ts is not None:
                out = out[out["time"] >= tmin_ts]
            if tmax_ts is not None:
                out = out[out["time"] <= tmax_ts]

        # Keyword filter
        keys = self._keyword_list()
        if keys and label_col in out.columns:
            lab = out[label_col].astype(str).str.lower()
            mask = np.zeros(len(out), dtype=bool)
            for k in keys:
                mask |= lab.str.contains(re.escape(k), na=False)
            out = out[mask]

        return out

    # ---------- Plotting ----------
    def update_plot(self):
        try:
            self._style_polar_axes()

            # Prepare track from metrics
            track = self._extract_track_from_metrics()

            # Prepare faults
            faults = self._collect_fault_points(track)

            # Plot components
            if self.show_heatmap.get():
                self._plot_heatmap(faults)

            if self.show_track.get():
                self._plot_track(track)

            if self.show_fault_points.get():
                self._plot_fault_points(faults)

            self.canvas.draw()

            n_track = 0 if track is None else len(track)
            n_fault = 0 if faults is None else len(faults)
            self.status.config(text=f"Plotted. Track points: {n_track:,} | Fault points: {n_fault:,}")
        except Exception as e:
            messagebox.showerror("Plot error", f"Failed to plot:\n{e}")

    def _extract_track_from_metrics(self) -> Optional[pd.DataFrame]:
        df = self.state.df_metrics
        if df is None or df.empty:
            return None

        tcol = self.metrics_time_col.get().strip()
        azcol = self.metrics_az_col.get().strip()
        elcol = self.metrics_el_col.get().strip()

        if not tcol or not azcol or not elcol:
            # Allow plotting faults only (events)
            return None

        out = df[[tcol, azcol, elcol]].copy()
        out.columns = ["time_raw", "az", "el"]

        # Parse time
        tseries = _try_parse_datetime_series(out["time_raw"])
        if tseries is None:
            out["time"] = pd.NaT
        else:
            out["time"] = tseries

        # Clean numeric
        out["az"] = pd.to_numeric(out["az"], errors="coerce")
        out["el"] = pd.to_numeric(out["el"], errors="coerce")

        out = out.dropna(subset=["az", "el"])
        out["az"] = out["az"].map(_normalize_az)
        out["el"] = out["el"].clip(lower=0, upper=90)

        # Time filter
        tmin_ts, tmax_ts = self._parse_tmin_tmax()
        if "time" in out.columns and out["time"].notna().any():
            if tmin_ts is not None:
                out = out[out["time"] >= tmin_ts]
            if tmax_ts is not None:
                out = out[out["time"] <= tmax_ts]

        return out.reset_index(drop=True)

    def _collect_fault_points(self, track: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Returns DataFrame with columns: az, el, label, source
        """
        pieces = []

        # Faults from events TXT
        if self.events_use.get():
            ev = self.state.df_events
            if ev is not None and not ev.empty:
                ev2 = ev.copy()
                # Keep only rows with az/el present
                ev2["az"] = pd.to_numeric(ev2["az"], errors="coerce")
                ev2["el"] = pd.to_numeric(ev2["el"], errors="coerce")
                ev2 = ev2.dropna(subset=["az", "el"])
                ev2["az"] = ev2["az"].map(_normalize_az)
                ev2["el"] = ev2["el"].clip(lower=0, upper=90)
                ev2 = self._filter_fault_df(ev2, label_col="label")
                ev2["source"] = "events_txt"
                pieces.append(ev2[["az", "el", "label", "source"]])

        # Faults from metrics fault/status column (if enabled)
        if self.metrics_fault_use.get():
            dfm = self.state.df_metrics
            fcol = self.metrics_fault_col.get().strip()
            azcol = self.metrics_az_col.get().strip()
            elcol = self.metrics_el_col.get().strip()

            if dfm is not None and not dfm.empty and fcol and azcol and elcol:
                tmp = dfm[[azcol, elcol, fcol]].copy()
                tmp.columns = ["az", "el", "label"]

                tmp["az"] = pd.to_numeric(tmp["az"], errors="coerce")
                tmp["el"] = pd.to_numeric(tmp["el"], errors="coerce")

                # Fault definition: label not blank/zero/OK
                lab = tmp["label"].astype(str).str.strip()
                is_fault = (
                    lab.ne("") &
                    ~lab.str.fullmatch(r"0+(\.0+)?", na=False) &
                    ~lab.str.lower().isin(["ok", "normal", "none", "nan"])
                )

                tmp = tmp[is_fault].dropna(subset=["az", "el"])
                tmp["az"] = tmp["az"].map(_normalize_az)
                tmp["el"] = tmp["el"].clip(lower=0, upper=90)

                tmp = self._filter_fault_df(tmp, label_col="label")
                tmp["source"] = "metrics_csv"
                pieces.append(tmp[["az", "el", "label", "source"]])

        if not pieces:
            return pd.DataFrame(columns=["az", "el", "label", "source"])

        out = pd.concat(pieces, ignore_index=True)
        return out

    def _plot_track(self, track: Optional[pd.DataFrame]):
        if track is None or track.empty:
            return
        theta = np.deg2rad(track["az"].to_numpy())
        r = np.array([_to_polar_r(e) for e in track["el"].to_numpy()], dtype=float)

        # Track line (motion)
        self.ax.plot(theta, r, linewidth=2.0)

        # Marker at start (small)
        self.ax.scatter(theta[:1], r[:1], s=35, marker="^")

    def _plot_fault_points(self, faults: pd.DataFrame):
        if faults is None or faults.empty:
            return
        theta = np.deg2rad(faults["az"].to_numpy())
        r = np.array([_to_polar_r(e) for e in faults["el"].to_numpy()], dtype=float)

        # Points
        self.ax.scatter(theta, r, s=28, alpha=0.9)

    def _plot_heatmap(self, faults: pd.DataFrame):
        if faults is None or faults.empty:
            return

        az_bin = float(self.az_bin_deg.get())
        el_bin = float(self.el_bin_deg.get())

        az_bin = _clamp(az_bin, 1.0, 60.0)
        el_bin = _clamp(el_bin, 1.0, 45.0)

        az_edges = np.arange(0, 360 + az_bin, az_bin)
        el_edges = np.arange(0, 90 + el_bin, el_bin)

        az = faults["az"].to_numpy(dtype=float)
        el = faults["el"].to_numpy(dtype=float)

        # 2D histogram in az/el space
        H, az_e, el_e = np.histogram2d(az, el, bins=[az_edges, el_edges])

        # Convert to polar mesh:
        # - theta edges: az edges
        # - r edges: 90 - el edges (note reverse)
        theta_edges = np.deg2rad(az_e)
        r_edges = 90.0 - el_e  # el=0 -> r=90, el=90 -> r=0 (descending)

        # histogram2d returns H shape (len(az_edges)-1, len(el_edges)-1)
        # but our r_edges is descending; we want ascending r for pcolormesh
        # So flip elevation axis:
        H_flip = np.flip(H, axis=1)
        r_edges_asc = np.sort(r_edges)

        # Make meshgrid
        T, R = np.meshgrid(theta_edges, r_edges_asc, indexing="ij")

        # pcolormesh expects C shape (M-1, N-1) if X,Y are (M,N)
        # Here T,R are (len(theta_edges), len(r_edges_asc))
        # so C should be (len(theta_edges)-1, len(r_edges_asc)-1)
        # H_flip currently (len(az_edges)-1, len(el_edges)-1) matches that.
        self.ax.pcolormesh(T, R, H_flip, shading="auto", alpha=0.45)

    # ---------- Export ----------
    def export_png(self):
        out = filedialog.asksaveasfilename(
            title="Save plot as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not out:
            return
        try:
            self.fig.savefig(out, dpi=200, bbox_inches="tight")
            self.status.config(text=f"Saved: {out}")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not save PNG:\n{e}")


def main():
    try:
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

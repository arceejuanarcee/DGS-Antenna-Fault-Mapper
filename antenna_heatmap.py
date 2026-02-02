#!/usr/bin/env python3
"""
Antenna Fault Heatmap GUI (Az/El Skyplot)
- User selects a root folder that contains:
    root/
      Events/   -> Events_YYYY-MM-DD_*.log.txt
      Metrics/  -> Metrics_YYYY-MM-DD_*.log.csv
- User selects Year, Month (All or specific), Day (All or specific)
- Script auto-loads matching files, matches events -> nearest metrics az/el, and plots:
    - heatmap (fault density in az/el bins)
    - fault points
    - motion track (optional)

Install:
  pip install pandas numpy matplotlib
Run:
  python antenna_fault_heatmap_gui.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# CONFIG (defaults)
# =========================

MATCH_TOLERANCE_SEC = 5   # events must be within this many seconds of a metrics row
AZ_BINS_DEG = 5           # heatmap az bin width
EL_BINS_DEG = 5           # heatmap el bin width

# Fault detection: treat events whose message begins with one of these as a "fault"
FAULT_PREFIXES = (
    "Fault:",
    "FAULT:",
    "Alarm:",
    "ALARM:",
    "Error:",
    "ERROR:",
    "Motion: Axis error",
)

# Filename patterns
# Metrics_2025-12-19_00-00-00.log.csv
# Events_2025-12-19_00-29-31.log.txt
METRICS_RE = re.compile(r"^Metrics_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.csv$", re.IGNORECASE)
EVENTS_RE  = re.compile(r"^Events_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.txt$", re.IGNORECASE)

# Events line pattern:
# 2025-12-19 06:05:59.638,Fault: ...
EVENT_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?),\s*(?P<msg>.*)$"
)


# =========================
# Helpers
# =========================

def _parse_dt(s: str) -> datetime:
    # supports optional fractional seconds
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
        base = datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
        return base.replace(microsecond=int(frac))
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def _to_polar_r(el_deg: float) -> float:
    # center=zenith (90), outer=horizon (0)
    return 90.0 - el_deg

def _date_range_from_selection(year: int, month: str, day: str) -> Tuple[datetime, datetime]:
    """
    month: "All" or "01".."12"
    day: "All" or "01".."31"
    returns [start, end)
    """
    if month == "All":
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)

    m = int(month)
    if day == "All":
        start = datetime(year, m, 1)
        end = datetime(year + 1, 1, 1) if m == 12 else datetime(year, m + 1, 1)
        return start, end

    d = int(day)
    start = datetime(year, m, d)
    return start, start + timedelta(days=1)

def _is_fault_message(msg: str) -> bool:
    return any(msg.startswith(pfx) for pfx in FAULT_PREFIXES)

def _find_files_in_range(metrics_dir: Path, events_dir: Path, start: datetime, end: datetime) -> Tuple[List[Path], List[Path]]:
    metrics_files: List[Path] = []
    events_files: List[Path] = []

    if metrics_dir.exists():
        for p in metrics_dir.iterdir():
            m = METRICS_RE.match(p.name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            if start <= d < end:
                metrics_files.append(p)

    if events_dir.exists():
        for p in events_dir.iterdir():
            m = EVENTS_RE.match(p.name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            if start <= d < end:
                events_files.append(p)

    metrics_files.sort()
    events_files.sort()
    return metrics_files, events_files

def _list_available_years(metrics_dir: Path, events_dir: Path) -> List[int]:
    years = set()

    if metrics_dir.exists():
        for p in metrics_dir.iterdir():
            m = METRICS_RE.match(p.name)
            if m:
                years.add(int(m.group(1)[:4]))

    if events_dir.exists():
        for p in events_dir.iterdir():
            m = EVENTS_RE.match(p.name)
            if m:
                years.add(int(m.group(1)[:4]))

    return sorted(years)

def _list_available_dates_for_year(metrics_dir: Path, events_dir: Path, year: int) -> List[str]:
    dates = set()

    if metrics_dir.exists():
        for p in metrics_dir.iterdir():
            m = METRICS_RE.match(p.name)
            if m and m.group(1).startswith(f"{year:04d}-"):
                dates.add(m.group(1))

    if events_dir.exists():
        for p in events_dir.iterdir():
            m = EVENTS_RE.match(p.name)
            if m and m.group(1).startswith(f"{year:04d}-"):
                dates.add(m.group(1))

    return sorted(dates)

def _load_metrics_csv(path: Path) -> pd.DataFrame:
    """
    Metrics CSV may have junk lines before the real CSV header.
    We find the first row that starts with "Time" and read from there.
    Required columns:
      Time
      Antenna azimuth (deg)
      Antenna elevation (deg)
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('"Time"') or s.startswith("Time,") or s.startswith("Time\t"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Metrics file has no CSV header row starting with Time: {path.name}")

    csv_text = "".join(lines[header_idx:])
    df = pd.read_csv(pd.io.common.StringIO(csv_text), engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]

    needed = {"Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Antenna azimuth (deg)"] = pd.to_numeric(df["Antenna azimuth (deg)"], errors="coerce")
    df["Antenna elevation (deg)"] = pd.to_numeric(df["Antenna elevation (deg)"], errors="coerce")

    df = df.dropna(subset=["Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"])
    df = df.sort_values("Time")

    df["Antenna azimuth (deg)"] = df["Antenna azimuth (deg)"] % 360.0
    df["Antenna elevation (deg)"] = df["Antenna elevation (deg)"].clip(lower=0.0, upper=90.0)

    return df[["Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"]].reset_index(drop=True)

@dataclass
class EventRow:
    time: datetime
    msg: str

def _load_events_txt(path: Path) -> List[EventRow]:
    out: List[EventRow] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = EVENT_LINE_RE.match(line)
            if not m:
                continue
            ts = _parse_dt(m.group("ts"))
            msg = m.group("msg").strip()
            out.append(EventRow(ts, msg))
    return out

def _build_fault_points(metrics: pd.DataFrame, events: List[EventRow], tolerance_sec: int) -> pd.DataFrame:
    """
    Match each fault event to nearest metrics timestamp using merge_asof.
    Returns DataFrame columns:
      Time_event, msg, Time_metrics, az_deg, el_deg
    """
    if metrics.empty:
        return pd.DataFrame(columns=["Time_event", "msg", "Time_metrics", "az_deg", "el_deg"])

    fault_events = [(e.time, e.msg) for e in events if _is_fault_message(e.msg)]
    if not fault_events:
        return pd.DataFrame(columns=["Time_event", "msg", "Time_metrics", "az_deg", "el_deg"])

    df_ev = pd.DataFrame(fault_events, columns=["Time_event", "msg"]).sort_values("Time_event")
    df_met = metrics.rename(columns={
        "Time": "Time_metrics",
        "Antenna azimuth (deg)": "az_deg",
        "Antenna elevation (deg)": "el_deg",
    }).sort_values("Time_metrics")

    merged = pd.merge_asof(
        df_ev,
        df_met,
        left_on="Time_event",
        right_on="Time_metrics",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_sec)
    )
    merged = merged.dropna(subset=["Time_metrics", "az_deg", "el_deg"]).reset_index(drop=True)
    return merged


# =========================
# GUI App
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Antenna Fault Heat Map (Az/El) — Auto Match Events ↔ Metrics")
        self.geometry("1200x780")

        self.root_dir: Optional[Path] = None
        self.events_dir: Optional[Path] = None
        self.metrics_dir: Optional[Path] = None

        self.year_var = tk.StringVar(value="")
        self.month_var = tk.StringVar(value="All")
        self.day_var = tk.StringVar(value="All")

        self.show_track_var = tk.BooleanVar(value=True)
        self.show_fault_points_var = tk.BooleanVar(value=True)
        self.show_heatmap_var = tk.BooleanVar(value=True)

        self.tolerance_var = tk.IntVar(value=MATCH_TOLERANCE_SEC)
        self.az_bin_var = tk.DoubleVar(value=AZ_BINS_DEG)
        self.el_bin_var = tk.DoubleVar(value=EL_BINS_DEG)

        self.status_var = tk.StringVar(value="Select root folder that contains Events/ and Metrics/.")

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # Folder selector
        lf_folder = ttk.LabelFrame(left, text="1) Select Root Folder", padding=10)
        lf_folder.pack(fill="x", pady=(0, 10))

        ttk.Button(lf_folder, text="Choose Root Folder", command=self.choose_root_folder).pack(fill="x")
        self.folder_label = ttk.Label(lf_folder, text="(none)")
        self.folder_label.pack(fill="x", pady=(6, 0))

        # Date selector
        lf_date = ttk.LabelFrame(left, text="2) Date Selection", padding=10)
        lf_date.pack(fill="x", pady=(0, 10))

        ttk.Label(lf_date, text="Year:").pack(anchor="w")
        self.year_cb = ttk.Combobox(lf_date, textvariable=self.year_var, state="readonly", values=[])
        self.year_cb.pack(fill="x", pady=(0, 6))
        self.year_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_month_day_options())

        ttk.Label(lf_date, text="Month:").pack(anchor="w")
        self.month_cb = ttk.Combobox(lf_date, textvariable=self.month_var, state="readonly",
                                     values=["All"] + [f"{i:02d}" for i in range(1, 13)])
        self.month_cb.pack(fill="x", pady=(0, 6))
        self.month_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_month_day_options())

        ttk.Label(lf_date, text="Day:").pack(anchor="w")
        self.day_cb = ttk.Combobox(lf_date, textvariable=self.day_var, state="readonly", values=["All"])
        self.day_cb.pack(fill="x", pady=(0, 2))

        # Plot options
        lf_opts = ttk.LabelFrame(left, text="3) Plot Options", padding=10)
        lf_opts.pack(fill="x", pady=(0, 10))

        ttk.Checkbutton(lf_opts, text="Show motion track (from Metrics)", variable=self.show_track_var).pack(anchor="w")
        ttk.Checkbutton(lf_opts, text="Show fault points", variable=self.show_fault_points_var).pack(anchor="w")
        ttk.Checkbutton(lf_opts, text="Show heat map", variable=self.show_heatmap_var).pack(anchor="w")

        row = ttk.Frame(lf_opts)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Match tol (sec):").pack(side="left")
        ttk.Entry(row, textvariable=self.tolerance_var, width=6).pack(side="left", padx=6)

        row2 = ttk.Frame(lf_opts)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Az bin (deg):").pack(side="left")
        ttk.Entry(row2, textvariable=self.az_bin_var, width=6).pack(side="left", padx=6)
        ttk.Label(row2, text="El bin (deg):").pack(side="left")
        ttk.Entry(row2, textvariable=self.el_bin_var, width=6).pack(side="left", padx=6)

        # Actions
        lf_actions = ttk.LabelFrame(left, text="4) Actions", padding=10)
        lf_actions.pack(fill="x", pady=(0, 10))

        ttk.Button(lf_actions, text="Load + Plot", command=self.load_and_plot).pack(fill="x")
        ttk.Button(lf_actions, text="Export PNG", command=self.export_png).pack(fill="x", pady=(6, 0))

        # Status
        ttk.Label(left, textvariable=self.status_var, wraplength=340).pack(fill="x", pady=(6, 0))

        # Plot container
        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill="both", expand=True)

    def _build_plot(self):
        self.fig = plt.Figure(figsize=(7.5, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="polar")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._style_axes()
        self.canvas.draw()

    def _style_axes(self):
        ax = self.ax
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)     # clockwise; E right
        ax.set_rlim(0, 90)

        rings = [0, 15, 30, 45, 60, 75, 90]
        ax.set_yticks(rings)
        ax.set_yticklabels([f"{int(90-r)}°" for r in rings])  # elevation labels
        ax.grid(True, alpha=0.35)
        ax.set_title("Antenna Fault Heat Map (Az/El Sky Plot)", pad=18)

    # -------------
    # Folder + date
    # -------------
    def choose_root_folder(self):
        folder = filedialog.askdirectory(title="Select ROOT folder that contains Events/ and Metrics/")
        if not folder:
            return

        root = Path(folder)
        events = root / "Events"
        metrics = root / "Metrics"

        if not events.exists() or not metrics.exists():
            messagebox.showerror(
                "Folder structure not found",
                "Your selected root folder must contain:\n"
                "  Events\\  (txt logs)\n"
                "  Metrics\\ (csv logs)\n\n"
                f"Selected: {root}"
            )
            return

        self.root_dir = root
        self.events_dir = events
        self.metrics_dir = metrics
        self.folder_label.config(text=str(root))

        years = _list_available_years(metrics, events)
        if not years:
            self.year_cb["values"] = []
            self.year_var.set("")
            self.status_var.set("No matching Metrics_/Events_ files found in those folders.")
            return

        self.year_cb["values"] = [str(y) for y in years]
        self.year_var.set(str(years[-1]))  # default most recent year
        self.refresh_month_day_options()

        self.status_var.set("Folder ok. Choose Year/Month/Day then click Load + Plot.")

    def refresh_month_day_options(self):
        if not (self.metrics_dir and self.events_dir):
            return
        if not self.year_var.get().isdigit():
            return

        year = int(self.year_var.get())
        dates = _list_available_dates_for_year(self.metrics_dir, self.events_dir, year)

        # Day dropdown depends on month selection
        month = self.month_var.get()

        valid_days = set()
        if month == "All":
            # day can be All only (or we could allow 01..31 but it’s ambiguous without month)
            self.day_cb["values"] = ["All"]
            self.day_var.set("All")
            return

        # If month fixed, list available days in that month
        for d in dates:
            if d[5:7] == month:
                valid_days.add(d[8:10])

        day_values = ["All"] + sorted(valid_days)
        self.day_cb["values"] = day_values
        if self.day_var.get() not in day_values:
            self.day_var.set("All")

    # -------------
    # Load + plot
    # -------------
    def load_and_plot(self):
        if not (self.metrics_dir and self.events_dir):
            messagebox.showwarning("Missing folder", "Please choose the root folder first.")
            return
        if not self.year_var.get().isdigit():
            messagebox.showwarning("Missing year", "Please select a year.")
            return

        year = int(self.year_var.get())
        month = self.month_var.get()
        day = self.day_var.get()

        start, end = _date_range_from_selection(year, month, day)

        metrics_files, events_files = _find_files_in_range(self.metrics_dir, self.events_dir, start, end)

        if not metrics_files:
            messagebox.showwarning("No metrics files", f"No metrics files found for {start.date()} to {end.date()}.")
            return
        if not events_files:
            messagebox.showwarning("No events files", f"No events files found for {start.date()} to {end.date()}.")

        # Load metrics
        metrics_dfs = []
        for p in metrics_files:
            try:
                metrics_dfs.append(_load_metrics_csv(p))
            except Exception as e:
                messagebox.showerror("Metrics load failed", f"{p.name}\n\n{e}")
                return

        metrics = pd.concat(metrics_dfs, ignore_index=True).sort_values("Time").reset_index(drop=True)

        # Load events
        all_events: List[EventRow] = []
        for p in events_files:
            try:
                all_events.extend(_load_events_txt(p))
            except Exception as e:
                messagebox.showerror("Events load failed", f"{p.name}\n\n{e}")
                return

        tol = int(self.tolerance_var.get())
        faults = _build_fault_points(metrics, all_events, tolerance_sec=tol)

        # Plot
        self._style_axes()

        if self.show_heatmap_var.get():
            self._plot_heatmap(faults)

        if self.show_track_var.get():
            self._plot_track(metrics)

        if self.show_fault_points_var.get():
            self._plot_fault_points(faults)

        self.canvas.draw()

        self.status_var.set(
            f"Loaded metrics files: {len(metrics_files)} | events files: {len(events_files)} | "
            f"matched faults: {len(faults)} (tol={tol}s)"
        )

    def _plot_track(self, metrics: pd.DataFrame):
        if metrics.empty:
            return
        theta = np.deg2rad(metrics["Antenna azimuth (deg)"].to_numpy())
        r = np.array([_to_polar_r(e) for e in metrics["Antenna elevation (deg)"].to_numpy()], dtype=float)
        self.ax.plot(theta, r, linewidth=2.0)
        self.ax.scatter(theta[:1], r[:1], s=35, marker="^")

    def _plot_fault_points(self, faults: pd.DataFrame):
        if faults.empty:
            return
        theta = np.deg2rad(faults["az_deg"].to_numpy(dtype=float))
        r = np.array([_to_polar_r(e) for e in faults["el_deg"].to_numpy(dtype=float)], dtype=float)
        self.ax.scatter(theta, r, s=28, alpha=0.9)

    def _plot_heatmap(self, faults: pd.DataFrame):
        if faults.empty:
            return

        az_bin = float(self.az_bin_var.get())
        el_bin = float(self.el_bin_var.get())
        az_bin = max(1.0, min(60.0, az_bin))
        el_bin = max(1.0, min(45.0, el_bin))

        az_edges = np.arange(0, 360 + az_bin, az_bin)
        el_edges = np.arange(0, 90 + el_bin, el_bin)

        az = faults["az_deg"].to_numpy(dtype=float)
        el = faults["el_deg"].to_numpy(dtype=float)

        H, az_e, el_e = np.histogram2d(az, el, bins=[az_edges, el_edges])

        theta_edges = np.deg2rad(az_e)
        r_edges = 90.0 - el_e  # descending
        H_flip = np.flip(H, axis=1)
        r_edges_asc = np.sort(r_edges)

        T, R = np.meshgrid(theta_edges, r_edges_asc, indexing="ij")
        self.ax.pcolormesh(T, R, H_flip, shading="auto", alpha=0.45)

    # -------------
    # Export
    # -------------
    def export_png(self):
        out = filedialog.asksaveasfilename(
            title="Save plot as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not out:
            return
        try:
            self.fig.savefig(out, dpi=220, bbox_inches="tight")
            self.status_var.set(f"Saved: {out}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Antenna Fault Heatmap GUI (Az/El Skyplot) — ERROR CODE ONLY + CLICK INSPECT

Folder structure:
  Root/
    Events/   -> Events_YYYY-MM-DD_*.log.txt
    Metrics/  -> Metrics_YYYY-MM-DD_*.log.csv

Behavior:
- Only reads lines containing "Error code ####" from Events
- Matches fault timestamp to nearest Metrics timestamp (within tolerance)
- Uses matched az/el to build a frequency heatmap
- Heatmap: bins with 0 faults are WHITE
- Click on a heatmap cell -> shows which fault codes are inside that bin

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
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# Defaults
# =========================
MATCH_TOLERANCE_SEC = 5    # max allowed time difference between event and nearest metrics row
AZ_BIN_DEG = 5             # heatmap az bin width (degrees)
EL_BIN_DEG = 5             # heatmap el bin width (degrees)

# Filename patterns
METRICS_RE = re.compile(r"^Metrics_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.csv$", re.IGNORECASE)
EVENTS_RE  = re.compile(r"^Events_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.txt$", re.IGNORECASE)

# Events line pattern: timestamp at start
EVENT_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?),\s*(?P<msg>.*)$"
)

# Extract error code
ERROR_CODE_RE = re.compile(r"\bError\s*code\s*(?P<code>\d{3,6})\b", re.IGNORECASE)


# =========================
# Helpers
# =========================

def parse_dt(s: str) -> datetime:
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
        base = datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
        return base.replace(microsecond=int(frac))
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def to_polar_r(el_deg: float) -> float:
    # center=zenith (90), outer=horizon (0)
    return 90.0 - el_deg

def date_range_from_selection(year: int, month: str, day: str) -> Tuple[datetime, datetime]:
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

def find_files_in_range(metrics_dir: Path, events_dir: Path, start: datetime, end: datetime) -> Tuple[List[Path], List[Path]]:
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

def list_available_years(metrics_dir: Path, events_dir: Path) -> List[int]:
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

def list_available_dates_for_year(metrics_dir: Path, events_dir: Path, year: int) -> List[str]:
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

def load_metrics_csv(path: Path) -> pd.DataFrame:
    """
    Find header line that starts with Time then read CSV from there.
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

    df = df.dropna(subset=["Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"]).sort_values("Time")
    df["Antenna azimuth (deg)"] = df["Antenna azimuth (deg)"] % 360.0
    df["Antenna elevation (deg)"] = df["Antenna elevation (deg)"].clip(0.0, 90.0)

    return df[["Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"]].reset_index(drop=True)

@dataclass
class FaultEvent:
    time: datetime
    code: str
    msg: str

def load_fault_events_txt(path: Path) -> List[FaultEvent]:
    """
    Only returns lines that contain 'Error code ####' and have a timestamp at start.
    """
    out: List[FaultEvent] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = EVENT_LINE_RE.match(line)
            if not m:
                continue

            msg = m.group("msg").strip()
            code_m = ERROR_CODE_RE.search(msg)
            if not code_m:
                continue

            ts = parse_dt(m.group("ts"))
            code = code_m.group("code")
            out.append(FaultEvent(ts, code, msg))
    return out

def match_faults_to_metrics(metrics: pd.DataFrame, faults: List[FaultEvent], tolerance_sec: int) -> pd.DataFrame:
    """
    Match each fault timestamp to nearest metrics timestamp to obtain az/el.

    Output columns:
      Time_event, code, msg, Time_metrics, az_deg, el_deg
    """
    if metrics.empty or not faults:
        return pd.DataFrame(columns=["Time_event", "code", "msg", "Time_metrics", "az_deg", "el_deg"])

    df_ev = pd.DataFrame(
        [(f.time, f.code, f.msg) for f in faults],
        columns=["Time_event", "code", "msg"]
    ).sort_values("Time_event")

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
        tolerance=pd.Timedelta(seconds=tolerance_sec),
    )
    return merged.dropna(subset=["Time_metrics", "az_deg", "el_deg"]).reset_index(drop=True)


# =========================
# GUI App
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Antenna Fault Frequency Heat Map (Az/El) — Error Code Only")
        self.geometry("1280x820")

        self.root_dir: Optional[Path] = None
        self.events_dir: Optional[Path] = None
        self.metrics_dir: Optional[Path] = None

        self.year_var = tk.StringVar(value="")
        self.month_var = tk.StringVar(value="All")
        self.day_var = tk.StringVar(value="All")

        self.show_heatmap_var = tk.BooleanVar(value=True)
        self.show_fault_points_var = tk.BooleanVar(value=False)

        self.tolerance_var = tk.IntVar(value=MATCH_TOLERANCE_SEC)
        self.az_bin_var = tk.DoubleVar(value=AZ_BIN_DEG)
        self.el_bin_var = tk.DoubleVar(value=EL_BIN_DEG)

        self.status_var = tk.StringVar(value="Choose root folder containing Events/ and Metrics/.")
        self.details_var = tk.StringVar(value="Click a heatmap cell to see details here.")

        # For click inspection:
        self._last_H = None
        self._last_az_edges = None
        self._last_el_edges = None
        self._last_matched = None

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
        lf_folder = ttk.LabelFrame(left, text="1) Root Folder", padding=10)
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
        self.month_cb = ttk.Combobox(
            lf_date, textvariable=self.month_var, state="readonly",
            values=["All"] + [f"{i:02d}" for i in range(1, 13)]
        )
        self.month_cb.pack(fill="x", pady=(0, 6))
        self.month_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_month_day_options())

        ttk.Label(lf_date, text="Day:").pack(anchor="w")
        self.day_cb = ttk.Combobox(lf_date, textvariable=self.day_var, state="readonly", values=["All"])
        self.day_cb.pack(fill="x", pady=(0, 2))

        # Options
        lf_opts = ttk.LabelFrame(left, text="3) Options", padding=10)
        lf_opts.pack(fill="x", pady=(0, 10))

        ttk.Checkbutton(lf_opts, text="Show heat map (frequency)", variable=self.show_heatmap_var).pack(anchor="w")
        ttk.Checkbutton(lf_opts, text="Show fault points (on top of heatmap)", variable=self.show_fault_points_var).pack(anchor="w")

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

        # Details panel
        lf_details = ttk.LabelFrame(left, text="Clicked Cell Details", padding=10)
        lf_details.pack(fill="x", pady=(0, 10))
        ttk.Label(lf_details, textvariable=self.details_var, wraplength=360, justify="left").pack(fill="x")

        ttk.Label(left, textvariable=self.status_var, wraplength=360).pack(fill="x", pady=(6, 0))

        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill="both", expand=True)

    def _build_plot(self):
        self.fig = plt.Figure(figsize=(7.5, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="polar")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._style_axes()

        # Connect click handler
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        self.canvas.draw()

    def _style_axes(self):
        ax = self.ax
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)     # clockwise
        ax.set_rlim(0, 90)

        rings = [0, 15, 30, 45, 60, 75, 90]
        ax.set_yticks(rings)
        ax.set_yticklabels([f"{int(90-r)}°" for r in rings])  # elevation labels
        ax.grid(True, alpha=0.35)
        ax.set_title("Fault Frequency Heat Map (Az/El)", pad=18)

    # -------- Folder + date --------
    def choose_root_folder(self):
        folder = filedialog.askdirectory(title="Select ROOT folder with Events/ and Metrics/")
        if not folder:
            return

        root = Path(folder)
        events = root / "Events"
        metrics = root / "Metrics"

        if not events.exists() or not metrics.exists():
            messagebox.showerror(
                "Folder structure not found",
                "Selected root folder must contain:\n"
                "  Events\\  (txt logs)\n"
                "  Metrics\\ (csv logs)\n\n"
                f"Selected: {root}"
            )
            return

        self.root_dir = root
        self.events_dir = events
        self.metrics_dir = metrics
        self.folder_label.config(text=str(root))

        years = list_available_years(metrics, events)
        if not years:
            self.year_cb["values"] = []
            self.year_var.set("")
            self.status_var.set("No matching Metrics_/Events_ files found.")
            return

        self.year_cb["values"] = [str(y) for y in years]
        self.year_var.set(str(years[-1]))
        self.month_var.set("All")
        self.day_var.set("All")
        self.refresh_month_day_options()

        self.status_var.set("Folder OK. Select Year/Month/Day then Load + Plot.")
        self.details_var.set("Click a heatmap cell to see details here.")

    def refresh_month_day_options(self):
        if not (self.metrics_dir and self.events_dir):
            return
        if not self.year_var.get().isdigit():
            return

        year = int(self.year_var.get())
        dates = list_available_dates_for_year(self.metrics_dir, self.events_dir, year)

        month = self.month_var.get()
        if month == "All":
            self.day_cb["values"] = ["All"]
            self.day_var.set("All")
            return

        valid_days = sorted({d[8:10] for d in dates if d[5:7] == month})
        day_values = ["All"] + valid_days
        self.day_cb["values"] = day_values
        if self.day_var.get() not in day_values:
            self.day_var.set("All")

    # -------- Load + plot --------
    def load_and_plot(self):
        if not (self.metrics_dir and self.events_dir):
            messagebox.showwarning("Missing folder", "Choose the root folder first.")
            return
        if not self.year_var.get().isdigit():
            messagebox.showwarning("Missing year", "Select a year.")
            return

        year = int(self.year_var.get())
        month = self.month_var.get()
        day = self.day_var.get()

        start, end = date_range_from_selection(year, month, day)
        metrics_files, events_files = find_files_in_range(self.metrics_dir, self.events_dir, start, end)

        if not metrics_files:
            messagebox.showwarning("No metrics", f"No metrics files found for range {start.date()} to {end.date()}.")
            return
        if not events_files:
            messagebox.showwarning("No events", f"No events files found for range {start.date()} to {end.date()}.")

        # Load metrics
        metrics_dfs = []
        for p in metrics_files:
            try:
                metrics_dfs.append(load_metrics_csv(p))
            except Exception as e:
                messagebox.showerror("Metrics load failed", f"{p.name}\n\n{e}")
                return
        metrics = pd.concat(metrics_dfs, ignore_index=True).sort_values("Time").reset_index(drop=True)

        # Load coded faults
        faults_all: List[FaultEvent] = []
        for p in events_files:
            try:
                faults_all.extend(load_fault_events_txt(p))
            except Exception as e:
                messagebox.showerror("Events load failed", f"{p.name}\n\n{e}")
                return

        tol = int(self.tolerance_var.get())
        matched = match_faults_to_metrics(metrics, faults_all, tolerance_sec=tol)

        # Plot
        self._style_axes()

        # Reset click-inspection cache
        self._last_H = None
        self._last_az_edges = None
        self._last_el_edges = None
        self._last_matched = matched

        if self.show_heatmap_var.get():
            H, az_edges, el_edges = self._plot_heatmap_white_zero(matched)
            self._last_H = H
            self._last_az_edges = az_edges
            self._last_el_edges = el_edges

        if self.show_fault_points_var.get():
            self._plot_fault_points(matched)

        self.canvas.draw()

        self.status_var.set(
            f"Metrics files: {len(metrics_files)} | Events files: {len(events_files)} | "
            f"Coded faults: {len(faults_all)} | Matched: {len(matched)} (tol={tol}s)"
        )
        self.details_var.set("Click a heatmap cell to see details here.")

    def _plot_fault_points(self, matched: pd.DataFrame):
        if matched.empty:
            return
        theta = np.deg2rad(matched["az_deg"].to_numpy(dtype=float))
        r = np.array([to_polar_r(e) for e in matched["el_deg"].to_numpy(dtype=float)], dtype=float)
        self.ax.scatter(theta, r, s=22, alpha=0.9)

    def _plot_heatmap_white_zero(self, matched: pd.DataFrame):
        """
        Returns (H, az_edges, el_edges) in the *az/el domain*.
        Draws a polar pcolormesh using a masked array so H==0 shows white.
        """
        if matched.empty:
            return None, None, None

        az_bin = float(self.az_bin_var.get())
        el_bin = float(self.el_bin_var.get())
        az_bin = max(1.0, min(60.0, az_bin))
        el_bin = max(1.0, min(45.0, el_bin))

        az_edges = np.arange(0, 360 + az_bin, az_bin)
        el_edges = np.arange(0, 90 + el_bin, el_bin)

        az = matched["az_deg"].to_numpy(dtype=float)
        el = matched["el_deg"].to_numpy(dtype=float)

        H, az_e, el_e = np.histogram2d(az, el, bins=[az_edges, el_edges])

        # Mask zeros so they render as white
        H_masked = np.ma.masked_where(H == 0, H)

        # Convert to polar mesh: theta=az, r=90-el (needs elevation flip)
        theta_edges = np.deg2rad(az_e)
        r_edges_desc = 90.0 - el_e
        H_flip = np.flip(H_masked, axis=1)
        r_edges_asc = np.sort(r_edges_desc)

        T, R = np.meshgrid(theta_edges, r_edges_asc, indexing="ij")

        cmap = plt.cm.Reds.copy()
        cmap.set_bad(color="white")  # masked cells -> white

        self.ax.pcolormesh(
            T, R, H_flip,
            shading="auto",
            cmap=cmap,
            alpha=0.90
        )

        return H, az_edges, el_edges

    # -------- Click handling --------
    def on_plot_click(self, event):
        # Only respond if we have heatmap cache and click is inside axes
        if event.inaxes != self.ax:
            return
        if self._last_H is None or self._last_az_edges is None or self._last_el_edges is None:
            return
        if self._last_matched is None or self._last_matched.empty:
            return
        if event.xdata is None or event.ydata is None:
            return

        # event.xdata is theta (radians), event.ydata is r
        theta = event.xdata
        r = event.ydata

        # Convert to az and el
        az = (np.rad2deg(theta) % 360.0)
        el = 90.0 - r

        az_edges = self._last_az_edges
        el_edges = self._last_el_edges

        # Find bin indices
        iaz = np.searchsorted(az_edges, az, side="right") - 1
        iel = np.searchsorted(el_edges, el, side="right") - 1

        if iaz < 0 or iel < 0 or iaz >= len(az_edges) - 1 or iel >= len(el_edges) - 1:
            self.details_var.set("Clicked outside binned region.")
            return

        az_lo, az_hi = az_edges[iaz], az_edges[iaz + 1]
        el_lo, el_hi = el_edges[iel], el_edges[iel + 1]

        # Filter matched faults that fall in this bin
        m = self._last_matched
        in_bin = m[
            (m["az_deg"] >= az_lo) & (m["az_deg"] < az_hi) &
            (m["el_deg"] >= el_lo) & (m["el_deg"] < el_hi)
        ].copy()

        count = int(self._last_H[iaz, iel]) if self._last_H is not None else len(in_bin)

        if in_bin.empty:
            self.details_var.set(
                f"Bin AZ[{az_lo:.0f}–{az_hi:.0f})°, EL[{el_lo:.0f}–{el_hi:.0f})°\n"
                f"Count: 0"
            )
            return

        # Count by code
        code_counts = in_bin["code"].value_counts().to_dict()
        code_summary = ", ".join([f"{k}:{v}" for k, v in sorted(code_counts.items(), key=lambda kv: (-kv[1], kv[0]))])

        # Show a small sample of messages
        sample_n = 5
        samples = in_bin.sort_values("Time_event")[["Time_event", "code", "msg"]].head(sample_n)
        sample_lines = "\n".join([f"- {t} | {c} | {msg[:80]}" for t, c, msg in samples.to_records(index=False)])

        more = "" if len(in_bin) <= sample_n else f"\n(+{len(in_bin)-sample_n} more)"

        self.details_var.set(
            f"Bin AZ[{az_lo:.0f}–{az_hi:.0f})°, EL[{el_lo:.0f}–{el_hi:.0f})°\n"
            f"Count: {count}\n"
            f"Codes: {code_summary}\n"
            f"Sample:\n{sample_lines}{more}"
        )

    # -------- Export --------
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
    App().mainloop()

if __name__ == "__main__":
    main()

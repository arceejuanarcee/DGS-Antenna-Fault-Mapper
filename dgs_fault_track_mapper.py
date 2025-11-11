#!/usr/bin/env python3
"""
DGS Fault & Track Mapper (CSV/XLSX/TXT) + Hover Tooltips
— robust datetime parsing • safe track pairing • no full-day shading
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
from matplotlib import cm

# --- Optional hover tooltips
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except Exception:
    MPLCURSORS_AVAILABLE = False


# ============================== Config ======================================

FAULT_LIKE = {"fault", "error", "alarm"}
TRACK_LIKE = {"track", "pass", "session", "tracking"}

COL_DATETIME = ["timestamp", "datetime", "dt", "time", "time_utc", "time_local"]
COL_DATE = ["date", "day"]
COL_TIME = ["time", "clock"]
COL_TYPE = ["type", "event", "category", "kind"]
COL_CODE = ["code", "error_code", "fault_code", "errcode", "err_code"]
COL_START = ["start", "start_time", "begin", "window_start"]
COL_END = ["end", "end_time", "finish", "window_end"]
COL_SAT = ["satellite", "sat", "spacecraft", "sc", "target_name"]

# max duration we accept for a track window (guards parsing mistakes)
MAX_TRACK_MINUTES = 120  # 2 hours
# if a start has no stop, we synthesize a short window
SYNTH_WINDOW_MINUTES = 10
# when pairing start→nearest stop, maximum gap to accept
PAIR_MAX_GAP_MINUTES = 120


# ============================== Regexes =====================================

LOG_TS = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*,\s*(.*)$"
)
LOG_TRACK_RANGE = re.compile(
    r"\((\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?),\s*"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\)"
)
LOG_ERROR_CODE = re.compile(r"(?:Error\s*code|Err\.?\s*code)\s*(\d+)", flags=re.I)
LOG_SAT_NAME = re.compile(
    r"Track:\s*(?:Launching|Intercepting)?\s*([A-Z0-9][A-Z0-9\s\-]+?)\s*\(",
    flags=0,
)


# ============================== Helpers =====================================

def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    return None


_TIME_24_RE = re.compile(r"(\b24:00(?::00(?:\.0+)?)?\b)")
_EXPLICIT_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%m-%d-%y %H:%M:%S",
    "%d-%m-%y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%H:%M:%S.%f",
    "%H:%M:%S",
]

def _strip_tz(x: pd.Series) -> pd.Series:
    try:
        return x.dt.tz_convert(None)
    except Exception:
        try:
            return x.dt.tz_localize(None)
        except Exception:
            return x

def _fix_24h(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.str.replace(_TIME_24_RE, "23:59:59.999", regex=True)
    return s

def parse_dt(series: pd.Series) -> pd.Series:
    raw = series.astype(str)
    raw = _fix_24h(raw)
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    mask = out.isna()
    for fmt in _EXPLICIT_DT_FORMATS:
        try:
            trial = pd.to_datetime(raw[mask], format=fmt, errors="coerce")
            out.loc[mask] = trial
            mask = out.isna()
            if not mask.any():
                break
        except Exception:
            pass

    if out.isna().any():
        fallback = pd.to_datetime(raw[out.isna()], errors="coerce")
        out.loc[out.isna()] = fallback

    return _strip_tz(out)

def parse_date_time(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    combined = date_s.astype(str).str.strip() + " " + _fix_24h(time_s.astype(str)).str.strip()
    return parse_dt(combined)


# ============================== TXT Parser ==================================

def parse_txt_log(path: str) -> pd.DataFrame:
    """
    Produces rows with columns:
      timestamp (string),
      type ∈ {'fault','track'},
      code (string or None),
      start (string or None),
      end   (string or None),
      satellite (string or None)

    IMPORTANT:
    - Only explicit "Track: ... (start,end)" lines carry both start+end here.
    - Standalone "track_start"/"track_stop" lines carry start OR end (not both).
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_TS.match(line)
            if not m:
                continue
            ts_str, rest = m.groups()
            rest_low = rest.lower()

            type_guess = None
            code_val = None
            start_dt = None
            end_dt = None
            sat_name = None

            # faults
            err_m = LOG_ERROR_CODE.search(rest)
            if "fault:" in rest_low and err_m:
                type_guess = "fault"
                code_val = err_m.group(1)

            # explicit windows
            tr_m = LOG_TRACK_RANGE.search(rest)
            if ("track:" in rest_low) and tr_m:
                type_guess = "track"
                start_dt, end_dt = tr_m.groups()
                sat_m = LOG_SAT_NAME.search(rest)
                if sat_m:
                    sat_name = sat_m.group(1).strip()

            # starts/stops markers
            if "event (track_start)" in rest_low:
                type_guess = "track"
                start_dt = ts_str
                end_dt = None
            if "event (track_stop)" in rest_low:
                type_guess = "track"
                start_dt = None
                end_dt = ts_str

            if type_guess is None:
                continue

            rows.append({
                "timestamp": ts_str,
                "type": type_guess,
                "code": code_val,
                "start": start_dt,
                "end": end_dt,
                "satellite": sat_name
            })
    return pd.DataFrame(rows)


def load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(path)
    elif ext in (".txt", ".log"):
        return parse_txt_log(path)
    else:
        return pd.read_csv(path)


# ============================== Normalization ===============================

def _pair_starts_stops(track_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build clean windows from:
      A) explicit rows with both dt_start & dt_end
      B) separate start-only and stop-only rows (pair within PAIR_MAX_GAP_MINUTES on same day)
      C) synthesize short window for unpaired starts
    """
    # explicit windows
    explicit = track_rows[track_rows["dt_start"].notna() & track_rows["dt_end"].notna()][
        ["dt_start", "dt_end", "satellite"]
    ].copy()

    # starts / stops
    starts = track_rows[track_rows["dt_start"].notna() & track_rows["dt_end"].isna()][
        ["dt_start", "satellite"]
    ].sort_values("dt_start").reset_index(drop=True)

    stops = track_rows[track_rows["dt_end"].notna() & track_rows["dt_start"].isna()][
        ["dt_end", "satellite"]
    ].sort_values("dt_end").reset_index(drop=True)

    paired = []
    j = 0
    for _, srow in starts.iterrows():
        s_dt = srow["dt_start"]
        s_day = s_dt.date()
        # advance 'j' until stop >= start
        while j < len(stops) and stops.loc[j, "dt_end"] < s_dt:
            j += 1
        if j < len(stops):
            e_dt = stops.loc[j, "dt_end"]
            e_day = e_dt.date()
            gap = (e_dt - s_dt).total_seconds() / 60.0
            if s_day == e_day and 0 < gap <= PAIR_MAX_GAP_MINUTES:
                paired.append({"dt_start": s_dt, "dt_end": e_dt, "satellite": srow["satellite"]})
                j += 1
                continue
        # no acceptable stop → synthesize a short window
        paired.append({
            "dt_start": s_dt,
            "dt_end": s_dt + pd.Timedelta(minutes=SYNTH_WINDOW_MINUTES),
            "satellite": srow["satellite"]
        })

    out = pd.concat([explicit, pd.DataFrame(paired)], ignore_index=True) if len(paired) > 0 else explicit

    # sanitize duration + bounds
    if not out.empty:
        good = (out["dt_end"] > out["dt_start"]) & \
               ((out["dt_end"] - out["dt_start"]) <= pd.Timedelta(minutes=MAX_TRACK_MINUTES))
        out = out[good].copy()

    return out


def extract_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.copy()
    df2.rename(columns={c: c.strip() for c in df2.columns}, inplace=True)

    # flexible cols
    type_col = first_col(df2, COL_TYPE)
    dt_col = first_col(df2, COL_DATETIME)
    date_col = first_col(df2, COL_DATE)
    time_col = first_col(df2, COL_TIME)
    code_col = first_col(df2, COL_CODE)
    start_col = first_col(df2, COL_START)
    end_col = first_col(df2, COL_END)
    sat_col = first_col(df2, COL_SAT)

    # base datetimes
    if dt_col is not None:
        dt_series = parse_dt(df2[dt_col])
    elif date_col is not None and time_col is not None:
        dt_series = parse_date_time(df2[date_col], df2[time_col])
    elif "timestamp" in df2.columns:
        dt_series = parse_dt(df2["timestamp"])
    else:
        dt_series = pd.Series(pd.NaT, index=df2.index, dtype="datetime64[ns]")

    dt_start = parse_dt(df2[start_col]) if start_col else parse_dt(df2["start"]) if "start" in df2.columns else pd.Series(pd.NaT, index=df2.index, dtype="datetime64[ns]")
    dt_end   = parse_dt(df2[end_col])   if end_col   else parse_dt(df2["end"])   if "end"   in df2.columns else pd.Series(pd.NaT, index=df2.index, dtype="datetime64[ns]")

    # fallback base timestamp from start if needed
    base_dt_mask = dt_series.isna() & dt_start.notna()
    dt_series.loc[base_dt_mask] = dt_start.loc[base_dt_mask]

    # code
    code_series = None
    if code_col is not None:
        code_series = pd.to_numeric(df2[code_col].astype(str).str.extract(r"(-?\d+)", expand=False), errors="coerce")
    elif "code" in df2.columns:
        code_series = pd.to_numeric(df2["code"], errors="coerce")

    # type
    if type_col is not None:
        type_vals = df2[type_col].astype(str).str.lower()
    elif "type" in df2.columns:
        type_vals = df2["type"].astype(str).str.lower()
    else:
        type_vals = pd.Series("", index=df2.index)

    # satellite
    if sat_col is not None:
        sat_vals = df2[sat_col].astype(str).str.strip()
    elif "satellite" in df2.columns:
        sat_vals = df2["satellite"].astype(str).str.strip()
    else:
        sat_vals = pd.Series(pd.NA, index=df2.index)

    base = pd.DataFrame({
        "dt": dt_series,
        "dt_start": dt_start,
        "dt_end": dt_end,
        "etype": type_vals,
        "satellite": sat_vals
    })

    # infer missing type
    if (base["etype"] == "").any():
        code_present = code_series.notna() if code_series is not None else pd.Series(False, index=base.index)
        inferred = np.where(code_present, "fault",
                            np.where(base["dt_start"].notna() | base["dt_end"].notna(), "track", "unknown"))
        base.loc[base["etype"] == "", "etype"] = inferred[base["etype"] == ""]

    # labels/positions
    base["day_label"] = base["dt"].dt.strftime("%m-%d-%y")
    base.loc[base["dt"].isna() & base["dt_start"].notna(), "day_label"] = base["dt_start"].dt.strftime("%m-%d-%y")
    base["hour_float"] = base["dt"].dt.hour.add(base["dt"].dt.minute.div(60)).astype(float)

    # ---- FAULTS
    faults = base[base["etype"].isin(FAULT_LIKE | {"fault"})].copy()
    if code_series is not None:
        faults["code_num"] = code_series
        faults = faults[faults["code_num"].notna()]
        faults["code_str"] = "Fault " + faults["code_num"].astype(int).astype(str)
    else:
        faults = faults.iloc[0:0].copy()
    faults = faults[faults["day_label"].notna() & faults["hour_float"].notna()]

    # ---- TRACKS (pair starts/stops + sanitize)
    tr_raw = base[base["etype"].isin(TRACK_LIKE | {"track"})][["dt", "dt_start", "dt_end", "satellite"]].copy()
    # use dt as fallback start/end carriers for explicit rows that came via CSV
    tr_raw["dt_start"] = tr_raw["dt_start"].where(tr_raw["dt_start"].notna(), tr_raw["dt"])
    # (leave dt_end as is; we only care when explicit)
    tracks = _pair_starts_stops(tr_raw)

    if tracks.empty:
        return faults, tracks

    # y1/y2 and guardrails
    tracks["day_label"] = tracks["dt_start"].dt.strftime("%m-%d-%y")
    y1 = tracks["dt_start"].dt.hour + tracks["dt_start"].dt.minute.div(60)
    y2 = tracks["dt_end"].dt.hour + tracks["dt_end"].dt.minute.div(60)

    # swap if reversed (shouldn't happen after pairing, but safe)
    swap = y2 < y1
    y1, y2 = y1.where(~swap, y2), y2.where(~swap, y1)

    # clip to [0,24)
    y1 = y1.clip(lower=0, upper=23.999)
    y2 = y2.clip(lower=0.001, upper=24.0)

    # drop “full day” looking windows (protect against bad parses)
    keep = tracks["day_label"].notna() & (y2 > y1) & ~((y1 <= 0.001) & (y2 >= 23.999))
    tracks = tracks[keep].copy()
    tracks["y1"] = y1.loc[tracks.index]
    tracks["y2"] = y2.loc[tracks.index]

    return faults, tracks


# ============================== Plotting ====================================

def _color_map(keys: List[str]) -> Dict[str, tuple]:
    cmap = cm.get_cmap('tab20')
    return {k: cmap(i % 20) for i, k in enumerate(keys)}

def plot_fault_map(faults_list: List[pd.DataFrame], tracks_list: List[pd.DataFrame], month_title: Optional[str] = None):
    faults = pd.concat(faults_list, ignore_index=True) if faults_list else pd.DataFrame(columns=["day_label","hour_float","code_str","code_num","dt"])
    tracks = pd.concat(tracks_list, ignore_index=True) if tracks_list else pd.DataFrame(columns=["day_label","y1","y2","satellite","dt_start","dt_end"])

    unique_days = sorted(set(
        pd.Index(faults.get("day_label", pd.Series(dtype=str))).dropna().unique().tolist() +
        pd.Index(tracks.get("day_label", pd.Series(dtype=str))).dropna().unique().tolist()
    ))
    day_to_x = {d: i for i, d in enumerate(unique_days)}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    ax.grid(True, linestyle="--", alpha=0.3)

    hover_targets = []
    track_handles = []
    fault_handles = []

    # --- tracks
    if not tracks.empty and len(day_to_x) > 0:
        tr = tracks.copy()
        sat_series = tr["satellite"].astype("string")
        missing = sat_series.isna() | sat_series.str.strip().eq("") | sat_series.str.strip().str.lower().isin({"none","nan"})
        tr["satellite"] = sat_series.mask(missing, np.nan)

        named = tr.dropna(subset=["satellite"])
        sats = sorted(named["satellite"].unique().tolist())
        cmap = _color_map(sats)

        for sat in sats:
            grp = named[named["satellite"] == sat]
            color = cmap[sat]
            for _, row in grp.iterrows():
                x = day_to_x.get(row["day_label"])
                if x is None or pd.isna(row["y1"]) or pd.isna(row["y2"]):
                    continue
                y1, y2 = float(row["y1"]), float(row["y2"])
                if y2 <= y1 or (y1 <= 0.001 and y2 >= 23.999):
                    continue
                coll = ax.fill_between([x - 0.35, x + 0.35], y1, y2, color=color, alpha=0.18, zorder=1)
                start_s = pd.to_datetime(row["dt_start"]).strftime("%H:%M")
                end_s = pd.to_datetime(row["dt_end"]).strftime("%H:%M")
                text = f"Track: {sat}\nDay: {row['day_label']}\nStart–End: {start_s}–{end_s}"
                setattr(coll, "_hover_text", text)
                hover_targets.append((coll, text, False))
            track_handles.append(Patch(facecolor=color, alpha=0.25, label=sat))

        unnamed = tr[tr["satellite"].isna()]
        for _, row in unnamed.iterrows():
            x = day_to_x.get(row["day_label"])
            if x is None or pd.isna(row["y1"]) or pd.isna(row["y2"]):
                continue
            y1, y2 = float(row["y1"]), float(row["y2"])
            if y2 <= y1 or (y1 <= 0.001 and y2 >= 23.999):
                continue
            coll = ax.fill_between([x - 0.35, x + 0.35], y1, y2, color=(0.5,0.5,0.5,1), alpha=0.12, zorder=1)
            start_s = pd.to_datetime(row["dt_start"]).strftime("%H:%M")
            end_s = pd.to_datetime(row["dt_end"]).strftime("%H:%M")
            text = f"Track Window\nDay: {row['day_label']}\nStart–End: {start_s}–{end_s}"
            setattr(coll, "_hover_text", text)
            hover_targets.append((coll, text, False))

    # --- faults
    if not faults.empty and len(day_to_x) > 0:
        for name, grp in faults.groupby("code_str", dropna=True):
            xs = grp["day_label"].map(day_to_x)
            keep = xs.notna() & grp["hour_float"].notna()
            xs = xs[keep].astype(float).values
            ys = grp.loc[keep, "hour_float"].astype(float).clip(lower=0, upper=23.999).values
            if len(xs) == 0:
                continue
            sc = ax.scatter(xs, ys, s=35, label=name, zorder=3)
            fault_handles.append(sc)

            texts = []
            for _, r in grp.loc[keep].iterrows():
                if pd.notna(r.get("dt")):
                    ts = pd.to_datetime(r["dt"]).strftime("%Y-%m-%d %H:%M")
                else:
                    ts = f"{r['day_label']} @ {float(r['hour_float']):.2f}h"
                texts.append(f"{name}\nTime: {ts}")
            setattr(sc, "_hover_texts", texts)
            hover_targets.append((sc, texts, True))

    # axes
    ax.set_xlim(-0.5, (len(unique_days) - 0.5) if unique_days else 0.5)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 3))
    ax.set_ylabel("Time (24 Hour)")

    ax.set_xticks(range(len(unique_days)))
    ax.set_xticklabels(unique_days, rotation=45, ha="right")
    ax.set_xlabel("Day")
    ax.set_title(month_title or "DGS Fault & Track Map")

    # legends
    legend_y = -0.22
    all_h = fault_handles + track_handles
    if all_h:
        ax.legend(
            all_h, [h.get_label() for h in all_h],
            loc="upper center", bbox_to_anchor=(0.5, legend_y),
            ncol=min(6, max(1, len(all_h))), frameon=False
        )
    plt.subplots_adjust(bottom=0.28)
    plt.tight_layout()

    # hover
    if MPLCURSORS_AVAILABLE and hover_targets:
        artists = [a for (a, _, _) in hover_targets]
        cursor = mplcursors.cursor(artists, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            art = sel.artist
            if hasattr(art, "_hover_texts"):
                idx = sel.index
                texts = getattr(art, "_hover_texts", [])
                sel.annotation.set_text(texts[idx] if idx is not None and 0 <= idx < len(texts) else art.get_label())
            elif hasattr(art, "_hover_text"):
                sel.annotation.set_text(getattr(art, "_hover_text"))
            else:
                sel.annotation.set_text(art.get_label())
    return fig, ax


# ============================== GUI =========================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DGS Fault & Track Mapper (hover enabled)")
        self.geometry("1200x780")
        self.files: List[str] = []

        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.title_var = tk.StringVar(value="DGS Fault & Track Map")
        tk.Label(top, text="Plot Title:").pack(side=tk.LEFT)
        tk.Entry(top, textvariable=self.title_var, width=40).pack(side=tk.LEFT, padx=5)

        tk.Button(top, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Clear List", command=self.clear_files).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Plot", command=self.make_plot).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Save Plot", command=self.save_plot).pack(side=tk.LEFT, padx=5)

        self.listbox = tk.Listbox(self, height=6)
        self.listbox.pack(fill=tk.X, padx=10, pady=(0,10))

        self.fig, self.ax = plt.subplots(figsize=(12,6), dpi=120)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        if not MPLCURSORS_AVAILABLE:
            tk.Label(self, text="Tip: install 'mplcursors' for hover tooltips:  pip install mplcursors", fg="gray").pack(pady=(0,8))

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select logs (CSV/XLSX/TXT/LOG)",
            filetypes=[
                ("All supported", "*.csv *.xlsx *.xls *.txt *.log"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("Text logs", "*.txt *.log"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert(tk.END, p)

    def clear_files(self):
        self.files = []
        self.listbox.delete(0, tk.END)

    def make_plot(self):
        if not self.files:
            messagebox.showwarning("No files", "Please add one or more files.")
            return

        faults_all, tracks_all, errors = [], [], []

        for path in self.files:
            try:
                df = load_any(path)
                f, t = extract_events(df)
                if not f.empty:
                    f = f.copy(); f["source"] = os.path.basename(path)
                    faults_all.append(f)
                if not t.empty:
                    t = t.copy(); t["source"] = os.path.basename(path)
                    tracks_all.append(t)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        if errors:
            messagebox.showinfo("Some files skipped", "Issues encountered:\n" + "\n".join(errors))

        self.fig.clf()
        self.fig, self.ax = plot_fault_map(faults_all, tracks_all, month_title=self.title_var.get())
        self.canvas.figure = self.fig
        self.canvas.draw()

    def save_plot(self):
        if self.canvas is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Save plot as"
        )
        if not path:
            return
        self.canvas.figure.savefig(path, bbox_inches="tight", dpi=200)
        messagebox.showinfo("Saved", f"Plot saved to:\n{path}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

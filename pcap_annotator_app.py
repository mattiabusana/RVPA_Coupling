import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rv_coupling import compute_pcap, compute_wedge_empirical


st.set_page_config(page_title="Pcap/T0 Annotator", layout="wide")
st.title("Pcap/T0 Batch Annotator")


@st.cache_data
def _load_pa_trace_from_bytes(raw: bytes, name: str):
    xl = pd.ExcelFile(BytesIO(raw))
    if len(xl.sheet_names) < 1:
        return None, None, f"No sheet found in {name}"
    df = pd.read_excel(BytesIO(raw), sheet_name=0)
    if df.shape[1] < 2:
        return None, None, f"Sheet 1 has fewer than 2 columns in {name}"
    return _clean_trace(df.iloc[:, 0], df.iloc[:, 1])


def _clean_trace(t_col, y_col):
    t = pd.to_numeric(t_col, errors="coerce").to_numpy()
    y = pd.to_numeric(y_col, errors="coerce").to_numpy()
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 20:
        return None, None, "Too few valid PA samples"
    if np.any(np.diff(t) <= 0):
        fs_guess = 120.0
        t = np.arange(len(y)) / fs_guess
    return t, y, None


def _suggest_t0(t, y, fs=120.0):
    if len(y) < int(8 * fs):
        return float(t[len(t) // 2])
    win = max(5, int(0.2 * fs))
    ker = np.ones(win) / win
    ys = np.convolve(y, ker, mode="same")
    pre = int(2.5 * fs)
    post = int(2.5 * fs)
    low = pre + int(1.0 * fs)
    high = len(y) - post - int(1.0 * fs)
    best_idx = None
    best_drop = 0.0
    for i in range(low, high):
        pre_m = float(np.mean(ys[i - pre : i - int(0.5 * fs)]))
        post_m = float(np.mean(ys[i + int(0.5 * fs) : i + post]))
        drop = post_m - pre_m
        if drop < best_drop:
            best_drop = drop
            best_idx = i
    if best_idx is None:
        return float(t[len(t) // 2])
    return float(t[best_idx])


def _pa_diastolic_before_t0(t, y, t0, sec=5.0):
    mask = (t >= max(t[0], t0 - sec)) & (t < t0)
    if not np.any(mask):
        return float(np.nan)
    return float(np.percentile(y[mask], 2))


def _records_to_df(records):
    if not records:
        return pd.DataFrame()
    rows = list(records.values())
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["file_index", "file_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _save_records(csv_path, records):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = _records_to_df(records)
    df.to_csv(csv_path, index=False)


def _load_existing_records(csv_path):
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    rec = {}
    for _, r in df.iterrows():
        key = str(r.get("record_key", ""))
        if key:
            rec[key] = r.to_dict()
    return rec


def _build_items(uploaded_files):
    items = []
    for i, uf in enumerate(uploaded_files or []):
        key = f"upload::{i:04d}::{uf.name}"
        items.append(
            {
                "record_key": key,
                "file_index": i + 1,
                "file_name": uf.name,
                "file_ref": uf.name,
                "absolute_file": "",
                "uploaded": uf,
            }
        )
    return items


def _load_item_trace(item):
    raw = item["uploaded"].getvalue()
    return _load_pa_trace_from_bytes(raw, item["file_name"])


with st.sidebar:
    st.header("Batch Setup")
    uploaded_files = st.file_uploader(
        "Upload all .xlsx files (you can drag a whole folder)",
        type=["xlsx"],
        accept_multiple_files=True,
    )
    out_csv_in = st.text_input("Annotations CSV", value="RVPA_Coupling/pcap_annotations.csv")
    load_existing = st.checkbox("Load existing annotations CSV", value=False)
    min_r2 = st.number_input("Min R2 for fit_ok", value=0.80, min_value=0.0, max_value=1.0, step=0.01)
    clear_mem = st.button("Clear annotations in memory")
    if st.button("Reload Files"):
        st.cache_data.clear()
        st.rerun()

out_csv = Path(out_csv_in).resolve()
items = _build_items(uploaded_files)

# If user uploads a different file set, start a fresh in-memory annotation session.
upload_signature = tuple((uf.name, uf.size) for uf in (uploaded_files or []))
prev_signature = st.session_state.get("upload_signature")
if prev_signature is None:
    st.session_state.upload_signature = upload_signature
elif prev_signature != upload_signature:
    st.session_state.upload_signature = upload_signature
    st.session_state.records = {}
    st.session_state.idx = 0
    st.session_state.active_key = None
    st.session_state.t0_candidate = None
    st.session_state.t0_confirmed = None

records_key = (str(out_csv), bool(load_existing))
if "records" not in st.session_state or st.session_state.get("records_source") != records_key:
    st.session_state.records = _load_existing_records(out_csv) if load_existing else {}
    st.session_state.records_source = records_key
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "active_key" not in st.session_state:
    st.session_state.active_key = None
if "t0_candidate" not in st.session_state:
    st.session_state.t0_candidate = None
if "t0_confirmed" not in st.session_state:
    st.session_state.t0_confirmed = None

if clear_mem:
    st.session_state.records = {}
    st.session_state.active_key = None
    st.session_state.t0_candidate = None
    st.session_state.t0_confirmed = None
    st.rerun()

if not items:
    st.info("Upload .xlsx files to start.")
    st.stop()

st.session_state.idx = max(0, min(st.session_state.idx, len(items) - 1))
current = items[st.session_state.idx]

if st.session_state.active_key != current["record_key"]:
    st.session_state.active_key = current["record_key"]
    rec = st.session_state.records.get(current["record_key"], {})
    if "t0_s" in rec and pd.notna(rec["t0_s"]):
        t0_val = float(rec["t0_s"])
    else:
        t_tmp, y_tmp, err_tmp = _load_item_trace(current)
        t0_val = _suggest_t0(t_tmp, y_tmp) if err_tmp is None else 0.0
    st.session_state.t0_candidate = t0_val
    st.session_state.t0_confirmed = rec.get("t0_s", None)

st.caption(f"File {current['file_index']}/{len(items)}: `{current['file_ref']}`")
t_pa, p_pa, err = _load_item_trace(current)
if err:
    st.error(err)
    c1, c2 = st.columns(2)
    if c1.button("Prev", disabled=st.session_state.idx == 0):
        st.session_state.idx -= 1
        st.rerun()
    if c2.button("Next", disabled=st.session_state.idx >= len(items) - 1):
        st.session_state.idx += 1
        st.rerun()
    st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_pa, y=p_pa, mode="lines", name="PA", line=dict(color="royalblue", width=1)))
if st.session_state.t0_candidate is not None:
    fig.add_vline(x=float(st.session_state.t0_candidate), line_color="orange", line_dash="dash", annotation_text="T0 candidate")
if st.session_state.t0_confirmed is not None:
    fig.add_vline(x=float(st.session_state.t0_confirmed), line_color="green", line_dash="solid", annotation_text="T0 confirmed")
fig.update_layout(height=300, title="Full PA signal (overview)", xaxis=dict(nticks=35, tickformat=".2f"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("Select T0 directly on the full time axis")

if len(t_pa) > 1:
    dt = float(np.median(np.diff(t_pa)))
    step_t = max(0.001, dt if np.isfinite(dt) and dt > 0 else 0.01)
else:
    step_t = 0.01

t0_candidate = st.slider(
    "T0 candidate (s)",
    min_value=float(t_pa[0]),
    max_value=float(t_pa[-1]),
    value=float(np.clip(st.session_state.t0_candidate, float(t_pa[0]), float(t_pa[-1]))),
    step=step_t,
)
if abs(t0_candidate - float(st.session_state.t0_candidate)) > 1e-12:
    st.session_state.t0_candidate = float(t0_candidate)
    st.session_state.t0_confirmed = None

fit_res = compute_pcap(t_pa, p_pa, tzero_time=float(st.session_state.t0_candidate))
wedge = compute_wedge_empirical(t_pa, p_pa, tzero_time=float(st.session_state.t0_candidate), n_points=20)
pa_dia = _pa_diastolic_before_t0(t_pa, p_pa, float(st.session_state.t0_candidate), sec=5.0)

pcap_val = np.nan
fit_r2 = np.nan
fit_success = False
if fit_res.get("fit_success"):
    fit_success = True
    pcap_val = float(fit_res["pcap"])
    fit_r2 = float(fit_res["r_squared"])

pcap_lt_pdia = bool(np.isfinite(pcap_val) and np.isfinite(pa_dia) and pcap_val < pa_dia)
fit_ok = bool(fit_success and np.isfinite(fit_r2) and fit_r2 >= min_r2 and pcap_lt_pdia)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pcap", f"{pcap_val:.2f}" if np.isfinite(pcap_val) else "N/A")
c2.metric("Wedge", f"{wedge:.2f}")
c3.metric("R2", f"{fit_r2:.3f}" if np.isfinite(fit_r2) else "N/A")
c4.metric("PA diastolic", f"{pa_dia:.2f}" if np.isfinite(pa_dia) else "N/A")
c5.metric("Pcap < PAdia", "Yes" if pcap_lt_pdia else "No")
st.write(f"Fit status: `{'OK' if fit_ok else 'REVIEW'}`")

z0 = max(float(t_pa[0]), float(st.session_state.t0_candidate) - 8.0)
z1 = min(float(t_pa[-1]), float(st.session_state.t0_candidate) + 12.0)
mask_z = (t_pa >= z0) & (t_pa <= z1)
figz = go.Figure()
figz.add_trace(go.Scatter(x=t_pa[mask_z], y=p_pa[mask_z], mode="lines", name="PA", line=dict(color="lightgray", width=1)))
figz.add_vline(x=float(st.session_state.t0_candidate), line_color="orange", line_dash="dash", annotation_text="T0")
if fit_success:
    a, b, c = fit_res["params"]
    t_rel_fit = fit_res["time_rel"]
    if len(t_rel_fit) > 0 and t_rel_fit[0] > 0:
        t_rel_plot = np.insert(t_rel_fit, 0, 0.0)
    else:
        t_rel_plot = t_rel_fit
    t_abs = float(st.session_state.t0_candidate) + t_rel_plot
    p_fit = a * np.exp(-b * t_rel_plot) + c
    mz = (t_abs >= z0) & (t_abs <= z1)
    figz.add_trace(go.Scatter(x=t_abs[mz], y=p_fit[mz], mode="lines", name="Exp fit", line=dict(color="red", width=2)))
    figz.add_trace(
        go.Scatter(
            x=[float(st.session_state.t0_candidate)],
            y=[pcap_val],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            name="Pcap@t0",
        )
    )
figz.update_layout(height=320, title="Fit preview", xaxis=dict(nticks=35, tickformat=".2f"))
st.plotly_chart(figz, use_container_width=True)

def _make_record():
    return {
        "record_key": current["record_key"],
        "file_index": current["file_index"],
        "file_name": current["file_name"],
        "file": current["file_ref"],
        "absolute_file": current["absolute_file"],
        "t0_s": float(st.session_state.t0_candidate),
        "pcap": float(pcap_val) if np.isfinite(pcap_val) else np.nan,
        "wedge": float(wedge),
        "fit_r2": float(fit_r2) if np.isfinite(fit_r2) else np.nan,
        "fit_success": bool(fit_success),
        "pcap_lt_pdia": bool(pcap_lt_pdia),
        "fit_ok": bool(fit_ok),
        "pa_dia_pre_t0": float(pa_dia) if np.isfinite(pa_dia) else np.nan,
    }


c_save, c_prev, c_next = st.columns(3)
if c_save.button("OK - Confirm T0 & Save"):
    st.session_state.t0_confirmed = float(st.session_state.t0_candidate)
    st.session_state.records[current["record_key"]] = _make_record()
    _save_records(out_csv, st.session_state.records)
    st.success(f"Saved to {out_csv}")

if c_prev.button("Prev", disabled=st.session_state.idx == 0):
    st.session_state.idx -= 1
    st.rerun()

if c_next.button("Save & Next", disabled=st.session_state.idx >= len(items) - 1):
    st.session_state.t0_confirmed = float(st.session_state.t0_candidate)
    st.session_state.records[current["record_key"]] = _make_record()
    _save_records(out_csv, st.session_state.records)
    st.session_state.idx += 1
    st.rerun()

st.write("---")
df_ann = _records_to_df(st.session_state.records)
if not df_ann.empty:
    cols = ["file_index", "file_name", "t0_s", "pcap", "fit_r2", "pcap_lt_pdia", "fit_ok"]
    cols = [c for c in cols if c in df_ann.columns]
    st.subheader("Saved annotations")
    st.dataframe(df_ann[cols], use_container_width=True)
    st.download_button("Download annotations CSV", df_ann.to_csv(index=False), "pcap_annotations.csv", "text/csv")
else:
    st.info("No annotations saved yet.")

st.caption(json.dumps({"annotations_csv": str(out_csv), "saved_rows": int(len(st.session_state.records))}))

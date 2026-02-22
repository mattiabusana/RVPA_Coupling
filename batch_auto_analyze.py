import argparse
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from rv_coupling import RVCouplingAnalyzer, auto_detect_cvp_rv_windows


def _read_two_sheet_excel(path: Path, sampling_rate: float):
    xl = pd.ExcelFile(path)
    if len(xl.sheet_names) < 2:
        raise ValueError("Workbook does not contain at least 2 sheets")

    df_pa = pd.read_excel(path, sheet_name=0)
    df_rv = pd.read_excel(path, sheet_name=1)
    if df_pa.shape[1] < 2 or df_rv.shape[1] < 2:
        raise ValueError("First two sheets must each contain at least 2 columns")

    t_pa = pd.to_numeric(df_pa.iloc[:, 0], errors="coerce").to_numpy()
    p_pa = pd.to_numeric(df_pa.iloc[:, 1], errors="coerce").to_numpy()
    t_rv = pd.to_numeric(df_rv.iloc[:, 0], errors="coerce").to_numpy()
    p_rv = pd.to_numeric(df_rv.iloc[:, 1], errors="coerce").to_numpy()

    m_pa = np.isfinite(t_pa) & np.isfinite(p_pa)
    m_rv = np.isfinite(t_rv) & np.isfinite(p_rv)
    t_pa, p_pa = t_pa[m_pa], p_pa[m_pa]
    t_rv, p_rv = t_rv[m_rv], p_rv[m_rv]

    if len(p_pa) < 20 or len(p_rv) < 20:
        raise ValueError("Not enough valid points after cleaning")

    # Rebuild time if timestamps are missing, non-monotonic, or not informative.
    if len(t_pa) != len(p_pa) or np.any(np.diff(t_pa) <= 0):
        t_pa = np.arange(len(p_pa)) / sampling_rate
    if len(t_rv) != len(p_rv) or np.any(np.diff(t_rv) <= 0):
        t_rv = np.arange(len(p_rv)) / sampling_rate

    if np.nanmax(t_pa) <= 0.0:
        t_pa = np.arange(len(p_pa)) / sampling_rate
    if np.nanmax(t_rv) <= 0.0:
        t_rv = np.arange(len(p_rv)) / sampling_rate

    return t_pa, p_pa, t_rv, p_rv


def _stable_mean_in_range(
    signal: np.ndarray, sampling_rate: float, window_sec: float, start_idx: int = 0, end_idx: Optional[int] = None
):
    n = len(signal)
    if end_idx is None:
        end_idx = n
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx + 1, min(end_idx, n))
    seg = signal[start_idx:end_idx]
    n_seg = len(seg)

    win = max(8, int(window_sec * sampling_rate))
    if n_seg <= win:
        return float(np.mean(seg))

    step = max(1, win // 5)
    best_score = None
    best_slice = slice(0, win)
    x = np.arange(win)

    for start in range(0, n_seg - win + 1, step):
        end = start + win
        y = seg[start:end]
        sd = float(np.std(y))
        # Low slope + low variance picks stable regions.
        slope = float(np.polyfit(x, y, 1)[0]) if win >= 3 else 0.0
        score = sd + 2.0 * abs(slope)
        if best_score is None or score < best_score:
            best_score = score
            best_slice = slice(start, end)

    return float(np.mean(seg[best_slice]))


def _auto_stable_mean(signal: np.ndarray, sampling_rate: float, window_sec: float):
    return _stable_mean_in_range(signal, sampling_rate, window_sec)


def _detect_pa_occlusion_idx(p_pa: np.ndarray, sampling_rate: float):
    n = len(p_pa)
    if n < int(8 * sampling_rate):
        return None

    win = max(5, int(0.2 * sampling_rate))
    kernel = np.ones(win) / win
    p_smooth = np.convolve(p_pa, kernel, mode="same")

    pre = int(2.5 * sampling_rate)
    post = int(2.5 * sampling_rate)
    low = pre + int(1.0 * sampling_rate)
    high = n - post - int(1.0 * sampling_rate)
    if high <= low:
        return None

    best_idx = None
    best_drop = 0.0
    for i in range(low, high):
        pre_mean = float(np.mean(p_smooth[i - pre : i - int(0.5 * sampling_rate)]))
        post_mean = float(np.mean(p_smooth[i + int(0.5 * sampling_rate) : i + post]))
        drop = post_mean - pre_mean
        if drop < best_drop:
            best_drop = drop
            best_idx = i

    # Ignore weak events that are unlikely to be true occlusions.
    if best_idx is None or best_drop > -1.5:
        return None
    return best_idx


def _auto_mpap_near_occlusion(p_pa: np.ndarray, sampling_rate: float):
    occ_idx = _detect_pa_occlusion_idx(p_pa, sampling_rate)
    if occ_idx is None:
        return _auto_stable_mean(p_pa, sampling_rate=sampling_rate, window_sec=8.0), "stable_fallback", None

    # Target pre-occlusion segment: closest stable 5 s window ending before the drop.
    pre_end = max(1, occ_idx - int(0.5 * sampling_rate))
    pre_start = max(0, occ_idx - int(12.0 * sampling_rate))
    if pre_end - pre_start < int(2.0 * sampling_rate):
        return (
            _auto_stable_mean(p_pa, sampling_rate=sampling_rate, window_sec=8.0),
            "stable_fallback_short_pre",
            occ_idx,
        )

    mpap = _stable_mean_in_range(
        p_pa,
        sampling_rate=sampling_rate,
        window_sec=5.0,
        start_idx=pre_start,
        end_idx=pre_end,
    )
    return mpap, "pre_occlusion", occ_idx


def _exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c


def _fit_pcap_at_idx(t_pa: np.ndarray, p_smooth: np.ndarray, occ_idx: int, fs: float):
    tzero = float(t_pa[occ_idx])
    fit_start = tzero + 0.3
    fit_end = tzero + 2.0
    mask = (t_pa >= fit_start) & (t_pa <= fit_end)
    if np.count_nonzero(mask) < max(20, int(0.6 * fs)):
        return {"ok": False, "reason": "Insufficient points in Pcap fit window", "r2": np.nan}

    t_fit = t_pa[mask]
    y_fit = p_smooth[mask]
    t_rel = t_fit - tzero

    pre_lo = max(0, occ_idx - int(5.0 * fs))
    pre_hi = max(pre_lo + 1, occ_idx - int(0.2 * fs))
    p_dia = float(np.percentile(p_smooth[pre_lo:pre_hi], 2))

    c0 = float(np.median(y_fit[-max(5, len(y_fit) // 5) :]))
    a0 = max(float(y_fit[0] - c0), 0.5)
    b0 = 1.0
    p0 = [a0, b0, c0]
    bounds = ([0.01, 0.05, np.min(y_fit) - 20.0], [300.0, 20.0, np.max(y_fit) + 20.0])

    try:
        popt, _ = curve_fit(_exp_decay, t_rel, y_fit, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as exc:
        return {"ok": False, "reason": f"Exp fit failed: {exc}", "r2": np.nan}

    a, b, c = [float(v) for v in popt]
    y_hat = _exp_decay(t_rel, a, b, c)
    ss_res = float(np.sum((y_fit - y_hat) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    pcap = float(_exp_decay(0.095, a, b, c))

    pre_ref_lo = max(0, occ_idx - int(1.0 * fs))
    pre_ref_hi = max(pre_ref_lo + 1, occ_idx - int(0.2 * fs))
    post_ref_lo = min(len(p_smooth) - 2, occ_idx + int(1.0 * fs))
    post_ref_hi = min(len(p_smooth) - 1, occ_idx + int(2.0 * fs))
    if post_ref_hi <= post_ref_lo:
        drop = np.nan
    else:
        drop = float(np.mean(p_smooth[post_ref_lo:post_ref_hi]) - np.mean(p_smooth[pre_ref_lo:pre_ref_hi]))

    return {"ok": True, "pcap": pcap, "r2": r2, "p_dia": p_dia, "tzero": tzero, "drop": drop}


def _fit_pcap_strict(t_pa: np.ndarray, p_pa: np.ndarray, occ_idx: int, fs: float, min_r2: float = 0.75):
    smooth_win = max(5, int(0.08 * fs))
    if smooth_win % 2 == 0:
        smooth_win += 1
    kernel = np.ones(smooth_win) / smooth_win
    p_smooth = np.convolve(p_pa, kernel, mode="same")

    # Search locally around occlusion hint and keep the best physiologically valid fit.
    search_half = int(15.0 * fs)
    start_idx = max(int(2.0 * fs), occ_idx - search_half)
    end_idx = min(len(p_pa) - int(2.5 * fs), occ_idx + search_half)
    step = max(1, int(0.2 * fs))

    best = None
    for idx in range(start_idx, end_idx, step):
        fit = _fit_pcap_at_idx(t_pa, p_smooth, idx, fs)
        if not fit.get("ok", False):
            continue
        if not np.isfinite(fit["r2"]) or fit["r2"] < min_r2:
            continue
        if not np.isfinite(fit["pcap"]) or fit["pcap"] >= fit["p_dia"]:
            continue
        if not np.isfinite(fit["drop"]) or fit["drop"] > -1.0:
            continue
        if best is None or fit["r2"] > best["r2"]:
            best = fit

    if best is None:
        return {"ok": False, "reason": "No valid Pcap fit found (R2/physiology constraints)"}
    return best


def _analyze_file(path: Path, sampling_rate: float, co_l_min: float, pcap_annotation=None):
    t_pa, p_pa, t_rv, p_rv = _read_two_sheet_excel(path, sampling_rate)

    mpap, mpap_method, occ_idx = _auto_mpap_near_occlusion(p_pa, sampling_rate=sampling_rate)
    cvp_rv_regions = auto_detect_cvp_rv_windows(t_rv, p_rv, sampling_rate=sampling_rate)
    if cvp_rv_regions is None:
        raise ValueError("Could not auto-detect RV segment on combined trace")
    rv_start_idx, rv_end_idx = cvp_rv_regions["rv_idx_window"]
    if rv_end_idx <= rv_start_idx:
        raise ValueError("Invalid auto-detected RV window")
    t_rv_seg = t_rv[rv_start_idx : rv_end_idx + 1]
    p_rv_seg = p_rv[rv_start_idx : rv_end_idx + 1]

    ann_pcap = np.nan
    ann_t0 = np.nan
    ann_wedge = np.nan
    ann_status = ""
    if pcap_annotation is not None:
        ann_pcap = float(pcap_annotation.get("pcap", np.nan))
        ann_t0 = float(pcap_annotation.get("t0_s", np.nan))
        ann_wedge = float(pcap_annotation.get("wedge", np.nan))
        ann_status = str(pcap_annotation.get("status", ""))

    analyzer = RVCouplingAnalyzer(sampling_rate=sampling_rate)
    beats = analyzer.detect_beats(p_rv_seg)
    if len(beats) < 2:
        raise ValueError("No valid RV beats detected")

    rows = []
    for i, b in enumerate(beats):
        s = int(b["start_idx"])
        p_idx = int(b["p_max_idx"])
        e = int(b["dpdt_min_idx"])
        end_idx = int(b.get("end_idx", e + int(0.1 * sampling_rate)))

        if not (0 <= s < p_idx < e < len(p_rv_seg)):
            continue

        end_idx = min(len(p_rv_seg), max(e + 1, end_idx))
        p_slice = p_rv_seg[s:end_idx]
        if len(p_slice) < 8:
            continue

        b_loc = b.copy()
        b_loc["start_idx"] = 0
        b_loc["p_max_idx"] = p_idx - s
        b_loc["dpdt_min_idx"] = e - s
        b_loc["end_idx"] = len(p_slice)

        if len(p_slice) > 2:
            grad = np.gradient(p_slice)
            lim = b_loc["p_max_idx"]
            b_loc["dpdt_max_idx"] = int(np.argmax(grad[:lim])) if lim > 1 else 0

        if i < len(beats) - 1:
            next_start = int(beats[i + 1]["start_idx"])
            dur = (next_start - s) / sampling_rate
        else:
            dur = max(0.2, (e - s) / sampling_rate * 2.5)

        hr = 60.0 / dur if dur > 0 else np.nan
        sv = (co_l_min * 1000.0) / hr if np.isfinite(hr) and hr > 0 else np.nan
        pes = float(p_rv_seg[e])

        pmax_iso = analyzer.estimate_pmax_isovolumic(p_slice, b_loc)
        ees = (pmax_iso - pes) / sv if np.isfinite(sv) and sv > 0 else np.nan
        ea = pes / sv if np.isfinite(sv) and sv > 0 else np.nan
        zc, lam = analyzer.calculate_zc_lambda(p_slice, b_loc, co_l_min=co_l_min, m_pap=mpap)

        if not np.isfinite(ees) or not np.isfinite(ea):
            continue

        rows.append(
            {
                "beat": i + 1,
                "time_s": float(t_rv_seg[s]),
                "hr_bpm": float(hr),
                "pes_mmHg": pes,
                "ees": float(ees),
                "ea": float(ea),
                "ees_ea": float(ees / ea) if np.isfinite(ea) and ea > 0 else np.nan,
                "zc": float(zc) if np.isfinite(zc) else np.nan,
                "lambda": float(lam) if np.isfinite(lam) else np.nan,
                "mpap": float(mpap),
                "mpap_method": mpap_method,
                "occlusion_idx_auto": int(occ_idx) if occ_idx is not None else -1,
                "t0_annotated_s": ann_t0,
                "pcap_annotated": ann_pcap,
                "wedge_annotated": ann_wedge,
                "pcap_status": ann_status,
                "rv_start_s": float(cvp_rv_regions["rv_time_window"][0]),
                "rv_end_s": float(cvp_rv_regions["rv_time_window"][1]),
                "co_l_min": float(co_l_min),
            }
        )

    if not rows:
        raise ValueError("No analyzable beats after quality checks")

    df = pd.DataFrame(rows)
    return df


def _store_failed_file(src: Path, input_root: Path, failed_root: Path, move_failed: bool):
    try:
        rel = src.relative_to(input_root)
    except ValueError:
        rel = Path(src.name)
    dst = failed_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move_failed:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)
    return dst


def _iter_xlsx(input_root: Path, include_separati: bool):
    for p in sorted(input_root.rglob("*.xlsx")):
        if not include_separati and "separati" in p.parts:
            continue
        stem_low = p.stem.lower()
        if "dataset" in stem_low and "experiment" in stem_low:
            continue
        yield p


def _is_approved_status(val: str):
    s = str(val).strip().lower()
    return s in {"approved", "ok", "confirmed", "done", "true", "yes", "1"}


def _load_annotations_map(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "file" not in df.columns:
        raise ValueError("Annotations CSV must contain a 'file' column")

    amap = {}
    for _, row in df.iterrows():
        if "status" in df.columns and not _is_approved_status(row.get("status", "")):
            continue
        row_dict = row.to_dict()
        file_val = str(row_dict.get("file", "")).strip()
        if not file_val:
            continue
        amap[file_val] = row_dict
        amap[str(Path(file_val).name)] = row_dict
        abs_file = str(row_dict.get("absolute_file", "")).strip()
        if abs_file:
            amap[abs_file] = row_dict
    return amap


def _read_tabular_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported CO table format: {path.suffix} (use .csv or .xlsx)")


def _find_first_column(df: pd.DataFrame, candidates, label: str):
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    raise ValueError(f"CO table must contain a {label} column (accepted: {', '.join(candidates)})")


def _load_co_map(table_path: Path):
    df = _read_tabular_file(table_path)
    if df.empty:
        return {}

    file_col = _find_first_column(
        df,
        candidates=[
            "file",
            "file_name",
            "filename",
            "path",
            "relative_file",
            "absolute_file",
        ],
        label="file/path",
    )
    co_col = _find_first_column(
        df,
        candidates=[
            "co",
            "co_l_min",
            "co_lmin",
            "co_lpm",
            "cardiac_output",
            "cardiac_output_l_min",
            "cardiac_output_lmin",
        ],
        label="cardiac output",
    )

    cmap = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        file_val = str(row_dict.get(file_col, "")).strip()
        if not file_val:
            continue

        co_val = pd.to_numeric(pd.Series([row_dict.get(co_col)]), errors="coerce").iloc[0]
        if not np.isfinite(co_val):
            continue

        entry = dict(row_dict)
        entry["_co_l_min"] = float(co_val)

        cmap[file_val] = entry
        cmap[str(Path(file_val).name)] = entry
        abs_file = str(row_dict.get("absolute_file", "")).strip()
        if abs_file:
            cmap[abs_file] = entry

    return cmap


def main():
    parser = argparse.ArgumentParser(description="Automatic overnight RV-PA batch analysis.")
    parser.add_argument("--input-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("batch_results"))
    parser.add_argument("--sampling-rate", type=float, default=120.0)
    parser.add_argument("--co", type=float, default=5.0, help="Deprecated fallback CO in L/min (ignored: per-file CO table is required)")
    parser.add_argument("--include-separati", action="store_true")
    parser.add_argument("--co-table", type=Path, default=None, help="CSV/XLSX with per-file cardiac output values")
    parser.add_argument(
        "--require-co-table-match",
        action="store_true",
        help="Deprecated (now default behavior): fail files without a matching per-file CO",
    )
    parser.add_argument("--pcap-annotations", type=Path, default=None, help="CSV from pcap_annotator_app.py")
    parser.add_argument(
        "--require-annotations",
        action="store_true",
        help="Fail files without an approved annotation when --pcap-annotations is provided",
    )
    parser.add_argument(
        "--move-failed",
        action="store_true",
        help="Move (instead of copy) failed files into failed_files folder",
    )
    parser.add_argument("--max-files", type=int, default=0, help="For dry tests; 0 means all files")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    success_root = output_root / "success"
    failed_root = output_root / "failed_files"
    success_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    failed_rows = []
    missing_co_rows = []

    files = list(_iter_xlsx(input_root, args.include_separati))
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit("No .xlsx files found to process")

    annotations_map = None
    if args.pcap_annotations is not None:
        ann_path = args.pcap_annotations.resolve()
        if not ann_path.exists():
            raise SystemExit(f"Annotations CSV not found: {ann_path}")
        annotations_map = _load_annotations_map(ann_path)

    co_map = None
    if args.co_table is not None:
        co_path = args.co_table.resolve()
        if not co_path.exists():
            raise SystemExit(f"CO table not found: {co_path}")
        co_map = _load_co_map(co_path)

    for f in files:
        try:
            annotation = None
            if annotations_map is not None:
                rel_key = str(f.relative_to(input_root).as_posix())
                abs_key = str(f.resolve())
                base_key = f.name
                annotation = annotations_map.get(rel_key) or annotations_map.get(abs_key) or annotations_map.get(base_key)
                if annotation is None and args.require_annotations:
                    raise ValueError("Missing approved annotation")

            if co_map is None:
                raise ValueError("Missing CO table (--co-table required; no fallback)")

            rel_key = str(f.relative_to(input_root).as_posix())
            abs_key = str(f.resolve())
            base_key = f.name
            co_row = co_map.get(rel_key) or co_map.get(abs_key) or co_map.get(base_key)
            if co_row is None:
                raise ValueError("Missing CO table match")

            co_l_min = float(co_row["_co_l_min"])
            co_source = "co_table"

            df = _analyze_file(f, sampling_rate=args.sampling_rate, co_l_min=co_l_min, pcap_annotation=annotation)
            df["co_source"] = co_source
            out_name = f"{f.stem}_beats.csv"
            out_csv = success_root / out_name
            df.to_csv(out_csv, index=False)

            tail = df.tail(min(30, len(df)))
            summary_rows.append(
                {
                    "file": str(f),
                    "beats_total": int(len(df)),
                    "ees_mean": float(tail["ees"].mean()),
                    "ea_mean": float(tail["ea"].mean()),
                    "ees_ea_mean": float(tail["ees_ea"].mean()),
                    "zc_mean": float(tail["zc"].mean()),
                    "lambda_mean": float(tail["lambda"].mean()),
                    "mpap": float(tail["mpap"].mean()),
                    "pcap_annotated_mean": float(tail["pcap_annotated"].mean()),
                    "t0_annotated_s_mean": float(tail["t0_annotated_s"].mean()),
                    "co_l_min": float(co_l_min),
                    "co_source": co_source,
                    "output_csv": str(out_csv),
                }
            )
            print(f"[OK] {f}")
        except Exception as exc:
            err_msg = str(exc)
            copied_to = _store_failed_file(
                f,
                input_root=input_root,
                failed_root=failed_root,
                move_failed=args.move_failed,
            )
            failed_rows.append({"file": str(f), "reason": err_msg, "copied_to": str(copied_to)})
            if "CO table" in err_msg:
                missing_co_rows.append({"file": str(f), "reason": err_msg})
            print(f"[FAIL] {f} -> {exc}")

    summary_csv = output_root / "summary.csv"
    failed_csv = output_root / "failed.csv"
    missing_co_csv = output_root / "missing_co.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)
    pd.DataFrame(missing_co_rows).to_csv(missing_co_csv, index=False)

    print("\nBatch completed")
    print(f"Input files scanned: {len(files)}")
    print(f"Successful analyses: {len(summary_rows)}")
    print(f"Failed analyses: {len(failed_rows)}")
    print(f"Missing CO table entries: {len(missing_co_rows)}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Failed CSV: {failed_csv}")
    print(f"Missing CO CSV: {missing_co_csv}")
    print(f"Failed files folder: {failed_root}")


if __name__ == "__main__":
    main()

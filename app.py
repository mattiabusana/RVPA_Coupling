import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.signal as signal
import importlib
import rv_coupling
importlib.reload(rv_coupling)
from rv_coupling import RVCouplingAnalyzer, compute_pcap, compute_wedge_empirical

# --- Configuration & State ---
st.set_page_config(page_title="RV-PA Coupling Wizard", layout="wide")

if 'step' not in st.session_state:
    st.session_state.step = 1

if 'data' not in st.session_state:
    st.session_state.data = {} 

def reset_analysis():
    st.session_state.step = 1
    st.session_state.data = {}

st.title("🫀 RV-PA Coupling Wizard")

# --- Sidebar ---
st.sidebar.header("Settings")
sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", value=120.0, min_value=1.0)
# Step 2 now handles CO, but we keep this as a fallback or global setting
co_sidebar = st.sidebar.number_input("Global Cardiac Output (Default)", value=5.0, min_value=0.1)

if st.sidebar.button("Reset Analysis"):
    reset_analysis()
    st.rerun()

# --- Helper Functions ---
@st.cache_data
def load_excel_data(file):
    try:
        xl = pd.ExcelFile(file)
        
        # Sheet 1 -> PA
        df_pa = pd.read_excel(file, sheet_name=0)
        t_pa = pd.to_numeric(df_pa.iloc[:, 0], errors='coerce').values
        p_pa = pd.to_numeric(df_pa.iloc[:, 1], errors='coerce').values
        
        # Sheet 2 -> CVP+RV
        df_rv = pd.read_excel(file, sheet_name=1)
        t_rv = pd.to_numeric(df_rv.iloc[:, 0], errors='coerce').values
        p_rv = pd.to_numeric(df_rv.iloc[:, 1], errors='coerce').values
        
        # Clean NaNs
        mask_pa = ~np.isnan(t_pa) & ~np.isnan(p_pa)
        t_pa, p_pa = t_pa[mask_pa], p_pa[mask_pa]
        
        mask_rv = ~np.isnan(t_rv) & ~np.isnan(p_rv)
        t_rv, p_rv = t_rv[mask_rv], p_rv[mask_rv]
        
        return t_pa, p_pa, t_rv, p_rv
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, None


def plot_full_curve_selector(time, signal, title, key_prefix, selection_mode="box"):
    """
    Plots a large signal with downsampling and handles selection.
    """
    # Ensure session state keys exist for inputs
    k_start = f"{key_prefix}_start"
    k_end = f"{key_prefix}_end"
    
    min_t, max_t = float(time[0]), float(time[-1])
    
    # Validation
    if k_start in st.session_state:
        if st.session_state[k_start] < min_t or st.session_state[k_start] > max_t:
             st.session_state[k_start] = min_t
             
    if k_end in st.session_state:
        if st.session_state[k_end] < min_t or st.session_state[k_end] > max_t:
             st.session_state[k_end] = min(max_t, min_t + 5.0)
    
    if k_start not in st.session_state:
        st.session_state[k_start] = min_t
    if k_end not in st.session_state:
        st.session_state[k_end] = min(max_t, min_t + 5.0)

    # 1. OPTIMIZATION: DOWNSAMPLE
    target_points = 5000
    step = 1
    if len(time) > target_points:
        step = int(len(time) / target_points)
    
    t_plot = time[::step]
    s_plot = signal[::step]
    
    if step > 1:
        st.caption(f"⚡ Optimized View: Displaying 1 out of {step} points.")

    # 2. PLOT
    fig = go.Figure()
    
    # Selection Mode Logic
    if selection_mode == "points":
        # Robust Click: Use standard Scatter (not GL) and show markers
        fig.add_trace(go.Scatter(
            x=t_plot, y=s_plot, 
            line=dict(color='blue', width=1),
            marker=dict(size=3, color='rgba(0,0,255,0.5)', line=dict(width=1, color='blue')), 
            name="Signal", 
            mode='lines+markers'
        ))
        drag_mode = "zoom" 
        click_mode = "event+select"
    else:
        # Performance: Use Scattergl for large data box selection
        fig.add_trace(go.Scattergl(
            x=t_plot, y=s_plot, 
            line=dict(color='blue', width=1), 
            name="Signal", 
            mode='lines' 
        ))
        drag_mode = "select"
        click_mode = "event"

    fig.update_layout(
        title=title, 
        xaxis_title="Time (s)", 
        yaxis_title="Pressure", 
        margin=dict(l=0,r=0,t=40,b=0),
        dragmode=drag_mode,
        clickmode=click_mode,
        showlegend=False,
        hovermode='closest' if selection_mode == "points" else False
    )
    
    # 3. INTERACTION
    mode_arg = [selection_mode]
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=mode_arg, key=f"{key_prefix}_plot")
    
    # 4. UPDATE STATE
    if event and "selection" in event:
        selection = event["selection"]
        new_s = None
        
        if selection_mode == "points":
            if "points" in selection and selection["points"]:
                pt = selection["points"][0]
                new_s = pt["x"]
        
        elif selection_mode == "box":
            if "box" in selection and selection["box"]:
                box = selection["box"][0] 
                if "x" in box:
                    new_s = box["x"][0]
                    new_e = box["x"][1]
                    if new_s > new_e: new_s, new_e = new_e, new_s
                    st.session_state[k_end] = float(new_e)
            elif "points" in selection and selection["points"]:
                 points = selection["points"]
                 xs = [p["x"] for p in points]
                 if xs:
                     new_s = float(min(xs))
                     new_e = float(max(xs))
                     st.session_state[k_end] = float(new_e)

        if new_s is not None:
            st.session_state[k_start] = float(new_s)
            if selection_mode == "points":
                 st.session_state[k_end] = min(max_t, float(new_s) + 5.0)
    
    # Widgets
    c1, c2 = st.columns(2)
    with c1:
        start_sel = st.number_input(f"{title} Start (s)", min_value=min_t, max_value=max_t, key=k_start)
    with c2:
        end_sel = st.number_input(f"{title} End (s)", min_value=min_t, max_value=max_t, key=k_end)
        
    return start_sel, end_sel


# ==============================================================================
# STEP 1: PA ANALYSIS (mPAP + Sys/Dia)
# ==============================================================================
if st.session_state.step == 1:
    st.header("Step 1: Pulmonary Artery Analysis - Hemodynamics")
    uploaded = st.file_uploader("Upload Data File", type=["xlsx"])
    
    if uploaded:
        t_pa, p_pa, t_rv, p_rv = load_excel_data(uploaded)
        
        if t_pa is not None:
            # Store full datasets
            # Parse Metadata from Filename
            fname = uploaded.name
            # Heuristic Parsing: Assume "PigID_Step_..." or "PigID Step ..."
            # Remove extension
            fname_clean = fname.rsplit('.', 1)[0]
            parts = fname_clean.replace('-', '_').replace(' ', '_').split('_')
            
            if len(parts) >= 2:
                pig_id = parts[0]
                step_id = "_".join(parts[1:])
            else:
                pig_id = fname_clean
                step_id = "Baseline" # Default
            
            st.session_state.data = {
                't_pa': t_pa, 'p_pa': p_pa,
                't_rv_full': t_rv, 'p_rv_full': p_rv,
                'pig_id': pig_id, 'step_id': step_id, 'filename': fname
            }
            
            st.caption(f"**Detected Metadata**: Pig ID=`{pig_id}`, Step=`{step_id}`")
            
            st.info("Select a stable window to calculate PAP metrics (Smoothed).")
            s_pa, e_pa = plot_full_curve_selector(t_pa, p_pa, "PA Signal (Sheet 1)", "pa")
            
            mask_zoom = (t_pa >= s_pa) & (t_pa <= e_pa)
            if np.any(mask_zoom):
                p_win_raw = p_pa[mask_zoom]
                t_win = t_pa[mask_zoom]
                
                # Apply Smoothing (Moving Average, N=20)
                win_size = 20
                if len(p_win_raw) >= win_size:
                    kernel = np.ones(win_size) / win_size
                    # Use 'valid' or 'same'. 'same' keeps length but boundary effects.
                    # Given selection, let's use 'same' but be aware of edges.
                    p_win_smooth = np.convolve(p_win_raw, kernel, mode='same')
                    # Trim edges to avoid artifact
                    trim = win_size // 2
                    p_calc = p_win_smooth[trim:-trim] 
                    t_calc = t_win[trim:-trim]
                else:
                    p_calc = p_win_raw
                    t_calc = t_win
                
                mpap = np.mean(p_calc)
                # Robust min/max using percentiles on SMOOTHED data
                p_sys = np.percentile(p_calc, 98) 
                p_dia = np.percentile(p_calc, 2)
                
                # --- AUTO HR CALCULATION ---
                # Find peaks in smoothed PA signal to estimate HR
                # Approx distance: 0.2s (Max 300 bpm), height > mean
                peaks, _ = signal.find_peaks(p_calc, distance=sampling_rate*0.2, height=mpap)
                if len(peaks) > 1:
                    peak_indices = peaks
                    peak_times = t_calc[peak_indices]
                    intervals = np.diff(peak_times)
                    mean_interval = np.mean(intervals)
                    hr_calc = 60.0 / mean_interval if mean_interval > 0 else 0
                else:
                    hr_calc = 0.0

                with st.expander("Selection Preview (Smoothing Applied)", expanded=True):
                    fig_z = go.Figure()
                    # Raw (Faint)
                    fig_z.add_trace(go.Scatter(x=t_win, y=p_win_raw, line=dict(color='lightgrey', width=1), name="Raw"))
                    # Smoothed (Bold)
                    fig_z.add_trace(go.Scatter(x=t_calc, y=p_calc, line=dict(color='green', width=2), name="Smoothed"))
                    # Peaks
                    if len(peaks) > 0:
                         fig_z.add_trace(go.Scatter(x=t_calc[peaks], y=p_calc[peaks], mode='markers', marker=dict(color='red', size=8), name='Peaks'))
                    
                    fig_z.add_hline(y=mpap, line_dash="dash", annotation_text=f"Mean: {mpap:.1f}")
                    fig_z.add_hline(y=p_sys, line_dash="dot", line_color="red", annotation_text=f"Sys: {p_sys:.1f}")
                    fig_z.add_hline(y=p_dia, line_dash="dot", line_color="blue", annotation_text=f"Dia: {p_dia:.1f}")
                    fig_z.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig_z, use_container_width=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Systolic", f"{p_sys:.1f}")
                c2.metric("Diastolic", f"{p_dia:.1f}")
                c3.metric("Mean", f"{mpap:.1f}")
                c4.metric("Est. Rate", f"{hr_calc:.0f} bpm")
                
                # NAVIGATION: Step 1 -> Step 2 (Wedge)
                if st.button("Confirm PAP Analysis & Next"):
                    st.session_state.data['mpap'] = mpap
                    st.session_state.data['p_sys'] = p_sys
                    st.session_state.data['p_dia'] = p_dia
                    st.session_state.data['hr_calc'] = hr_calc # Pass calculated HR
                    st.session_state.step = 2
                    st.rerun()

# ==============================================================================
# STEP 2: PA WEDGE & HEMODYNAMICS
# ==============================================================================
elif st.session_state.step == 2:
    st.header("Step 2: PA Wedge & Hemodynamic Profile")
    
    t_pa = st.session_state.data['t_pa']
    p_pa = st.session_state.data['p_pa']
    
    if 'pcap_zoom_window' not in st.session_state:
        st.session_state.pcap_zoom_window = None

    # --- STAGE 1: ZOOM SELECTION ---
    if st.session_state.pcap_zoom_window is None:
        st.info("Phase 1: **ZOOM**. Draw a BOX around the full occlusion maneuver.")
        
        s_zoom, e_zoom = plot_full_curve_selector(t_pa, p_pa, "Full PA Signal", "pcap_stage1", selection_mode="box")
        
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("Back"):
                st.session_state.step = 1
                st.rerun()
        with c2:
            if st.button("Confirm Zoom & Proceed"):
                if e_zoom <= s_zoom:
                    st.error("Invalid selection.")
                else:
                    st.session_state.pcap_zoom_window = (s_zoom, e_zoom)
                    st.rerun()

    # --- STAGE 2: PINPOINT TZERO ---
    else:
        st.info("Phase 2: **PINPOINT**. Click on the plot to set the exact **Start of Occlusion (tzero)**.")
        
        s_base, e_base = st.session_state.pcap_zoom_window
        mask_d = (t_pa >= s_base) & (t_pa <= e_base)
        
        if not np.any(mask_d):
             st.error("No data in selected range.")
             if st.button("Reset"):
                 st.session_state.pcap_zoom_window = None
                 st.rerun()
        else:
            t_view = t_pa[mask_d]
            p_view = p_pa[mask_d]
            
            dyn_key = f"pcap_stage2_{int(s_base)}"
            master_key_start = f"{dyn_key}_start"
            
            if 'pending_ts_update' in st.session_state:
                update_val = st.session_state.pop('pending_ts_update')
                st.session_state[master_key_start] = float(update_val)
            
            s_click, _ = plot_full_curve_selector(t_view, p_view, "Zoomed Occlusion", dyn_key, selection_mode="points")
            tzero = s_click
            
            # CALCULATION
            fit_res = compute_pcap(t_pa, p_pa, tzero_time=tzero)
            wedge_val = compute_wedge_empirical(t_pa, p_pa, tzero_time=tzero, t_end=e_base, n_points=20)
            
            final_pcap = 0.0
            r2 = 0.0
            b = 0.0
            
            if fit_res['fit_success']:
                pcap_calc = fit_res['pcap']
                r2 = fit_res['r_squared']
                a, b, c = fit_res['params']
                final_pcap = pcap_calc
                
                # Plot
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=t_view, y=p_view, mode='lines', name='Raw Signal', line=dict(color='lightgrey', width=1)))
                
                if 'pap_smooth' in fit_res:
                     full_smooth = fit_res['pap_smooth']
                     p_view_smooth = full_smooth[mask_d]
                     fig_z.add_trace(go.Scatter(x=t_view, y=p_view_smooth, mode='lines', name='Smoothed', line=dict(color='blue', width=1)))
                
                t_fit_abs = tzero + fit_res['time_rel']
                p_fit_curve = a * np.exp(-b * fit_res['time_rel']) + c
                fig_z.add_trace(go.Scatter(x=t_fit_abs, y=p_fit_curve, mode='lines', name='Exp Fit', line=dict(color='red', width=3)))
                
                fig_z.add_hline(y=wedge_val, line_dash="dot", line_color="orange", annotation_text="Pwedge")
                fig_z.add_vline(x=tzero, line_dash="dash", line_color="green", annotation_text="tzero")
                fig_z.add_trace(go.Scatter(x=[tzero + 0.095], y=[pcap_calc], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Pcap (95ms)'))

                fig_z.update_layout(title="Interactive Fit", height=450, margin=dict(l=0,r=0,t=40,b=0))
                
                pres_key = f"{dyn_key}_result_plot"
                event_res = st.plotly_chart(fig_z, use_container_width=True, on_select="rerun", selection_mode="points", key=pres_key)
                
                if event_res and "selection" in event_res:
                    sel = event_res["selection"]
                    if "points" in sel and sel["points"]:
                        pt = sel["points"][0]
                        new_tzero = pt["x"]
                        if st.session_state.get(master_key_start) != new_tzero:
                            st.session_state['pending_ts_update'] = new_tzero
                            st.rerun()

            else:
                st.warning("Fit failed.")
                final_pcap = wedge_val

            # Manual Override
            if st.checkbox("Override Pcap?", key="ovr_pcap"):
                 final_pcap = st.number_input("Manual Value", value=float(final_pcap))
            
            # --- HEMODYNAMICS ---
            st.write("---")
            c_res, c_co = st.columns([2, 1])
            with c_res:
                 st.subheader("Wedge Fit Results")
                 c1, c2, c3 = st.columns(3)
                 c1.metric("Pcap (Fit)", f"{final_pcap:.2f}")
                 c2.metric("Wedge (Empirical)", f"{wedge_val:.2f}")
                 c3.metric("R²", f"{r2:.3f}")
            with c_co:
                st.subheader("Hemodynamics Input")

                
                # Determine default HR: prefer calculated from Step 1, else existing state, else 60
                calc_hr = st.session_state.data.get('hr_calc', 60.0)
                # If we just arrived from Step 1, we might want to update the default
                # But we don't want to overwrite user manual input if they stay on Step 2
                # Simple logic: Use calc_hr as the value. If user changes it, session_state['hr_val'] updates?
                # Actually, st.number_input uses 'value' as initial.
                
                co_val = st.number_input("Cardiac Output (L/min)", value=st.session_state.get('co_val', co_sidebar), step=0.1)
                
                # We use 'value' = calc_hr. Note: if user previously edited 'hr_val', we might want to respect that?
                # User request: "must calculate it". So let's prioritize the calculated value from Step 1.
                # If they re-run Step 1, it updates.
                hr_val = st.number_input("Heart Rate (bpm)", value=float(calc_hr), step=1.0)
                
                st.session_state['co_val'] = co_val
                st.session_state['hr_val'] = hr_val

            # Table
            if 'mpap' in st.session_state.data:
                mpap = st.session_state.data['mpap']
                p_sys = st.session_state.data.get('p_sys', 0)
                p_dia = st.session_state.data.get('p_dia', 0)
                
                # PVR = (mPAP - Wedge) / CO
                rv_loading_grad = (mpap - wedge_val)
                pvr = rv_loading_grad / co_val if co_val > 0 else 0
                
                # PAC (Compliance) = SV / Pulse Pressure
                # SV = (CO * 1000) / HR
                sv_calc = (co_val * 1000) / hr_val if hr_val > 0 else 0
                pulse_pressure = p_sys - p_dia
                pac_est = sv_calc / pulse_pressure if pulse_pressure > 0 else 0
                
                st.markdown("### 🫀 Hemodynamic Profile")
                hemo_data = {
                    "Metric": [
                        "Systolic PAP", "Diastolic PAP", "Mean PAP", 
                        "Wedge Pressure (P_awp)", "Capillary Pressure (Pcap)", 
                        "Cardiac Output", "Heart Rate", "Stroke Volume", 
                        "PVR (Resistance)", "PAC (Compliance)", "RC Time (fit)"
                    ],
                    "Value": [
                        f"{p_sys:.1f}", f"{p_dia:.1f}", f"{mpap:.1f}", 
                        f"{wedge_val:.1f}", f"{final_pcap:.1f}", 
                        f"{co_val:.1f}", f"{hr_val:.0f}", f"{sv_calc:.1f}", 
                        f"{pvr:.2f}", f"{pac_est:.2f}", f"{1/b if b>0 else 0:.3f}"
                    ],
                    "Unit": [
                        "mmHg", "mmHg", "mmHg", 
                        "mmHg", "mmHg", 
                        "L/min", "bpm", "mL", 
                        "WU", "mL/mmHg", "s"
                    ]
                }
                st.table(pd.DataFrame(hemo_data))

            # NAVIGATION Step 2 -> Step 3
            c1, c2 = st.columns([1, 5])
            with c1:
                if st.button("Reset Zoom", key="rst_zm"):
                    st.session_state.pcap_zoom_window = None
                    st.rerun()
            with c2:
                if st.button("Confirm & Next (Save Pcap)"):
                    st.session_state.data['mpawp'] = final_pcap # Save Pcap as the reference for Ees calc? Or Wedge?
                    # Usually Ees uses Pes. Ea uses Pes/SV.
                    # Zc/Lambda might use mPAP.
                    # Wedge is useful for PVR. 
                    # Let's save both?
                    st.session_state.data['p_wedge_empirical'] = wedge_val
                    st.session_state.data['co_val'] = co_val
                    st.session_state.step = 3
                    st.rerun()

# ==============================================================================
# STEP 3: RV BEAT SELECTION
# ==============================================================================
elif st.session_state.step == 3:
    st.header("Step 3: RV Beat Selection")
    
    t_full = st.session_state.data['t_rv_full']
    p_full = st.session_state.data['p_rv_full']
    
    st.info("Select the segment containing the RV beats to analyze.")
    s_rv, e_rv = plot_full_curve_selector(t_full, p_full, "RV Signal (Sheet 2)", "rv_beats")
    
    mask_rv = (t_full >= s_rv) & (t_full <= e_rv)
    if np.any(mask_rv):
        with st.expander("Selection Preview", expanded=True):
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=t_full[mask_rv], y=p_full[mask_rv], line=dict(color='red')))
            fig_z.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_z, use_container_width=True)
        
        c1, c2 = st.columns([1, 5])
        with c1:
             if st.button("Back"):
                st.session_state.step = 2
                st.rerun()
        with c2:
            if st.button("Extract Beats & Analyze"):
                p_seg = p_full[mask_rv]
                t_seg = t_full[mask_rv]
                
                analyzer = RVCouplingAnalyzer(sampling_rate=sampling_rate)
                beats = analyzer.detect_beats(p_seg)
                
                if len(beats) > 0:
                    st.session_state.data['beats'] = beats
                    st.session_state.data['p_rv_segment'] = p_seg
                    st.session_state.data['t_rv_segment'] = t_seg
                    st.session_state.step = 4
                    st.rerun()
                else:
                    st.error("No beats detected.")

# ==============================================================================
# STEP 4: INTERACTIVE RESULTS
# ==============================================================================
elif st.session_state.step == 4:
    st.header("Step 4: Interactive Analysis")
    
    beats = st.session_state.data['beats']
    p_seg = st.session_state.data['p_rv_segment']
    t_seg = st.session_state.data['t_rv_segment']
    mpap = st.session_state.data['mpap']
    mpawp = st.session_state.data.get('mpawp', 0.0)
    co_default = st.session_state.data.get('co_val', 5.0)
    
    analyzer = RVCouplingAnalyzer(sampling_rate=sampling_rate)
    
    glob_c1, glob_c2 = st.columns(2)
    with glob_c1:
        b_idx = st.selectbox("Select Beat", range(len(beats)), format_func=lambda i: f"Beat {i+1}")
    with glob_c2:
        co_input_final = st.number_input("Cardiac Output (L/min)", value=co_default)
    
    beat_data = beats[b_idx]
    
    col_adj, col_viz = st.columns([1, 2])
    with col_adj:
        st.subheader("Fine-tune Beat")
        s_loc, p_loc, e_loc = beat_data['start_idx'], beat_data['p_max_idx'], beat_data['dpdt_min_idx']
        
        new_s = st.slider("Start", max(0, s_loc-50), min(len(p_seg), s_loc+50), s_loc, key="s_slide")
        new_p = st.slider("Peak", new_s, min(len(p_seg), e_loc), p_loc, key="p_slide")
        new_e = st.slider("End", new_p, min(len(p_seg)-1, e_loc+150), e_loc, key="e_slide")
        
        beat_data.update({'start_idx': new_s, 'p_max_idx': new_p, 'dpdt_min_idx': new_e, 'end_idx': new_e + int(0.1*sampling_rate)})
        
        p_b_slice = p_seg[new_s : beat_data['end_idx']]
        b_calc = beat_data.copy()
        b_calc.update({'start_idx': 0, 'p_max_idx': new_p - new_s, 'dpdt_min_idx': new_e - new_s, 'end_idx': len(p_b_slice)})
        
        if len(p_b_slice) > 2:
            grad = np.gradient(p_b_slice)
            sl = b_calc['p_max_idx']
            b_calc['dpdt_max_idx'] = np.argmax(grad[:sl]) if sl > 1 else 0
        
        zc, lam = analyzer.calculate_zc_lambda(p_b_slice, b_calc, co_l_min=co_input_final, m_pap=mpap)
        
        # Ea
        if b_idx < len(beats)-1:
            dur = (beats[b_idx+1]['start_idx'] - new_s) / sampling_rate
            hr = 60/dur if dur > 0 else 0
        else:
            hr = 60 / ((new_e - new_s)/sampling_rate * 2.5)
            
        sv = (co_input_final * 1000) / hr if hr > 0 else 0
        pes = p_seg[new_e]
        ea = pes / sv if sv > 0 else 0
        
        # Ees
        p_max_iso = analyzer.estimate_pmax_isovolumic(p_b_slice, b_calc)
        ees = (p_max_iso - pes) / sv if sv > 0 else 0
        
        st.markdown(f"**Ees:** {ees:.2f}")
        st.markdown(f"**Ea:** {ea:.2f}")
        st.markdown(f"**Ea/Ees (Coupling):** {ea/ees:.2f}" if ees > 0 else "N/A")
        
    with col_viz:
        st.subheader("Visualization")
        t_vis = t_seg[new_s : beat_data['end_idx']]
        p_vis = p_seg[new_s : beat_data['end_idx']]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_vis, y=p_vis, name="RV", line=dict(color='black')))
        
        # Markers
        # Start (Green)
        fig.add_trace(go.Scatter(x=[t_seg[new_s]], y=[p_seg[new_s]], mode="markers", marker=dict(color='green', size=12), name="Start"))
        # Peak (Red)
        fig.add_trace(go.Scatter(x=[t_seg[new_p]], y=[p_seg[new_p]], mode="markers", marker=dict(color='red', size=12), name="Peak"))
        # End Systole (Blue)
        fig.add_trace(go.Scatter(x=[t_seg[new_e]], y=[p_seg[new_e]], mode="markers", marker=dict(color='blue', size=12), name="End Systole"))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # --- RESULTS & EXPORT ---
    st.divider()
    st.subheader("Global Results")
    
    # Calculate All Beats
    batch = []
    for i, b in enumerate(beats):
        s, e = b['start_idx'], b['dpdt_min_idx']
        dur = (beats[i+1]['start_idx'] - s)/sampling_rate if i < len(beats)-1 else (e-s)/sampling_rate * 2.5
        hr_i = 60/dur if dur > 0 else 0
        sv_i = (co_input_final * 1000) / hr_i if hr_i > 0 else 0
        
        b_l = b.copy()
        b_l.update({'start_idx':0, 'p_max_idx': b['p_max_idx']-s, 'dpdt_min_idx': b['dpdt_min_idx']-s})
        p_sl = p_seg[s:b['end_idx']]
        b_l['end_idx'] = len(p_sl)
        
        # Recalc Relative dP/dt max
        if len(p_sl) > 2:
            gr = np.gradient(p_sl)
            limit = b_l['p_max_idx']
            b_l['dpdt_max_idx'] = np.argmax(gr[:limit]) if limit > 1 else 0
        
        pm_iso = analyzer.estimate_pmax_isovolumic(p_sl, b_l)
        pes_i = p_seg[e]
        ees_i = (pm_iso - pes_i)/sv_i if sv_i > 0 else 0
        ea_i = pes_i/sv_i if sv_i > 0 else 0
        zc_i, lam_i = analyzer.calculate_zc_lambda(p_sl, b_l, co_input_final, mpap)
        z0_i = mpap / co_input_final if co_input_final > 0 else 0
        
        batch.append({
            "Beat": i+1, "Time": t_seg[s], "HR": hr_i, "Ees": ees_i, "Ea": ea_i, "Ea/Ees": ea_i/ees_i if ees_i else 0,
            "Lambda": lam_i, "Zc": zc_i, "Z0": z0_i, "mPAP": mpap, "Wedge": mpawp
        })
        
    df_all = pd.DataFrame(batch)
    
    # Calculate Statistics (Last 30)
    n_an = min(len(df_all), 30)
    df_stat = df_all.iloc[-n_an:]
    stats_rows = []
    # Metrics to summarize
    metrics_to_show = ["Ees", "Ea", "Ea/Ees", "Lambda", "Zc", "Z0"]
    
    for c in metrics_to_show:
        if c in df_stat.columns:
            s = df_stat[c]
            # UI Dict (Strings)
            stats_rows.append({
                "id": st.session_state.data.get('pig_id', 'Unknown'),
                "measure": st.session_state.data.get('step_id', 'Unknown'),
                "Variable": c, 
                "Mean": f"{s.mean():.2f}", 
                "SD": f"{s.std():.2f}", 
                "Min": f"{s.min():.2f}", 
                "Max": f"{s.max():.2f}",
                "Range": f"{s.max()-s.min():.2f}"
            })
            
    df_summary = pd.DataFrame(stats_rows)
    
    st.markdown(f"**Average Values (Last {n_an} Beats)**")
    st.dataframe(df_summary) # scalable
    
    # CSV Generation (Summary Only - WIDE Format)
    # Pivot logic: One row per session
    wide_row = {
        "id": st.session_state.data.get('pig_id', 'Unknown'),
        "measure": st.session_state.data.get('step_id', 'Unknown')
    }
    
    # Re-iterate metrics to get raw floats for CSV
    for c in metrics_to_show:
        if c in df_stat.columns:
            s = df_stat[c]
            var = c.replace("/", "_per_").lower()
            wide_row[f"{var}_mean"] = s.mean()
            wide_row[f"{var}_sd"] = s.std()
            wide_row[f"{var}_min"] = s.min()
            wide_row[f"{var}_max"] = s.max()
            wide_row[f"{var}_range"] = s.max()-s.min()
        
    df_wide = pd.DataFrame([wide_row])
    # Mac/European Excel format: sep=';', decimal=','
    csv_stats = df_wide.to_csv(index=False, sep=';', decimal=',')
    
    # Filename based on ID
    pid = st.session_state.data.get('pig_id', 'result')
    sid = st.session_state.data.get('step_id', 'analysis')
    fname_csv = f"{pid}_{sid}_stats.csv"
    

    st.download_button("Download Statistics CSV", csv_stats, fname_csv, "text/csv")

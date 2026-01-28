
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from rv_coupling import RVCouplingAnalyzer

# Set page layout
st.set_page_config(page_title="RV-PA Coupling Analyzer", layout="wide")

st.title("🫀 RV-PA Coupling Interactive Analyzer")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Pressure Data (XLSX/CSV)", type=["xlsx", "csv"])

sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", value=120.0, min_value=1.0)
co_input = st.sidebar.number_input("Cardiac Output (L/min)", value=5.0, min_value=0.1)

# --- Data Loading ---
@st.cache_data
def load_data(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        
        # Column mapping
        st.sidebar.subheader("Map Columns")
        cols = df.columns.tolist()
        
        # Try to auto-select if names match common patterns
        rv_default = next((c for c in cols if 'rv' in c.lower() and 'press' in c.lower()), cols[0])
        pa_default = next((c for c in cols if 'pa' in c.lower() and 'press' in c.lower()), cols[1] if len(cols)>1 else cols[0])
        time_default = next((c for c in cols if 'time' in c.lower() or 'sec' in c.lower()), None)
        
        rv_col = st.sidebar.selectbox("RV Pressure Column", cols, index=cols.index(rv_default))
        pa_col = st.sidebar.selectbox("PA Pressure Column", cols, index=cols.index(pa_default))
        
        # If no time column, generate one
        if time_default:
            time_col = st.sidebar.selectbox("Time Column", cols, index=cols.index(time_default))
            t = df[time_col].values
        else:
            st.sidebar.info("No Time column detected. Generating from Sampling Rate.")
            t = np.arange(len(df)) / sampling_rate
            
        p_rv = df[rv_col].values
        p_pa = df[pa_col].values
        
        # --- Analysis Initialization ---
        analyzer = RVCouplingAnalyzer(sampling_rate=sampling_rate)
        
        # Detect beats
        with st.spinner("Detecting beats..."):
            beats = analyzer.detect_beats(p_rv)
        
        if not beats:
            st.error("No beats detected in the RV pressure signal. Please check the data or sampling rate.")
            st.stop()
            
        st.write(f"Detected **{len(beats)}** beats.")
        
        # --- Interactive Analysis ---
        
        # Select Beat
        beat_indices = list(range(len(beats)))
        selected_beat_idx = st.selectbox("Select Beat to Analyze", beat_indices, format_func=lambda x: f"Beat {x+1} (Start: {t[beats[x]['start_idx']]:.2f}s)")
        
        beat = beats[selected_beat_idx]
        
        # --- Sliders for Adjustment ---
        col_controls, col_plot = st.columns([1, 2])
        
        with col_controls:
            st.subheader("Fine-tune Points")
            
            # Global indices within the full array
            start_global = beat['start_idx']
            end_global = beat['end_idx']
            # Define a window for the sliders to allow adjustment range
            window_size = int(sampling_rate * 1.5) # 1.5 seconds window
            view_start = max(0, start_global - int(0.2 * sampling_rate))
            view_end = min(len(p_rv), end_global + int(0.5 * sampling_rate))
            
            # Create local sliders relative to the view window or absolute?
            # Streamlit sliders work best with absolute values if range is not huge, or we can use small ranges.
            
            # Start of Ejection / Beat
            new_start = st.slider("Start (EDP)", 
                                  min_value=max(0, start_global - 50), 
                                  max_value=start_global + 50, 
                                  value=start_global)
            
            # Peak Flow (Pf) / Max Pressure
            peak_curr = beat['p_max_idx']
            new_peak = st.slider("Peak Flow (Pf / Pmax)", 
                                 min_value=new_start, 
                                 max_value=end_global + 20, 
                                 value=peak_curr)
            
            # End of Ejection
            end_curr = beat['dpdt_min_idx'] # Default end of ejection
            new_end = st.slider("End Ejection (dP/dt min)", 
                                min_value=new_peak, 
                                max_value=end_global + 100, 
                                value=end_curr)
            
            # Update beat dict
            beat['start_idx'] = new_start
            beat['p_max_idx'] = new_peak
            beat['dpdt_min_idx'] = new_end
            beat['end_idx'] = new_end + int(0.1*sampling_rate) # arbitrary end of beat for slicing
            
            # Calculate HR from this beat's duration (or previous/next if available)
            # Instantaneous HR = 60 / period. Period = distance between starts?
            # If we adjust 'start', we change period? Not necessarily, period is beat-to-beat.
            # Let's use the DETECTED period for HR, unless user manually sets HR.
            # Calculate instantaneous HR based on surrounding beats if possible
            if selected_beat_idx < len(beats)-1:
                next_start = beats[selected_beat_idx+1]['start_idx']
                period_sec = (next_start - new_start) / sampling_rate
                hr_calc = 60.0 / period_sec
            else:
                hr_calc = 60.0 / ((new_end - new_start)/sampling_rate * 2) # Rough estimate
                
            st.metric("Calculated HR (bpm)", f"{hr_calc:.1f}")
            
            # Calculate mPAP for this beat (concurrent window)
            # mPAP is mean of PA pressure during the whole cycle or just ejection? 
            # Usually mean over the cardiac cycle.
            # Let's take mean PA from start to next start (or end of beat)
            p_pa_segment = p_pa[new_start : beats[selected_beat_idx+1]['start_idx'] if selected_beat_idx < len(beats)-1 else new_end+50]
            if len(p_pa_segment) > 0:
                mpap_calc = np.mean(p_pa_segment)
            else:
                mpap_calc = 0.0
            
            st.metric("Calculated mPAP (mmHg)", f"{mpap_calc:.1f}")

        # --- Calculations ---
        # Ees
        p_segment = p_rv[new_start : beat['end_idx']] # slice for fitting
        # Adjust beat info for local segment
        beat_local = beat.copy()
        beat_local['start_idx'] = 0
        beat_local['p_max_idx'] = new_peak - new_start
        beat_local['dpdt_min_idx'] = new_end - new_start
        beat_local['end_idx'] = len(p_segment)
        
        ees_pmax = analyzer.calculate_single_beat_ees(p_segment, beat_local)
        
        # Zc & Lambda
        # Use calculated mPAP and input CO
        zc, lam = analyzer.calculate_zc_lambda(p_segment, beat_local, co_l_min=co_input, m_pap=mpap_calc)
        
        # Ea calculation: (Pes - P_d?) / SV?
        # We don't have SV directly (unless stroke volume = CO/HR). 
        # SV = CO / HR
        if hr_calc > 0:
            sv_ml = (co_input * 1000) / hr_calc
        else:
            sv_ml = np.nan
        
        # Ea = Pes / SV? Or (Pes - P_d)/SV?
        # Oakland: Ea = RV ESP / SV index.
        # Let's use End Systolic Pressure (at dP/dt min or slightly before?)
        # Conventionally Pes is near dP/dt_min (incisura).
        pes = p_rv[new_end]
        
        if sv_ml > 0:
            ea = pes / sv_ml
        else:
            ea = np.nan
            
        coupling = ees_pmax / ea if ea > 0 else np.nan

        # --- Display Results ---
        with col_plot:
            st.subheader("Results")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Ees (mmHg/mL)", f"{ees_pmax:.2f}")
            r2.metric("Ea (mmHg/mL)", f"{ea:.2f}")
            r3.metric("Zc (mmHg·s/L)", f"{zc:.2f}")
            r4.metric("Lambda", f"{lam:.2f}")
            r5.metric("Coupling (Ees/Ea)", f"{coupling:.2f}")
            
            # --- Plotting ---
            fig = go.Figure()
            
            # Full signal trace (subset to relevant window)
            plot_start = max(0, new_start - 100)
            plot_end = min(len(p_rv), new_end + 100)
            t_subset = t[plot_start:plot_end]
            rv_subset = p_rv[plot_start:plot_end]
            pa_subset = p_pa[plot_start:plot_end]
            
            fig.add_trace(go.Scatter(x=t_subset, y=rv_subset, name="RV Pressure", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=t_subset, y=pa_subset, name="PA Pressure", line=dict(color='lightgreen')))
            
            # Markers
            fig.add_trace(go.Scatter(x=[t[new_start]], y=[p_rv[new_start]], mode='markers', marker=dict(color='blue', size=10), name="Start (EDP)"))
            fig.add_trace(go.Scatter(x=[t[new_peak]], y=[p_rv[new_peak]], mode='markers', marker=dict(color='gold', size=10), name="Peak (Pf)"))
            fig.add_trace(go.Scatter(x=[t[new_end]], y=[p_rv[new_end]], mode='markers', marker=dict(color='purple', size=10), name="End (Pes)"))
            
            fig.update_layout(title="Beat Analysis", xaxis_title="Time (s)", yaxis_title="Pressure (mmHg)", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # --- Batch Analysis & Download ---
        st.divider()
        st.subheader("Batch Analysis")
        if st.button("Analyze All Beats"):
            results = []
            progress_bar = st.progress(0)
            
            for i, b in enumerate(beats):
                # Recalculate basic params for each beat automatically (no manual adjust)
                # Note: Automated results might differ from manually tuned ones
                
                # Estimate HR
                if i < len(beats)-1:
                    per_sec = (beats[i+1]['start_idx'] - b['start_idx']) / sampling_rate
                    hr_b = 60.0 / per_sec
                else:
                    hr_b = hr_calc # Use last known good
                
                # Estimate mPAP
                p_pa_seg = p_pa[b['start_idx'] : beats[i+1]['start_idx'] if i < len(beats)-1 else b['end_idx']]
                mpap_b = np.mean(p_pa_seg) if len(p_pa_seg) > 0 else 0
                
                # Ees
                p_seg = p_rv[b['start_idx']:b['end_idx']]
                b_loc = b.copy()
                b_loc['start_idx'] = 0
                b_loc['p_max_idx'] -= b['start_idx']
                b_loc['dpdt_min_idx'] -= b['start_idx']
                b_loc['end_idx'] = len(p_seg)
                
                ees_val = analyzer.calculate_single_beat_ees(p_seg, b_loc)
                zc_val, lam_val = analyzer.calculate_zc_lambda(p_seg, b_loc, co_l_min=co_input, m_pap=mpap_b)
                
                # Ea
                sv_b = (co_input * 1000) / hr_b if hr_b > 0 else np.nan
                pes_b = p_rv[b['dpdt_min_idx']]
                ea_val = pes_b / sv_b if sv_b > 0 else np.nan
                
                results.append({
                    "Beat": i+1,
                    "Time_Start": t[b['start_idx']],
                    "HR_bpm": hr_b,
                    "mPAP_mmHg": mpap_b,
                    "Ees": ees_val,
                    "Ea": ea_val,
                    "Coupling_Ees_Ea": ees_val/ea_val if ea_val else np.nan,
                    "Zc": zc_val,
                    "Lambda": lam_val,
                    "CO_Input": co_input
                })
                progress_bar.progress((i+1)/len(beats))
                
            res_df = pd.DataFrame(results)
            st.dataframe(res_df)
            
            # Download
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results (CSV)",
                csv,
                "rv_coupling_results.csv",
                "text/csv",
                key='download-csv'
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin.")

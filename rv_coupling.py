import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.optimize import curve_fit

def weibull_4_param(x, a, x0, b, c):
    """
    4-parameter Weibull peak fit function used by SigmaPlot.
    Reference: f(x) = a * ((c-1)/c)^((1-c)/c) * (abs((x-x0)/b + ((c-1)/c)^(1/c))^(c-1)) * exp(-abs((x-x0)/b + ((c-1)/c)^(1/c))^c + (c-1)/c)
    """
    # To avoid complex numbers and domain errors, we handle the term inside the power carefully
    # The term Z = (x-x0)/b + ((c-1)/c)^(1/c) must be handled.
    # The formula is valid for x > x0 - b*((c-1)/c)^(1/c)
    
    try:
        term1 = ((c-1)/c)**((1-c)/c)
        offset = ((c-1)/c)**(1/c)
        
        Z_arg = (x - x0)/b + offset
        
        # We use absolute value as per the formula found, but strictly the domain is limited
        Z = np.abs(Z_arg)
        
        # If Z is very close to 0, avoid division by zero or log errors if powers are negative
        # For simplicity in fitting, we can return 0 if out of bounds or invalid
        
        part1 = Z**(c-1)
        part2 = np.exp(-Z**c + (c-1)/c)
        
        y = a * term1 * part1 * part2
        
        # Handle the cut-off condition: if x is below the starting point of the Weibull
        # Theoretically f(x) = 0
        limit_x = x0 - b * offset
        mask = x > limit_x
        
        return np.where(mask, y, 0.0)
    except:
        return np.zeros_like(x)

class RVCouplingAnalyzer:
    def __init__(self, sampling_rate=100.0):
        self.fs = sampling_rate

    def detect_beats(self, pressure, ecg=None):
        """
        Detect individual beats from RV pressure.
        Returns a list of dicts with indices for start, end, peak, dPdt_max, dPdt_min.
        """
        # Simple peak detection for systole
        # Identify peaks with minimal height and distance
        peaks, _ = signal.find_peaks(pressure, height=np.mean(pressure), distance=self.fs*0.2)
        
        beats = []
        for i in range(len(peaks) - 1):
            p_idx = peaks[i]
            
            # Find dP/dt
            dpdt = np.gradient(pressure, 1/self.fs)
            
            # Search window around peak
            start_search = max(0, p_idx - int(0.2 * self.fs))
            end_search = min(len(pressure), p_idx + int(0.2 * self.fs))
            
            window_slice = slice(start_search, end_search)
            window_dpdt = dpdt[window_slice]
            
            # dP/dt max (Start of Ejection)
            if len(window_dpdt) == 0:
                continue
            dpdt_max_loc = np.argmax(window_dpdt) + start_search
            
            # dP/dt min (End of Ejection)
            dpdt_min_loc = np.argmin(window_dpdt) + start_search
            
            # Start of beat (EDP)
            pre_eject_start = max(0, dpdt_max_loc - int(0.15*self.fs))
            pre_eject_end = dpdt_max_loc
            
            if pre_eject_end > pre_eject_start:
                 beat_start = np.argmin(pressure[pre_eject_start:pre_eject_end]) + pre_eject_start
            else:
                 # Fallback if too close to start
                 beat_start = max(0, dpdt_max_loc - 10)
            
            # P_max location
            p_max_loc = p_idx
            
            beats.append({
                'start_idx': beat_start,
                'p_max_idx': p_max_loc,
                'dpdt_max_idx': dpdt_max_loc,
                'dpdt_min_idx': dpdt_min_loc,
                'end_idx': dpdt_min_loc + int(0.1 * self.fs) # Approximate end of beat
            })
            
        return beats

    def estimate_pmax_isovolumic(self, pressure_segment, beat_info):
        """
        Estimate theoretical max isovolumic pressure (Pmax) using Weibull fit.
        pressure_segment: numpy array of pressure for the beat
        beat_info: dict with indices relative to the segment
        
        Returns:
            float: Pmax (mmHg)
        """
        # Implementation of the Chen/Oakland method
        # 1. Normalize time and amplitude
        # 2. Fit the isovolumic portions (upstroke and relaxation)
        # But Oakland paper specifically says: 
        # "Pressure segments... from half of first peak (EDP) to second peak (dP/dt max?)..." - Wait, need to re-read carefully
        
        # Oakland text: "from half of the first peak (end diastolic pressure or EDP) to the second peak (the first inflection point, Pi), and from the third peak to the fourth (end)."
        # This refers to the SECOND DERIVATIVE squared peaks.
        
        # Let's compute second derivative squared
        if len(pressure_segment) >= 11:
            p_smooth = signal.savgol_filter(pressure_segment, 11, 3) # Smooth slightly
        elif len(pressure_segment) > 3:
            # Fallback for very short segments: use smaller window (must be odd)
            win_len = len(pressure_segment) if len(pressure_segment) % 2 == 1 else len(pressure_segment) - 1
            p_smooth = signal.savgol_filter(pressure_segment, win_len, 3) if win_len > 3 else pressure_segment
        else:
            p_smooth = pressure_segment
        d1 = np.gradient(p_smooth)
        d2 = np.gradient(d1)
        d2_sq = d2**2
        
        # Identify the peaks in d2_sq to find isovolumic periods
        # This usually marks the "corners" of the pressure curve
        # We need specific logic here, but for now let's implement the standard wrapping
        
        # Standard Single Beat Method (Chen et al.):
        # Fit sinusoidal or model to isovolumic upstroke and downstroke
        
        # Using the specific Oakland Weibull fit:
        # We need to extract the segments corresponding to isovolumic contraction and relaxation
        # Typically:
        # Isovolumic Contraction: EDP to dP/dt_max
        # Isovolumic Relaxation: dP/dt_min to MVO (Mitral Valve Opening equivalent)
        
        t = np.arange(len(pressure_segment)) / self.fs
        
        # Let's try to fit the whole curve or just the systolic part to find Pmax?
        # The paper fits "Pmax prediction" from "pressure segments".
        # Assuming we can fit the defined Weibull function to the active systolic phase (isovolumic parts).
        
        # Simplified approach for this task:
        # Use simple sinusoidal approximation if Weibull assumes complex segmentation, 
        # OR try to fit Weibull to the whole beat (it describes a peaked shape).
        
        # Let's treat the whole systolic phase (EDP -> End Ejection) + some relaxation as the "Peak"
        # and fit the Weibull to find the "Theoretical Max Pressure".
        
        try:
            # Initial guesses
            a_guess = np.max(pressure_segment) * 1.5 # Pmax > Ppeak
            x0_guess = np.argmax(pressure_segment) / self.fs
            b_guess = 0.2
            c_guess = 2.0
            
            popt, _ = curve_fit(weibull_4_param, t, pressure_segment, p0=[a_guess, x0_guess, b_guess, c_guess], maxfev=10000)
            
            p_max_theoretical = popt[0] # The amplitude 'a'
            
            # Es (Ees) = (Pmax - Pes) / SV
            # Wait, Ees is slope. Ees ~ Pmax / Vs (if V0=0).
            # Oakland: "Ees was then determined as: (Pmax - RV ESP) / stroke volume index"
            # Wait, usually Ees = Pes / Ves.
            # Single beat estimation: Ees = (Pmax - Pes) / SV. (where Pes is end-systolic P).
            # Yes, standard formula is Ees(sb) = (Pmax - Pes) / SV.
            
            return p_max_theoretical
        except:
            return np.nan

    def calculate_zc_lambda(self, pressure_beat, beat_info, co_l_min, m_pap=None):
        """
        Calculate Zc and Lambda.
        co_l_min: Cardiac Output in L/min
        """
        # Parse inputs
        if co_l_min is None or co_l_min <= 0:
            return np.nan, np.nan
        
        # 1. Calculate Flow Index / eCO components
        # eCO = HR * [ ((Pf - P_dpdt_min) * ED) / 2 ] * (1/Zc)
        # But we want Zc = eCO_unscaled / CO_measured
        # Actually formula says: Zc = estimated_CO / TD_CO ? No.
        # "From the eCO (equation) and simultaneous measurement of CO (TD CO), Zc can be estimated as the ratio of eCO to TD CO."
        # Wait. "eCO ... * 1/Zc".
        # So eCO_val = Term / Zc.
        # If we assume eCO_val should equal TD_CO, then
        # TD_CO = Term / Zc  =>  Zc = Term / TD_CO.
        
        # Term = HR * [ ((Pf - P_dpdt_min) * ED) / 2 ]
        
        p_f = pressure_beat[beat_info['p_max_idx']] # Pf = P_max (Assumption)
        p_end_eject = pressure_beat[beat_info['dpdt_min_idx']] # P at dP/dt_min
        
        # ED = Ejection Duration (seconds)
        ed_idx = beat_info['dpdt_min_idx'] - beat_info['dpdt_max_idx']
        ed_sec = ed_idx / self.fs
        
        # Instantaneous HR for this beat (or use average if provided, here instantaneous)
        # Just use 1 beat so HR is not really defined per beat, but we can assume 60/T_beat or just use the scalar term
        # The formula includes "heart rate".
        # Let's assume we need the HR of the animal.
        # If analyzing single beat, we might not have HR.
        # For now, let's assume terms are per minute.
        
        # Let's calculate the "Pressure-Time Area" term for ejection triangle
        # Area ~ (Height * Base) / 2 = (Pf - P_end) * ED / 2
        
        # Zc Calculation
        # Formula: CO = HR * Area * (1/Zc)  =>  Zc = (HR * Area) / CO
        # Area = (Pf - P_min) * ED / 2
        
        # 1. Calculate Area (mmHg * sec)
        area_mmHg_sec = ((p_f - p_end_eject) * ed_sec) / 2.0
        
        # 2. Estimate HR (bpm) if not provided
        # Use beat duration from detection (approximate)
        beat_duration_sec = (beat_info['end_idx'] - beat_info['start_idx']) / self.fs
        # Better: use previous beat interval if available, or just estimating 60/T
        hr_bpm = 60.0 / beat_duration_sec if beat_duration_sec > 0 else 0
        
        # 3. Calculate Zc in hybrid units (mmHg * sec / (L/min)) / min ??
        # Wait, formula: eCO(L/min) = HR(bpm) * Area * (1/Zc).
        # So Zc = (HR * Area) / eCO.
        # Unit check:
        # bpm * mmHg * sec / (L/min) = (1/min) * mmHg * sec / (L/min) 
        # = mmHg * sec / L.
        # This is strictly resistance units (Pressure * Time / Volume).
        
        if co_l_min > 0:
            zc_hybrid = (hr_bpm * area_mmHg_sec) / co_l_min
        else:
            zc_hybrid = np.nan
            
        # 4. Lambda Calculation: (TPR - Zc) / (TPR + Zc)
        # We need TPR (Total Pulmonary Resistance).
        # TPR = mPAP / CO.
        # mPAP should be Mean Pulmonary Artery Pressure. 
        # If not provided, we can't calculate Lambda accurately.
        # Approximating mPAP from RV pressure? (mRV ~ mPAP? No).
        # We return Zc and partial Lambda if mPAP is missing.
        
        lambda_val = np.nan
        if m_pap is not None and co_l_min > 0:
            tpr = m_pap / co_l_min # Units: mmHg / (L/min) -> Hybrid Resistance Units
            # Note: TPR and Zc must be in SAME units for the ratio.
            # zc_hybrid is also in mmHg / (L/min) approx?
            # zc_hybrid = (bpm * mmHg * sec) / (L/min).
            # Unit: mmHg * (sec/min) * (1/(L/min)) ?? No.
            # Let's check scalar dimensions.
            # Formula: eCO = HR * Area * (1/Zc).
            # L/min = (1/min) * (mmHg*sec) * (1/Zc).
            # Zc = (1/min) * mmHg * sec / (L/min) = mmHg * sec / L.
            
            # TPR = mPAP / CO = mmHg / (L/min) = mmHg * min / L = mmHg * (60 sec) / L.
            # So TPR_hybrid = mPAP / CO.
            # Zc_hybrid needs to be in (mmHg * min / L) to match TPR.
            # Zc_hybrid_current = mmHg * sec / L.
            # So we typically need to convert Zc to 'mmHg * min / L' or TPR to 'mmHg * sec / L'.
            # Usually strict CGS units avoid this.
            
            # Let's convert BOTH to CGS (dyne * sec * cm^-5).
            # Factor for mmHg/L/min to CGS = 80.
            # Factor for mmHg*sec/L to CGS:
            # 1 mmHg = 1333.2 dyne/cm2
            # 1 L = 1000 cm3
            # Zc(CGS) = Zc(hybrid) * 1333.2 / 1000 = 1.3332 * Zc(hybrid).
            
            # Wait. 
            # TPR(CGS) = (mPAP / CO) * 80.
            # Zc(CGS) from hybrid (mmHg*sec/L) = (Area*HR/CO) * ??? 
            # Let's re-derive Zc(CGS) directly.
            
            # Zc = (HR/60) * Area / (CO/60) ? No.
            # Let's stick to the paper's Ratio.
            # Lambda returns a dimensionless ratio.
            # As long as TPR and Zc are in same units.
            
            # TPR_units = Pressure / Flow_mean.
            # Zc_units = Pressure / Flow_pulsatile.
            
            # Zc = (HR_bpm * Area_mmHg_sec) / CO_L_min.
            # = (1/60 Hz) * (mmHg*sec) / (L/60 sec) ??
            # = mmHg * sec / L.
             
            # TPR = mPAP / CO = mmHg / (L/min) = mmHg / (L / 60 sec) = 60 * mmHg * sec / L.
            
            # So: Zc_comparable = Zc_hybrid.
            # TPR_comparable = TPR_hybrid * (1/60)? No.
            # Flow_mean (L/sec) = CO / 60.
            # TPR_sec_units = mPAP / (CO/60) = 60 * mPAP / CO.
            
            # So Zc (mmHg*sec/L) vs TPR (mmHg*sec/L).
            
            # Zc_val = (hr_bpm * area_mmHg_sec) / co_l_min  <- This is WRONG dimensional analysis.
            # eCO (L/min) = HR(bpm) * Area * (1/Zc).
            # Zc = HR * Area / eCO.
            # Dim(Zc) = (1/min) * (mmHg*sec) / (L/min) = mmHg * sec / L.
            
            # TPR = mPAP / CO(L/min).
            # Dim(TPR) = mmHg / (L/min) = mmHg * min / L = 60 * mmHg * sec / L.
            
            # So, to compare:
            # Zc_compatible = Zc_val
            # TPR_compatible = (mPAP / CO) * 60.
            
            tpr_sec_units = (m_pap / co_l_min) * 60.0
            zc_sec_units = zc_hybrid
            
            lambda_val = (tpr_sec_units - zc_sec_units) / (tpr_sec_units + zc_sec_units)
            
        return zc_hybrid, lambda_val



def compute_wedge_empirical(time, pap, tzero_time, t_end=None, fs=100.0, window_size=20, n_points=20):
    """
    Computes Empirical Wedge Pressure as the mean of the N lowest SMOOTHED values 
    after tzero, optionally bounded by t_end.
    
    Args:
        time (array): Time array.
        pap (array): Raw Pressure array.
        tzero_time (float): Occlusion start time.
        t_end (float): Optional end time for the search window.
        fs (float): Sampling frequency.
        window_size (int): Smoothing window size (default 20).
        n_points (int): Number of lowest points to average (default 20).
        
    Returns:
        float: The calculated Wedge Pressure.
    """
    # 1. Smooth the signal (same method as compute_pcap)
    if len(pap) < window_size:
        return 0.0
        
    # Simple moving average
    kernel = np.ones(window_size) / window_size
    pap_smooth = np.convolve(pap, kernel, mode='same')
    
    # 2. Slice from tzero onwards (and up to t_end)
    if t_end is not None:
        mask = (time >= tzero_time) & (time <= t_end)
    else:
        mask = time >= tzero_time
        
    p_segment = pap_smooth[mask]
    
    if len(p_segment) < n_points:
        return np.mean(p_segment) if len(p_segment) > 0 else 0.0
    
    # 3. Find N lowest values
    lowest_n = np.sort(p_segment)[:n_points]
    
    # 4. Average
    wedge_val = np.mean(lowest_n)
    
    return wedge_val


def compute_pcap(time, pap, tzero_time, fs=100.0):
    """
    Compute Pcap using the same logic used in the reference notebook/R workflow.
    Core points:
    - smooth PAP with moving window (rolling mean, window=20)
    - discard the first 50 samples after smoothing
    - fit exp decay on [tzero+0.3s, tzero+2.0s]
    - fit is done on scaled pressure and sample index (not absolute time)
    - Pcap is evaluated at tzero (pcap_0)
    """
    time = np.asarray(time, dtype=float)
    pap = np.asarray(pap, dtype=float)
    if len(time) < 70 or len(pap) < 70:
        return {'fit_success': False, 'message': 'Not enough samples for pcap computation'}

    # Infer sampling rate from time vector when possible.
    fs_local = fs
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) > 0:
        fs_local = 1.0 / np.median(dt)

    # Exact smoothing style used in notebook: rolling mean (window=20).
    pap_smooth = pd.Series(pap).rolling(window=20, min_periods=20).mean().to_numpy()

    # Match notebook preprocessing.
    trim_n = 50
    t_proc = time[trim_n:]
    p_proc = pap_smooth[trim_n:]
    valid = np.isfinite(t_proc) & np.isfinite(p_proc)
    t_proc = t_proc[valid]
    p_proc = p_proc[valid]
    if len(t_proc) < 20:
        return {'fit_success': False, 'message': 'Not enough valid smoothed points after trim'}

    # tzero from absolute time to processed index.
    tzero_idx = int(np.argmin(np.abs(t_proc - float(tzero_time))))
    tzero_proc_time = float(t_proc[tzero_idx])

    # Fit window in sample offsets.
    offset_1 = int(0.3 * fs_local)
    offset_2 = int(2.0 * fs_local)
    t1_idx = tzero_idx + offset_1
    t2_idx = tzero_idx + offset_2
    if t1_idx < 0 or t2_idx >= len(p_proc) or t2_idx <= t1_idx:
        return {'fit_success': False, 'message': 'Fit window out of signal bounds'}

    p_fit_vals = p_proc[t1_idx : t2_idx + 1]
    if len(p_fit_vals) < 10:
        return {'fit_success': False, 'message': 'Not enough points in fit window (0.3-2.0s)'}

    max_pap = np.nanmax(p_fit_vals)
    if not np.isfinite(max_pap) or max_pap <= 0:
        return {'fit_success': False, 'message': 'Invalid fit scaling factor'}
    p_scaled = p_fit_vals / max_pap
    time_fit_vals = np.arange(1, len(p_fit_vals) + 1, dtype=float)

    # Fit in sample domain as in notebook.
    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    try:
        popt, _ = curve_fit(
            exp_decay,
            time_fit_vals,
            p_scaled,
            p0=[0.5, 0.01, 0.4],
            bounds=([0, 0, 0], [10, 1.0, 5]),
            maxfev=10000,
        )
        a_fit, b_fit, c_fit = popt

        # Back-scale parameters; convert decay constant to 1/seconds for plotting usage.
        a = a_fit * max_pap
        b = b_fit * fs_local
        c = c_fit * max_pap

        # R^2 on scaled fit domain.
        residuals = p_scaled - exp_decay(time_fit_vals, a_fit, b_fit, c_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((p_scaled - np.mean(p_scaled))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Reference output: pcap at tzero.
        pcap_val = a + c
        pcap_95ms = a * np.exp(-b * 0.095) + c

        # Keep returned time axis compatible with existing plotting code.
        t_fit_abs = t_proc[t1_idx : t2_idx + 1]
        t_rel = t_fit_abs - tzero_proc_time

        return {
            'fit_success': True,
            'pcap': pcap_val,
            'pcap_95ms': pcap_95ms,
            'params': (a, b, c),
            'r_squared': r_squared,
            'time_rel': t_rel,
            'pap_smooth': pap_smooth,
            'fit_func': exp_decay
        }

    except Exception as e:
        return {'fit_success': False, 'message': str(e)}

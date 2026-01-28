# RV-PA Coupling Analysis Tool & Methodology

## Overview
This software provides an interactive tool to analyze **Right Ventricular - Pulmonary Arterial (RV-PA) Coupling** using raw pressure waveforms. It implements the methodology described by **Oakland et al. (2021)**, allowing for the estimation of advanced coupling variables (Zc, Lambda) without requiring high-fidelity flow signals, alongside standard Single-Beat Ees estimation.

## 1. Scientific Methodology

### 1.1 End-Systolic Elastance ($E_{es}$) - Single Beat Method
$E_{es}$ represents the contractility of the right ventricle.
- **Method**: We use a **Single-Beat Estimation** technique.
- **Algorithm**: A 4-parameter **Weibull function** is fitted to the systolic portion of the RV pressure curve. This allows us to extrapolate the theoretical maximum isovolumic pressure ($P_{max}$) that would have been generated if the valve had not opened.
- **Formula**:
  $$E_{es} = \frac{P_{max} - P_{es}}{SV}$$
  *Where $P_{es}$ is End-Systolic Pressure and $SV$ is Stroke Volume.*

### 1.2 Arterial Elastance ($E_a$)
$E_a$ represents the total afterload presented to the ventricle.
- **Formula**:
  $$E_a = \frac{P_{es}}{SV}$$
  *(Note: Sometimes defined as $(P_{es} - P_{d})/SV$, but we use the standard ratio for coupling)*.

### 1.3 Characteristic Impedance ($Z_c$)
$Z_c$ represents the stiffness of the proximal pulmonary arteries and the resistance to pulsatile flow.
- **Oakland et al. Approximation**: Instead of needing flow waves, we estimate $Z_c$ using the shape of the pressure ejection triangle.
- **Formula**:
  $$Z_c = \frac{\text{FlowIndex}}{CO}$$
  where $\text{FlowIndex} = \text{HR} \times \frac{(P_f - P_{min\_eject}) \times ED}{2}$
- **Variables**:
  - $P_f$ (Pressure at Peak Flow): **Assumed to be $P_{max\_RV}$** (Peak RV Pressure) based on validation.
  - $P_{min\_eject}$: Pressure at end of ejection ($dP/dt_{min}$).
  - $ED$: Ejection Duration.
  - $CO$: Cardiac Output (Scalar reference value).

### 1.4 Wave Reflection Coefficient ($\lambda$)
$\lambda$ quantifies the amount of backward wave reflection from the distal vasculature.
- **Formula**:
  $$\lambda = \frac{TPR - Z_c}{TPR + Z_c}$$
  where Total Pulmonary Resistance $TPR = \frac{mPAP}{CO}$ (converted to consistent units).

---

## 2. Interactive App User Guide

### 2.1 Installation & Startup
1. Ensure you have Python installed.
2. Install dependencies:
   ```bash
   pip install streamlit plotly pandas openpyxl scipy
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

### 2.2 Loading Data
- **File Format**: The app accepts Excel (`.xlsx`) or CSV (`.csv`) files.
- **Structure**: The file should contain columns for **Right Ventricular Pressure** and **Pulmonary Artery Pressure**. A Time column is optional (if missing, time is generated from the sampling rate).
- **Column Mapping**: Use the sidebar to map your specific column names to the analysis variables.

### 2.3 Configuration Parameters (Sidebar)
- **Sampling Rate**: Default is 120 Hz. Crucial for accurate time calculations.
- **Cardiac Output (CO)**: Must be entered manually (L/min) as it cannot be derived from pressure alone.
- **Beat Selection**: Use the dropdown menu to analyze specific beats one by one.

### 2.4 Interactive Analysis (Main View)
1. **The Plot**: Shows the RV (Red) and PA (Green) traces for the selected beat.
2. **Sliders**: Adjust the key definition points if the automatic detection is inaccurate:
   - **Start (EDP)**: Start of systole.
   - **Peak (Pf)**: Pressure at peak flow.
   - **End (Peak)**: End of ejection (incisura).
3. **Real-time Updates**: As you move sliders, $E_{es}$, $Z_c$, and Coupling ratios recalculate instantly.

### 2.5 Batch Analysis & Export
- Click **"Analyze All Beats"** to process the entire file using the current settings.
- A table will generate with beat-by-beat variables.
- Click **"Download Results (CSV)"** to save the full report.

---

## 3. Troubleshooting
- **No beats detected?** Check your Sampling Rate or ensure the pressure units are correct (mmHg). Non-physiological ranges (< 5 mmHg peak) may fail detection.
- **Negative Lambda?** Physiologically impossible; usually indicates $Z_c > TPR$. Check if the input Cardiac Output is correct or if the mPAP calculation (from PA trace) is noisy.

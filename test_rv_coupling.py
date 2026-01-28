
import numpy as np
import matplotlib.pyplot as plt
from rv_coupling import RVCouplingAnalyzer, weibull_4_param

def generate_synthetic_pressure(fs=100, heart_rate=80, duration=5):
    t = np.linspace(0, duration, int(fs*duration))
    # Synthetic RV pressure: base sine + harmonics to make it skewed
    freq = heart_rate / 60.0
    pressure = 10 + 20 * np.sin(2 * np.pi * freq * t - 0.5) + \
               10 * np.sin(4 * np.pi * freq * t - 0.5) * 0.5
    pressure[pressure < 5] = 5 # Baseline
    
    # Add some noise
    pressure += np.random.normal(0, 0.5, size=len(t))
    
    return t, pressure

def test_analyzer():
    fs = 200
    t, p = generate_synthetic_pressure(fs=fs)
    
    analyzer = RVCouplingAnalyzer(sampling_rate=fs)
    beats = analyzer.detect_beats(p)
    
    print(f"Detected {len(beats)} beats.")
    
    if len(beats) > 0:
        beat = beats[0]
        print("Beat 0 info:", beat)
        
        p_segment = p[beat['start_idx']:beat['end_idx']+50]
        
        # Test Weibull
        p_max_theo = analyzer.calculate_single_beat_ees(p_segment, beat)
        print(f"Theoretical Pmax: {p_max_theo:.2f} mmHg (Actual Peak: {np.max(p_segment):.2f})")
        
        # Test Zc term
        zc_term = analyzer.calculate_zc_lambda(p, beat, co_l_min=5.0)
        print(f"Zc Nucleus Term (mmHg*sec): {zc_term:.4f}")

if __name__ == "__main__":
    test_analyzer()

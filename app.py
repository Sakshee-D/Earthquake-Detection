import io
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import google.generativeai as genai
import textwrap
from datetime import datetime
import matplotlib as mpl
from urllib.request import urlopen

from dsp_pipeline import (
    run_pipeline,
    parse_acc_v2c,
    bandpass_filter,
    wavelet_decompose,
    wavelet_cf_highfreq,
    detect,
    normalize_to_v2c,
)

st.set_page_config(page_title="DSP Based Earthquake Detection", layout="wide")
st.title("DSP Based Earthquake Detection")

with st.sidebar:
    st.header("Inputs & Parameters")
    data_source = st.radio(
        "Data source",
        ["Upload file", "Synthetic example", "From URL"],
        index=0,
    )
    fs = st.number_input("Sampling rate (Hz)", min_value=1.0, max_value=5000.0, value=100.0, step=1.0)
    lowcut = st.number_input("Low cut (Hz)", min_value=0.001, max_value=max(0.002, fs/2 - 0.001), value=0.1, step=0.05, format="%.3f")
    highcut = st.number_input("High cut (Hz)", min_value=lowcut + 0.001, max_value=max(lowcut + 0.002, fs/2 - 0.001), value=40.0, step=1.0)
    wavelet = st.selectbox("Wavelet", ["db2", "db3", "db4", "sym4", "coif1"], index=2)
    thr_mode = st.selectbox("Threshold mode", ["mean_std", "absolute"], index=0)
    k = st.slider("k (mean + k*std)", 0.0, 5.0, 1.0, 0.1)
    absolute_thr = st.number_input("Absolute threshold (if mode=absolute)", value=0.5)
    units_label = st.text_input("Units label", value="Acceleration (cm/s²)")

    st.caption("Upload a V2c-like text file where values begin after a line containing 'acceleration pts'. Lines starting with '|' are ignored; -999 placeholders are skipped.")

if data_source == "Upload file":
    uploaded = st.file_uploader("Upload file (any text/CSV/V2/V2c)")

    if uploaded is None:
        st.info("Waiting for file upload...")
        st.stop()

    file_bytes = uploaded.read()
    api_key = st.secrets.get("GEMINI_API_KEY")
    normalized_bytes = normalize_to_v2c(
        file_bytes=file_bytes,
        fs_hint=fs,
        units_hint=units_label,
        use_gemini=bool(api_key),
        api_key=api_key,
    )
elif data_source == "From URL":
    url = st.text_input("File URL (text format with 'acceleration pts' header)")
    if not url:
        st.info("Enter a URL to fetch the file.")
        st.stop()
    try:
        with urlopen(url) as resp:
            file_bytes = resp.read()
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        st.stop()
else:
    # Create a synthetic signal: noise + band-limited impulse between samples
    n = 12000
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.3, n)
    burst = np.zeros(n)
    start, end = int(0.4 * n), int(0.45 * n)
    burst[start:end] = 5.0 * np.sin(2 * np.pi * 5 * t[start:end])
    acc = noise + burst
    # Serialize to a V2c-like text for reuse of the same parser
    header = "Synthetic example\n{} acceleration pts\n".format(n)
    content = header + "\n".join(f"{v:.6f}" for v in acc)
    file_bytes = content.encode("utf-8")

# Normalize to V2c-like if needed (backend-only)
api_key = st.secrets.get("GEMINI_API_KEY")
normalized_bytes = normalize_to_v2c(
    file_bytes=file_bytes,
    fs_hint=fs,
    units_hint=units_label,
    use_gemini=bool(api_key),
    api_key=api_key,
)

# Run pipeline
try:
    results = run_pipeline(
        file_bytes=normalized_bytes,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        wavelet=wavelet,
        threshold_mode=thr_mode,
        k=k,
        absolute=absolute_thr if thr_mode == "absolute" else None,
    )
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

acc = np.asarray(results.get("acc", []))
filtered = np.asarray(results.get("filtered", []))
cf = np.asarray(results.get("cf", []))
detections = np.atleast_1d(np.asarray(results.get("detections", [])))  # ✅ Force it to array
thr = results.get("threshold", np.nan)

# --- P-wave detection visualization (from wavelet CF) ---
if detections.size > 0:
    # Take first detection as P-wave pick
    p_pick = int(detections[0])
    
    import matplotlib.pyplot as plt
    fig_p, ax_p = plt.subplots(figsize=(10, 3))
    
    # Plot original signal
    ax_p.plot(acc, color="gray", lw=0.8, label="Original Signal")
    
    # Mark detected P-wave pick
    ax_p.axvline(p_pick, color="red", ls="--", lw=1.2, label=f"P-wave Pick = {p_pick}")
    
    ax_p.set_title("P-wave Detection (Wavelet-based)")
    ax_p.set_xlabel("Sample Index")
    ax_p.set_ylabel("Acceleration (cm/s²)")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper right")
    
    st.pyplot(fig_p, clear_figure=True)
else:
    st.info("No P-wave detected using wavelet CF method.")

# --- determine P-wave index robustly using STA/LTA (preferred for first arrivals) ---
def sta_lta_trigger(sig, fs, sta_sec=0.5, lta_sec=5.0, rel_thresh=3.0):
    """
    Simple STA/LTA: short-term average over sta_sec, long-term avg over lta_sec.
    Returns first index where STA/LTA > rel_thresh (or None).
    """
    n = len(sig)
    sta_n = max(1, int(round(sta_sec * fs)))
    lta_n = max(sta_n + 1, int(round(lta_sec * fs)))

    # use absolute amplitude
    a = np.abs(sig)

    # moving averages via convolution
    sta = np.convolve(a, np.ones(sta_n)/sta_n, mode='same')
    lta = np.convolve(a, np.ones(lta_n)/lta_n, mode='same')

    # avoid divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(lta > 0, sta / lta, 0.0)

    # find first crossing
    idxs = np.where(ratio > rel_thresh)[0]
    if idxs.size > 0:
        return int(idxs[0])
    return None

# Try STA/LTA first (recommended)
p_wave_index = None
try:
    # tune sta/lta params if necessary
    p_wave_index = sta_lta_trigger(filtered, fs, sta_sec=0.2, lta_sec=3.0, rel_thresh=2.5)
except Exception:
    p_wave_index = None

# If STA/LTA failed, fallback to CF detection (first detection)
if p_wave_index is None:
    if detections is not None and getattr(detections, "size", 0) > 0:
        p_wave_index = int(detections[0])
    else:
        p_wave_index = None

# --- Plot a safe P-wave window around detected index ---
if p_wave_index is not None:
    # define window length in seconds (pre and post)
    pre_sec = 0.5   # show 0.5 s before arrival
    post_sec = 1.0  # show 1.0 s after arrival
    pre_samples = int(round(pre_sec * fs))
    post_samples = int(round(post_sec * fs))

    start = max(0, p_wave_index - pre_samples)
    end = min(len(filtered), p_wave_index + post_samples)

    p_wave_segment = filtered[start:end]
    x_global = np.arange(start, end)  # global sample indices for correct axis

    st.subheader("Detected P-Wave Segment")
    fig_p, ax_p = plt.subplots(figsize=(10, 3))
    ax_p.plot(x_global, p_wave_segment, color="tab:orange", lw=1.2, label="Filtered")
    ax_p.axvline(p_wave_index, color="r", ls="--", label="Detected P-Wave")
    ax_p.set_xlabel("Sample index")
    ax_p.set_ylabel(f"Filtered {units_label}")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper right")
    st.pyplot(fig_p, clear_figure=True)
else:
    st.info("No P-wave detected (try lowering thresholds or alternative method).")

# --- Identify first P-wave detection (if any) ---
detections = np.asarray(detections)
if detections.size > 0:
    p_wave_index = detections[0]
else:
    p_wave_index = None



if acc.size == 0:
    st.warning("Parsed signal is empty. Check file format or try synthetic example.")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Acceleration")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(acc, lw=0.8)
    ax.set_xlabel("Sample index")
    ax.set_ylabel(units_label)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Filtered Acceleration")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(filtered, lw=0.8, color="tab:green")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel(f"Filtered {units_label}")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

with col2:
    st.subheader("Characteristic Function (CF) & Detections")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(cf, lw=0.9, color="tab:purple", label="CF")
    ax3.axhline(thr, color="tab:red", ls="--", label=f"Threshold = {thr:.3f}")
    if detections.size > 0:
        ax3.scatter(detections, cf[detections], color="orange", s=10, label="Detections")
    ax3.set_xlabel("Sample index")
    ax3.set_ylabel("CF (norm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")
    st.pyplot(fig3, clear_figure=True)

st.divider()

# Stats
detections = np.asarray(detections)
st.subheader("Summary")
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Samples", int(acc.size))
with colB:
    st.metric("Min/Max (raw)", f"{float(np.min(acc)):.3f} / {float(np.max(acc)):.3f}")
with colC:
    detections = np.asarray(detections)
    st.metric("Detections", int(detections.size))
with colD:
    st.metric("Threshold", f"{thr:.3f}")

with st.expander("Preview first 20 values (raw)"):
    st.write(acc[:20])

st.caption("This demo implements parsing, band-pass filtering, wavelet decomposition and a simple CF-based detection, adapted from your notebook steps.")

st.divider()

st.subheader("Analysis")
generate = st.button("Generate analysis")

if generate:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("API key is not configured in secrets.")
    else:
        try:
            genai.configure(api_key=api_key)
            summary_stats = {
                "samples": int(acc.size),
                "min": float(np.min(acc)),
                "max": float(np.max(acc)),
                "detections": int(detections.size),
                "threshold": float(thr),
                "fs": float(fs),
                "lowcut": float(lowcut),
                "highcut": float(highcut),
                "wavelet": str(wavelet),
                "threshold_mode": str(thr_mode),
                "k": float(k) if thr_mode == "mean_std" else None,
            }
            prompt = (
                "You are an expert signal processing assistant. "
                "Given the following DSP pipeline results, write a concise, well-structured markdown report with sections: Overview, Data & Preprocessing, Wavelet & CF Analysis, Detections & Interpretation, Recommendations. "
                "Use bullet points and short paragraphs. Include parameter values and interpret detections."
            )
            prompt += "\n\nParameters and stats:\n" + str(summary_stats)
            
            if detections.size > 0:
                prompt += f"\nFirst 10 detection indices: {detections[:10].tolist()}"

            report_md = ""
            last_err = None
            try:
                available = []
                for m in genai.list_models():
                    name = getattr(m, "name", "") or ""
                    methods = getattr(m, "supported_generation_methods", []) or []
                    # Exclude experimental/preview and 2.5 models to reduce 404/429 issues
                    if ("generateContent" in methods) and ("-exp" not in name) and ("2.5" not in name):
                        available.append(name)
            except Exception as me:
                available = []
                last_err = me

            # Preference order among available models
            preferred_order = [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "gemini-pro",
            ]

            candidate_list = []
            for pref in preferred_order:
                # Some SDKs expose names as 'models/<name>'
                if pref in available:
                    candidate_list.append(pref)
                full = f"models/{pref}"
                if full in available:
                    candidate_list.append(full)

            # If none of the preferred are available, use the first available
            if not candidate_list and available:
                candidate_list = available[:1]

            for mname in candidate_list:
                try:
                    model = genai.GenerativeModel(mname)
                    resp = model.generate_content(prompt)
                    report_md = resp.text if hasattr(resp, "text") else ""
                    if report_md:
                        break
                except Exception as me:
                    last_err = me
                    continue

            if not report_md and last_err is not None:
                raise last_err

            if not report_md:
                st.warning("No analysis text returned.")
            else:
                st.markdown(report_md)

                # Only on-screen analysis; downloads removed per request.
        except Exception as e:
            st.error(f"AI analysis error: {e}")

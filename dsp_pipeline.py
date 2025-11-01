import io
import re
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import pywt
import pandas as pd


def _to_v2c_like_bytes(values: np.ndarray, fs: float | None = None, units: str = "cm/s^2", per_line: int = 10) -> bytes:
    from datetime import datetime
    lines = []
    lines.append("COSMOS Strong-Motion Data (Converted)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if fs is not None:
        lines.append(f"Sampling rate (Hz): {fs}")
    lines.append(f"Units: {units}")
    lines.append("Note: Normalized to V2c-like plain text.")
    lines.append("")
    lines.append(f"{len(values)} acceleration pts")
    lines.append("")
    # values
    for i in range(0, len(values), per_line):
        block = values[i:i+per_line]
        lines.append(" ".join(f"{float(v):.6f}" for v in block))
    return ("\n".join(lines) + "\n").encode("utf-8")


def normalize_to_v2c(file_bytes: bytes, fs_hint: float | None = None, units_hint: str = "cm/s^2", use_gemini: bool = False, api_key: str | None = None) -> bytes:
    """
    Try to convert arbitrary uploaded content into a V2c-like text bytes.
    Strategy:
    1) Try existing COSMOS V2/V2c parser; if it yields data, return original bytes.
    2) Try CSV via pandas: detect numeric column (prefer columns named with 'acc').
    3) Try generic text: regex-extract floats from entire content; if many, use them.
    4) Optional Gemini fallback: ask model to identify the acceleration series and return numbers; then format as V2c.
    """
    # 1) Attempt native V2/V2c parse
    try:
        arr = parse_acc_v2c(file_bytes)
        if arr.size > 0:
            return file_bytes
    except Exception:
        pass

    text = file_bytes.decode("utf-8", errors="ignore")

    # 2) CSV attempt
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(text))
        if not df.empty:
            # Prefer columns with 'acc' in name
            cand_cols = [c for c in df.columns if isinstance(c, str) and "acc" in c.lower()]
            cols = cand_cols or list(df.columns)
            series = None
            for c in cols:
                try:
                    s = pd.to_numeric(df[c], errors="coerce").dropna()
                    if s.size > 0:
                        series = s
                        break
                except Exception:
                    continue
            if series is not None and series.size > 0:
                values = series.to_numpy(dtype=float)
                return _to_v2c_like_bytes(values, fs=fs_hint, units=units_hint)
    except Exception:
        pass

    # 3) Generic text floats extraction
    try:
        floats = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?", text)
        values = []
        for tok in floats:
            try:
                v = float(tok)
                values.append(v)
            except ValueError:
                continue
        if len(values) >= 100:  # require reasonable length
            return _to_v2c_like_bytes(np.asarray(values, dtype=float), fs=fs_hint, units=units_hint)
    except Exception:
        pass

    # 4) Gemini fallback (optional)
    if use_gemini and api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Keep prompt compact; request only a plain newline-separated list of numbers
            prompt = (
                "You are given arbitrary text content of a file that may contain labels and data. "
                "Extract the acceleration time series values in their native units as a newline-separated list of numbers, one per line. "
                "Do not include any extra words, headers, or explanations. Only output numbers.\n\n"
            )
            # Truncate to reasonable size to avoid token limits
            snippet = text[:20000]
            model_name = "gemini-1.5-flash"
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt + snippet)
                body = resp.text if hasattr(resp, "text") else ""
            except Exception:
                # fallback older name variations
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                resp = model.generate_content(prompt + snippet)
                body = resp.text if hasattr(resp, "text") else ""
            nums = []
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    nums.append(float(line))
                except ValueError:
                    # try space separated fallbacks
                    for t in line.split():
                        try:
                            nums.append(float(t))
                        except ValueError:
                            continue
            if len(nums) >= 50:
                return _to_v2c_like_bytes(np.asarray(nums, dtype=float), fs=fs_hint, units=units_hint)
        except Exception:
            pass

    # If all failed, return original bytes so upstream can error meaningfully
    return file_bytes


import io, re
import numpy as np

def parse_acc_v2c(file_bytes: bytes) -> np.ndarray:
    """
    Parse COSMOS V2/V2c acceleration text files safely.
    - Starts reading only *after* the line containing 'acceleration pts'.
    - Skips all integer/real header sections and comment lines.
    - Ignores -999 placeholders and empty lines.
    - Supports one or many values per line.
    Returns: numpy array of acceleration values (float).
    """
    acceleration = []
    data_started = False

    with io.StringIO(file_bytes.decode("utf-8", errors="ignore")) as f:
        for raw in f:
            line = raw.strip()

            # Skip empty or comment lines
            if not line or line.startswith("|"):
                continue

            # Detect start of acceleration data
            if not data_started:
                # Standard COSMOS marker
                if re.search(r"\bacceleration\s+pts\b", line, flags=re.IGNORECASE):
                    data_started = True
                    continue
                else:
                    continue  # Skip all header lines

            # Once data_started is True, read numbers only
            for t in line.split():
                try:
                    val = float(t)
                    if val != -999:
                        acceleration.append(val)
                except ValueError:
                    continue

    return np.asarray(acceleration, dtype=float)


def bandpass_filter(signal: np.ndarray, fs: float, lowcut: float = 0.1, highcut: float = 40.0, order: int = 4) -> np.ndarray:
    if signal.size == 0:
        return signal
    x = detrend(signal)
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    if high <= low:
        high = min(low * 2.0, 0.999999)
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, x, method="pad")
    return y


def wavelet_decompose(acc: np.ndarray, wavelet: str = "db4"):
    if acc.size == 0:
        return []
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(acc), w.dec_len)
    max_level = max(1, max_level)
    coeffs = pywt.wavedec(acc, w, level=max_level)
    return coeffs


def wavelet_cf_highfreq(acc, wavelet: str = "db4", num_high_levels: int = 3):
    if acc.size == 0:
        return np.zeros(1)

    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(acc), w.dec_len)
    coeffs = pywt.wavedec(acc, w, level=max_level)

    # Keep only high-frequency detail coefficients
    for i in range(len(coeffs)):
        if i > num_high_levels:
            coeffs[i] = np.zeros_like(coeffs[i])

    # Reconstruct high-frequency signal
    highfreq_signal = pywt.waverec(coeffs, wavelet)

    cf = highfreq_signal ** 2
    cf = cf / np.max(cf) if np.max(cf) != 0 else cf

    return np.asarray(cf)




def detect(cf: np.ndarray, method: str = "mean_std", k: float = 1.0, absolute: float | None = None):
    cf = np.asarray(cf)

    if cf.size == 0:
        return np.array([], dtype=int), np.nan
    if method == "absolute" and absolute is not None:
        thr = float(absolute)
    else:
        thr = float(np.mean(cf) + k * np.std(cf))
    idx = np.where(cf > thr)[0]
    return idx, thr


def run_pipeline(file_bytes: bytes | None, fs: float = 100.0, lowcut: float = 0.1, highcut: float = 40.0, wavelet: str = "db4", threshold_mode: str = "mean_std", k: float = 1.0, absolute: float | None = None):
    if file_bytes is None:
        raise ValueError("file_bytes is None")
    acc = parse_acc_v2c(file_bytes)
    filtered = bandpass_filter(acc, fs, lowcut, highcut)
    coeffs = wavelet_decompose(filtered, wavelet)
    cf = wavelet_cf_highfreq(filtered, wavelet=wavelet, num_high_levels=3)
    detections, thr = detect(cf, method=threshold_mode, k=k, absolute=absolute)
    return {
        "acc": acc,
        "filtered": filtered,
        "coeffs": coeffs,
        "cf": cf,
        "detections": detections,
        "threshold": thr,
    }

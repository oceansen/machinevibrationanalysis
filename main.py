import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime
import re
from typing import Tuple, Optional, Dict, List
import os

home = os.environ.get("STREAMLIT_HOME") or os.path.expanduser("~/.streamlit")
os.environ["STREAMLIT_HOME"] = home
os.makedirs(home, exist_ok=True)

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Machine Vibration Analysis App", layout="wide")

# Title and description
st.title("Machine Vibration Analysis App")
st.markdown(
    "Upload a JSON file to see variables with clear descriptions and per-channel plots.  \n"
    "**Memory (natural text)** now explains the tool-break result using *key frequencies* and their amplitudes.  \n"
    "**ML Training (Key‑freq only)** exports features built strictly from key frequencies (fr, ft, k·ft and sidebands) to predict tool breakage.  \n"
    "Key Frequencies tab shows spindle (fr), tooth‑passing (ft), TPF harmonics, and once‑per‑rev sidebands (± n·fr) with amplitude markers."
)

# --------------------------------------------------
# Sidebar: upload and settings
# --------------------------------------------------
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload JSON file", type="json")

harmonics_count = st.sidebar.number_input(
    "Number of harmonics to compute (for RPM-based analysis)", min_value=1, max_value=200, value=10, step=1
)

top_n = st.sidebar.number_input(
    "Top-N harmonics to list (for text)", min_value=1, max_value=int(harmonics_count), value=min(5, int(harmonics_count)), step=1
)

first_harmonic_threshold = st.sidebar.number_input(
    "List Top-N only if 1st harm. amplitude ≥", min_value=0.0, value=1000.0, step=100.0
)

# (this still governs how many TPF harmonics to consider in Key Frequencies)
k_tpf = st.sidebar.number_input(
    "TPF harmonics K (for Key Frequencies)", min_value=1, max_value=200, value=10, step=1
)
include_sidebands = st.sidebar.checkbox(
    "Add once-per-rev sidebands (± n·fr) around TPF harmonics", value=True
)
max_sideband_order = st.sidebar.number_input(
    "Max sideband order n (0 = none)", min_value=0, max_value=10, value=1, step=1
)
annotate_amplitudes = st.sidebar.checkbox("Annotate amplitudes at key frequencies", value=True)
annotation_min_amp = st.sidebar.number_input(
    "Annotation min amplitude (hide labels below)", min_value=0.0, value=0.0, step=1.0
)
apply_hann = st.sidebar.checkbox("Apply Hann window before FFT (recommended)", value=True)

# --------------------------------------------------
# Variable descriptions
# --------------------------------------------------
VAR_DESCRIPTIONS = {
    "d": "Tool diameter [mm]",
    "z": "Number of teeth [-]",
    "ap": "Axial depth of cut [mm]",
    "ae": "Radial depth of cut [mm]",
    "n": "Turning speed [rpm]",
    "f": "Feed per tooth [mm/z]",
    "type": "Type of machining (down=in accordance, up=in opposition)",
    "break": "Tool breakage (true=broken, false=intact)",
    "sample_frequency": "Sampling frequency [Hz]",
    "acel_x": "Accelerometer X-axis [m/s^2]",
    "acel_y": "Accelerometer Y-axis [m/s^2]",
}

def describe_key(k):
    return VAR_DESCRIPTIONS.get(k, k.replace("_", " ").capitalize())


def slug(s: str) -> str:
    """Safe feature name component."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------

def nearest_bin_amplitude(xf: np.ndarray, amp: np.ndarray, freq: float) -> Tuple[float, float]:
    """Return (bin_freq, amplitude) nearest to freq. If out of range, (nan, nan)."""
    if freq is None or freq <= 0 or len(xf) == 0 or np.isnan(freq) or freq > xf[-1]:
        return (float("nan"), float("nan"))
    idx = int(np.argmin(np.abs(xf - freq)))
    return (float(xf[idx]), float(amp[idx]))


def fmt_float(x, sig=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()):
            return f"{int(x)}"
        return f"{x:.{sig}g}"
    except Exception:
        return str(x)


@st.cache_data(show_spinner=False)
def compute_fft(signal: np.ndarray, fs: float, apply_hann: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute single-sided FFT amplitude spectrum."""
    if signal.size == 0 or fs <= 0:
        return np.array([]), np.array([])
    sig = signal.astype(float)
    if apply_hann:
        w = np.hanning(sig.size)
        sig = sig * w
    yf = np.fft.rfft(sig)
    xf = np.fft.rfftfreq(sig.size, 1.0 / fs)
    amp = np.abs(yf)
    return xf, amp


# --------------------------------------------------
# File upload and processing
# --------------------------------------------------
if uploaded_file:
    data = json.load(uploaded_file)

    # ---- Normalize channels ---------------------------------------------------
    channels = [k for k in data.keys() if k.startswith("Channel_")]
    axis_keys = [k for k in data.keys() if k.lower() in ("acel_x", "acel_y")]
    if axis_keys and not channels:
        for k in axis_keys:
            v = data.get(k, [])
            data[f"Channel_{k.upper()}"] = {
                "SignalName": describe_key(k),
                "Signal": v,
                "Unit": "m/s^2",
            }
        channels = [k for k in data.keys() if k.startswith("Channel_")]

    selected_channels = st.sidebar.multiselect(
        "Select Channels to Display (default: all)", channels, default=channels
    )

    # ---- Breakage flag --------------------------------------------------------
    broke = bool(data.get("break", False))
    st.sidebar.error("Tool Breakage: Yes" if broke else "Tool Breakage: No")

    # ---- Variables & Header ---------------------------------------------------
    blacklist = {"__header__", "__version__", "__globals__", "File_Header"}
    root_scalars = {
        k: v
        for k, v in data.items()
        if not isinstance(v, dict)
        and k not in blacklist
        and not isinstance(v, (list, tuple))
    }
    file_header = data.get("File_Header", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("File Variables")
        if root_scalars:
            df_vars = (
                pd.DataFrame({"Key": list(root_scalars.keys()), "Value": list(root_scalars.values())})
                .assign(Description=lambda d: d["Key"].map(describe_key))
                .set_index("Key")
            )
            st.table(df_vars[["Description", "Value"]])
        else:
            st.caption("No scalar variables found in the root of the JSON.")
    with col2:
        st.subheader("File Header")
        if file_header:
            df_header = pd.DataFrame(file_header, index=[0]).T.rename(columns={0: "Value"})
            st.table(df_header)
        else:
            st.caption("No 'File_Header' found.")

    # ---- Sample frequency & fundamental --------------------------------------
    fs = float(data.get("sample_frequency") or file_header.get("SampleFrequency", 1.0) or 1.0)

    f_fund, n_rpm = None, None
    if isinstance(data.get("n"), (int, float)) and data["n"] != 0:
        n_rpm = float(data["n"])
        f_fund = n_rpm / 60.0
    else:
        st.warning("Fundamental frequency not found: expected numeric key 'n' (RPM).")

    # ---- Teeth / TPF ----------------------------------------------------------
    z_teeth: Optional[int] = None
    if isinstance(data.get("z"), (int, float)) and data["z"] > 0:
        z_teeth = int(data["z"])  # number of flutes/teeth

    fr = f_fund if f_fund else None  # spindle rotational frequency
    ft = (z_teeth * fr) if (z_teeth and fr) else None  # tooth-passing frequency

    # --------------------------------------------------
    # Pre-pass: compute harmonics & quick stats (RPM-based)
    # --------------------------------------------------
    harmonic_tables: Dict[str, Tuple[str, str, Optional[pd.DataFrame]]] = {}
    bin_res_by_ch: Dict[str, float] = {}
    stats_by_ch: Dict[str, Dict[str, float]] = {}
    dom_by_ch: Dict[str, str] = {}

    for ch in selected_channels:
        ch_data = data.get(ch, {})
        label = ch_data.get("SignalName", ch)
        signal = np.asarray(ch_data.get("Signal", []), dtype=float)
        unit = ch_data.get("Unit", "")

        if signal.size == 0 or fs <= 0:
            harmonic_tables[ch] = (label, unit, None)
            continue

        xf, amp = compute_fft(signal, fs, apply_hann)
        bin_res = xf[1] - xf[0] if len(xf) > 1 else float("nan")
        bin_res_by_ch[ch] = bin_res

        # stats
        rms = float(np.sqrt(np.mean(signal ** 2))) if signal.size > 0 else np.nan
        peak = float(np.max(np.abs(signal))) if signal.size > 0 else np.nan
        stats_by_ch[ch] = {"rms": rms, "peak": peak, "unit": unit}

        df_h = None
        dom_text = "n/a"

        if f_fund and np.isfinite(f_fund) and len(xf) > 0:
            harmonics_idx = np.arange(1, int(harmonics_count) + 1)
            harmonics_freqs = harmonics_idx * f_fund

            harm_amps, bin_freqs = [], []
            for f_h in harmonics_freqs:
                bfreq, a = nearest_bin_amplitude(xf, amp, f_h)
                harm_amps.append(a)
                bin_freqs.append(bfreq)

            df_h = pd.DataFrame(
                {
                    "Harmonic #": harmonics_idx,
                    "Target f [Hz]": np.round(harmonics_freqs, 6),
                    "Bin f [Hz]": np.round(bin_freqs, 6),
                    "Amplitude": harm_amps,
                }
            )

            if np.isfinite(df_h["Amplitude"]).any():
                idx_dom = df_h["Amplitude"].astype(float).idxmax()
                dom_row = df_h.loc[idx_dom]
                dom_text = (
                    f"{int(dom_row['Harmonic #'])}× @ {dom_row['Bin f [Hz]']:.2f} Hz (amp {dom_row['Amplitude']:.3g}{(' ' + unit) if unit else ''})"
                )

        harmonic_tables[ch] = (label, unit, df_h)
        dom_by_ch[ch] = dom_text

    # --------------------------------------------------
    # Helper: compute per‑channel key‑frequency amplitudes & sidebands
    # --------------------------------------------------
    def compute_keyfreqs_for_channel(xf, amp, fr, ft, k_tpf: int, include_sb: bool, sb_orders: int):
        """Return dict with fr amplitude, list of k*ft amplitudes, and primary sideband ratios (n=1) per k."""
        out = {
            "fr": {"target_hz": fr, "bin_hz": float("nan"), "amp": float("nan")},
            "tpf": [],  # list of {k, target_hz, bin_hz, amp, sbr_n1}
        }
        if xf is None or len(xf) == 0:
            return out
        # spindle
        if fr:
            bfreq_fr, a_fr = nearest_bin_amplitude(xf, amp, fr)
            out["fr"] = {"target_hz": fr, "bin_hz": bfreq_fr, "amp": a_fr}
        # TPF harmonics
        if ft:
            fmax = xf[-1]
            for k in range(1, int(k_tpf) + 1):
                target = k * ft
                if target > fmax:
                    break
                bfreq_k, a_k = nearest_bin_amplitude(xf, amp, target)
                sbr = float("nan")
                if include_sb and fr and sb_orders >= 1 and np.isfinite(a_k) and a_k > 0:
                    # n=1 sidebands only for SBR metric
                    _, a_m = nearest_bin_amplitude(xf, amp, max(0.0, target - fr))
                    _, a_p = nearest_bin_amplitude(xf, amp, target + fr)
                    if np.isfinite(a_m) and np.isfinite(a_p):
                        sbr = (a_m + a_p) / a_k if a_k else float("nan")
                out["tpf"].append({"k": k, "target_hz": target, "bin_hz": bfreq_k, "amp": a_k, "sbr_n1": sbr})
        return out

    # spectra cache for key‑freq computations
    spectra_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ch in selected_channels:
        ch_data = data.get(ch, {})
        signal = np.asarray(ch_data.get("Signal", []), dtype=float)
        if signal.size > 0 and fs > 0:
            xf, amp = compute_fft(signal, fs, apply_hann)
            spectra_cache[ch] = (xf, amp)
        else:
            spectra_cache[ch] = (np.array([]), np.array([]))

    keyfreq_by_channel: Dict[str, dict] = {}
    for ch in selected_channels:
        label = data.get(ch, {}).get("SignalName", ch)
        xf, amp = spectra_cache.get(ch, (np.array([]), np.array([])))
        keyfreq_by_channel[label] = compute_keyfreqs_for_channel(
            xf, amp, fr, ft, int(k_tpf), bool(include_sidebands), int(max_sideband_order)
        )

    # --------------------------------------------------
    # Memory (natural language) – EXPLANATION based on key frequencies
    # --------------------------------------------------
    st.subheader("Memory (natural text)")

    header_context_text = "; ".join([f"{k}={file_header[k]}" for k in file_header]) or "no header context"
    n_text = f"{fmt_float(n_rpm)} RPM" if n_rpm else "n/a"
    f0_text = f"{fmt_float(f_fund)} Hz" if f_fund else "n/a"
    fs_text = f"{fmt_float(fs)} Hz" if np.isfinite(fs) else "n/a"
    break_text = "YES" if broke else "NO"

    # Build channel-specific interpretations from key‑frequency amplitudes
    channel_summaries: List[str] = []
    for ch in selected_channels:
        label, unit, _ = harmonic_tables[ch]
        s = stats_by_ch.get(ch, {})
        rms = s.get("rms", np.nan)
        kf = keyfreq_by_channel.get(label, {})
        fr_amp = kf.get("fr", {}).get("amp", np.nan)
        fr_bin = kf.get("fr", {}).get("bin_hz", np.nan)
        tpf_list = kf.get("tpf", [])
        if tpf_list:
            # metrics: max TPF amp and mean SBR (n=1)
            max_tpf = max(tpf_list, key=lambda r: (r.get("amp") if np.isfinite(r.get("amp", np.nan)) else -1))
            mean_sbr = np.nan
            if any(np.isfinite(r.get("sbr_n1", np.nan)) for r in tpf_list):
                vals = [r.get("sbr_n1") for r in tpf_list if np.isfinite(r.get("sbr_n1", np.nan))]
                mean_sbr = float(np.mean(vals)) if len(vals) else np.nan
            summary = (
                f"**{label}**: spindle fr≈{fmt_float(fr_bin)} Hz has amplitude {fmt_float(fr_amp)}{(' ' + unit) if unit else ''}; "
                f"TPF harmonics peak at k={max_tpf.get('k')} (f≈{fmt_float(max_tpf.get('bin_hz'))} Hz) "
                f"with amp {fmt_float(max_tpf.get('amp'))}{(' ' + unit) if unit else ''}. "
                f"Primary sideband ratio (±fr) ≈ {fmt_float(mean_sbr)}."
            )
        else:
            summary = (
                f"**{label}**: spindle fr≈{fmt_float(fr_bin)} Hz amp {fmt_float(fr_amp)}{(' ' + unit) if unit else ''}; "
                "TPF harmonics not within spectrum range."
            )
        if np.isfinite(rms):
            summary += f" RMS ≈ {fmt_float(rms)}{(' ' + unit) if unit else ''}."
        channel_summaries.append(summary)

    # Overall qualitative cue (non-binding heuristic for narrative only)
    # Heuristic: if many TPF harmonics are strong and sidebands are pronounced, narrative highlights possible damage.
    def heuristic_break_signal(channel_kf: Dict[str, dict]) -> str:
        flags = 0
        for label, kf in channel_kf.items():
            fr_amp = kf.get("fr", {}).get("amp", np.nan)
            tpf_list = kf.get("tpf", [])
            strong_tpf = sum(1 for r in tpf_list if np.isfinite(r.get("amp", np.nan)) and r["amp"] > (fr_amp if np.isfinite(fr_amp) else 0))
            sbr_vals = [r.get("sbr_n1") for r in tpf_list if np.isfinite(r.get("sbr_n1", np.nan))]
            mean_sbr = (np.mean(sbr_vals) if sbr_vals else 0)
            if strong_tpf >= 3:
                flags += 1
            if mean_sbr and mean_sbr > 0.7:
                flags += 1
        if flags >= 2:
            return "Key‑frequency pattern shows strong TPF content and pronounced sidebands, which often accompanies tool damage or chipping."
        elif flags == 1:
            return "Key‑frequency content shows some TPF/sideband prominence; monitor for degradation."
        else:
            return "Key‑frequency content is modest; spectra are consistent with an intact tool during stable cutting."

    narrative_hint = heuristic_break_signal(keyfreq_by_channel)

    mem_text = (
        "Machine vibration snapshot — tool break label: "
        f"{break_text}. Spindle speed n = {n_text}, fundamental f₀ = {f0_text}, sampling fs = {fs_text}. "
        + (f"Key frequencies: spindle f_r={fmt_float(fr)} Hz" if fr else "")
        + (f", tooth‑passing f_t={fmt_float(ft)} Hz (Z={z_teeth}). " if ft else ". ")
        + f"File header context: {header_context_text}. "
        + narrative_hint + " "
        + " ".join(channel_summaries)
    )

    st.write(mem_text)

    # Memory payload (keeps full amplitudes for downstream use)
    memory_payload = {
        "type": "vibration_memory_text",
        "schema_version": 8,  # bumped for key‑freq explanation text
        "created_at": datetime.utcnow().isoformat() + "Z",
        "tool_break": broke,
        "n_rpm": n_rpm,
        "f0_hz": f_fund,
        "sample_frequency_hz": fs,
        "z_teeth": z_teeth,
        "fr_hz": fr,
        "ft_hz": ft,
        "file_header": file_header,
        "text": mem_text,
        "key_frequencies_by_channel": keyfreq_by_channel,
    }

    colmj, colmt = st.columns(2)
    with colmj:
        st.download_button(
            "⬇️ Download Memory (JSON)",
            data=json.dumps(memory_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="machine_vibration_memory_text.json",
            mime="application/json",
        )
    with colmt:
        st.download_button(
            "⬇️ Download Memory (TXT)",
            data=mem_text.encode("utf-8"),
            file_name="machine_vibration_memory.txt",
            mime="text/plain",
        )

    st.divider()

    # --------------------------------------------------
    # ML Training (Key‑freq only)
    # --------------------------------------------------
    st.subheader("ML Training (Key‑freq only)")
    st.caption(
        "Single input row using only amplitudes from key frequencies: spindle fr and TPF harmonics k·ft (with optional sideband ratio SBR at ±fr). Target is `break` (boolean) provided separately."
    )

    # Build a single feature row composed *only* of key‑frequency features
    feature_row = {}

    # Global context — optionally include fr and ft as numeric context features
    if np.isfinite(fr) if fr is not None else False:
        feature_row["global_fr_hz"] = float(fr)
    if np.isfinite(ft) if ft is not None else False:
        feature_row["global_ft_hz"] = float(ft)

    # Per‑channel key‑frequency features
    for ch in selected_channels:
        label = data.get(ch, {}).get("SignalName", ch)
        prefix = slug(label) or slug(ch)
        kf = keyfreq_by_channel.get(label, {})
        fr_amp = kf.get("fr", {}).get("amp", np.nan)
        fr_bin = kf.get("fr", {}).get("bin_hz", np.nan)
        feature_row[f"{prefix}_fr_amp"] = float(fr_amp) if np.isfinite(fr_amp) else None
        feature_row[f"{prefix}_fr_bin_hz"] = float(fr_bin) if np.isfinite(fr_bin) else None

        tpf_list = kf.get("tpf", [])
        for r in tpf_list:
            k_idx = int(r.get("k", 0))
            a = r.get("amp", np.nan)
            b = r.get("bin_hz", np.nan)
            sbr = r.get("sbr_n1", np.nan)
            feature_row[f"{prefix}_tpf_h{k_idx}_amp"] = float(a) if np.isfinite(a) else None
            feature_row[f"{prefix}_tpf_h{k_idx}_bin_hz"] = float(b) if np.isfinite(b) else None
            # Sideband ratio (n=1)
            feature_row[f"{prefix}_tpf_h{k_idx}_sbr"] = float(sbr) if np.isfinite(sbr) else None

        # Lightweight summary stats for learning stability (still key‑freq derived)
        if tpf_list:
            amps = [r.get("amp", np.nan) for r in tpf_list]
            sbrs = [r.get("sbr_n1", np.nan) for r in tpf_list]
            if any(np.isfinite(amps)):
                feature_row[f"{prefix}_tpf_amp_max"] = float(np.nanmax(amps))
                feature_row[f"{prefix}_tpf_amp_mean"] = float(np.nanmean(amps))
            if any(np.isfinite(sbrs)):
                feature_row[f"{prefix}_tpf_sbr_mean"] = float(np.nanmean(sbrs))

    # One‑row DataFrame for editing; target kept separately
    df_feat = pd.DataFrame([feature_row])

    col_left, col_right = st.columns([3, 1])
    with col_left:
        edited_df_feat = st.data_editor(
            df_feat,
            use_container_width=True,
            num_rows="fixed",
            column_config={c: st.column_config.NumberColumn(format="%.6g") for c in df_feat.columns},
        )
    with col_right:
        st.metric("Target: break", "YES" if broke else "NO")
        st.caption("Provided separately from features")

    # Export JSON & CSV
    ebm_payload = {
        "schema_version": 5,  # bumped — key‑freq‑only features
        "created_at": datetime.utcnow().isoformat() + "Z",
        "task": "tool_breakage_detection",
        "target": {"break": broke},
        "features": edited_df_feat.to_dict(orient="records")[0],
    }

    colj, colc = st.columns(2)
    with colj:
        st.download_button(
            "⬇️ Download ML input (JSON)",
            data=json.dumps(ebm_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="machine_vibration_keyfreq_input.json",
            mime="application/json",
        )
    with colc:
        st.download_button(
            "⬇️ Download ML input (CSV)",
            data=edited_df_feat.to_csv(index=False).encode("utf-8"),
            file_name="machine_vibration_keyfreq_input.csv",
            mime="text/csv",
        )

    st.divider()

    # --------------------------------------------------
    # Per-channel plots (time, freq, Key Frequencies)
    # --------------------------------------------------
    if selected_channels:
        tabs = st.tabs([harmonic_tables[ch][0] for ch in selected_channels])
        for tab, ch in zip(tabs, selected_channels):
            with tab:
                label, unit, _ = harmonic_tables[ch]
                ch_data = data.get(ch, {})
                signal = np.asarray(ch_data.get("Signal", []), dtype=float)
                if signal.size == 0:
                    st.error("No signal data found for this channel.")
                    continue

                N = len(signal)
                t = np.arange(N) / fs
                xf, amp = compute_fft(signal, fs, apply_hann)

                st.markdown(
                    f"**Channel:** `{ch}`  \n"
                    f"**Name:** **{label}**  \n"
                    f"**Samples:** {N}  \n"
                    f"**fs:** {fs:g} Hz  \n"
                    f"**Bin Δf:** {fmt_float(xf[1]-xf[0] if len(xf)>1 else float('nan'))} Hz"
                )

                t_tab, f_tab, key_tab = st.tabs(["Time Domain", "Frequency Domain", "Key Frequencies"])  # improved
                with t_tab:
                    fig = go.Figure(go.Scatter(x=t, y=signal, mode="lines", name=label))
                    fig.update_layout(xaxis_title="Time [s]", yaxis_title=unit or "Amplitude")
                    st.plotly_chart(fig, use_container_width=True)

                with f_tab:
                    fig = go.Figure(go.Scatter(x=xf, y=amp, mode="lines", name=label))
                    if f_fund:
                        for f_h in np.arange(1, int(harmonics_count) + 1) * (f_fund or 0):
                            if f_h <= (xf[-1] if len(xf) else 0):
                                fig.add_vline(x=f_h, line_width=1, line_dash="dash", opacity=0.35)
                    fig.update_layout(xaxis_title="Frequency [Hz]", yaxis_title="Amplitude")
                    st.plotly_chart(fig, use_container_width=True)

                # --- Key Frequencies tab ---
                with key_tab:
                    if fr is None and ft is None:
                        st.info("Key Frequencies require 'n' (RPM) and 'z' (number of teeth). Provide these in the JSON.")
                    else:
                        # Base spectrum
                        figkf = go.Figure()
                        figkf.add_trace(go.Scatter(x=xf, y=amp, mode="lines", name=label, opacity=0.45))

                        rows = []
                        x_spindle, y_spindle, txt_spindle = [], [], []
                        x_tpf, y_tpf, txt_tpf = [], [], []
                        x_sb, y_sb, txt_sb = [], [], []

                        # Helper to maybe annotate
                        def _maybe_text(a: float, prefix: str) -> str:
                            if not annotate_amplitudes or not np.isfinite(a) or a < float(annotation_min_amp):
                                return ""
                            return f"{prefix}{fmt_float(a, sig=4)}"

                        fmax = xf[-1] if len(xf) else 0

                        # Spindle line & marker
                        if fr:
                            bfreq_fr, a_fr = nearest_bin_amplitude(xf, amp, fr)
                            rows.append({"Type": "Spindle (fr)", "k": 1, "Target f [Hz]": fr, "Bin f [Hz]": bfreq_fr, "Amplitude": a_fr})
                            if np.isfinite(bfreq_fr) and np.isfinite(a_fr):
                                figkf.add_vline(x=bfreq_fr, line_width=2, line_dash="dot", opacity=0.7)
                                x_spindle.append(bfreq_fr); y_spindle.append(a_fr); txt_spindle.append(_maybe_text(a_fr, "A= "))

                        # TPF harmonics and sidebands
                        if ft:
                            for k in range(1, int(k_tpf) + 1):
                                target = k * ft
                                if target > fmax:
                                    break
                                bfreq_k, a_k = nearest_bin_amplitude(xf, amp, target)
                                rows.append({"Type": "TPF", "k": k, "Target f [Hz]": target, "Bin f [Hz]": bfreq_k, "Amplitude": a_k})
                                if np.isfinite(bfreq_k):
                                    figkf.add_vline(x=bfreq_k, line_width=1, line_dash="dash", opacity=0.6)
                                    x_tpf.append(bfreq_k); y_tpf.append(a_k); txt_tpf.append(_maybe_text(a_k, "A= "))
                                # multiple sideband orders: ± n·fr
                                if include_sidebands and fr and int(max_sideband_order) > 0:
                                    for n_sb in range(1, int(max_sideband_order) + 1):
                                        f_minus = max(0.0, target - n_sb * fr)
                                        f_plus = target + n_sb * fr
                                        if f_minus <= fmax:
                                            bfreq_m, a_m = nearest_bin_amplitude(xf, amp, f_minus)
                                            rows.append({"Type": f"Sideband -{n_sb}", "k": k, "Target f [Hz]": f_minus, "Bin f [Hz]": bfreq_m, "Amplitude": a_m})
                                            if np.isfinite(bfreq_m):
                                                figkf.add_vline(x=bfreq_m, line_width=1, line_dash="dot", opacity=0.35)
                                                x_sb.append(bfreq_m); y_sb.append(a_m); txt_sb.append(_maybe_text(a_m, f"A= "))
                                        if f_plus <= fmax:
                                            bfreq_p, a_p = nearest_bin_amplitude(xf, amp, f_plus)
                                            rows.append({"Type": f"Sideband +{n_sb}", "k": k, "Target f [Hz]": f_plus, "Bin f [Hz]": bfreq_p, "Amplitude": a_p})
                                            if np.isfinite(bfreq_p):
                                                figkf.add_vline(x=bfreq_p, line_width=1, line_dash="dot", opacity=0.35)
                                                x_sb.append(bfreq_p); y_sb.append(a_p); txt_sb.append(_maybe_text(a_p, f"A= "))

                        # Add markers with optional labels
                        if x_spindle:
                            figkf.add_trace(
                                go.Scatter(
                                    x=x_spindle, y=y_spindle, mode="markers+text" if annotate_amplitudes else "markers",
                                    text=txt_spindle if annotate_amplitudes else None, textposition="top center",
                                    name="Spindle fr", marker_symbol="diamond", marker_size=10,
                                )
                            )
                        if x_tpf:
                            figkf.add_trace(
                                go.Scatter(
                                    x=x_tpf, y=y_tpf, mode="markers+text" if annotate_amplitudes else "markers",
                                    text=txt_tpf if annotate_amplitudes else None, textposition="top center",
                                    name="TPF harmonics k·ft", marker_symbol="x", marker_size=9,
                                )
                            )
                        if x_sb:
                            figkf.add_trace(
                                go.Scatter(
                                    x=x_sb, y=y_sb, mode="markers+text" if annotate_amplitudes else "markers",
                                    text=txt_sb if annotate_amplitudes else None, textposition="top center",
                                    name="Sidebands ± n·fr", marker_size=8,
                                )
                            )

                        figkf.update_layout(xaxis_title="Frequency [Hz]", yaxis_title=f"Amplitude{(' [' + unit + ']') if unit else ''}")
                        st.plotly_chart(figkf, use_container_width=True)

                        # Table of key frequencies
                        if rows:
                            df_kf = pd.DataFrame(rows)
                            st.dataframe(df_kf, use_container_width=True)
                            st.download_button(
                                label="⬇️ Download Key Frequencies (CSV)",
                                data=df_kf.to_csv(index=False).encode("utf-8"),
                                file_name=f"key_frequencies_{slug(label)}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.caption("No key frequency data available for this channel.")
else:
    st.info("Please upload a JSON file to get started.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© Sagar Sen 2025 — Machine Vibration Analysis App")



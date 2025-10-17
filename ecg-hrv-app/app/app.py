import io
from typing import Dict, List, Tuple

import altair as alt
import neurokit2 as nk
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "ECG HRV Feature Explorer"
SUPPORTED_MODELS = [
    "Neural Network",
    "SVM",
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "Naive Bayes",
]


def read_ecg_csv(file_buffer: io.BytesIO | io.StringIO) -> pd.DataFrame:
    """Read an ECG CSV/TSV file with tolerant parsing."""
    try:
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer, sep=None, engine="python")
    except Exception:
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer, sep=",", engine="python")

    cols = {c.strip().lower(): c for c in df.columns}
    ecg_col = None
    label_col = None
    for key, original in cols.items():
        if key in {"ecg", "signal", "ecg_signal"}:
            ecg_col = original
        if key in {"label", "target", "y"}:
            label_col = original

    if ecg_col is None:
        raise ValueError("Could not find ECG column. Expected 'ECG' (case-insensitive).")

    result = pd.DataFrame()
    result["ECG"] = pd.to_numeric(df[ecg_col], errors="coerce")
    if label_col is not None:
        result["Label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    else:
        result["Label"] = 0

    result = result.dropna(subset=["ECG"]).reset_index(drop=True)
    return result


def process_hrv_time_domain(ecg_values: np.ndarray, sampling_rate: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute HRV time-domain features using NeuroKit2."""
    signals, info = nk.ecg_process(ecg_values, sampling_rate=sampling_rate)
    peaks = info.get("ECG_R_Peaks")
    if peaks is None or len(peaks) < 3:
        raise ValueError("Unable to detect sufficient R-peaks for HRV computation.")

    hrv_time: pd.DataFrame = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
    metrics = hrv_time.iloc[0].to_dict()

    preferred_order: List[str] = [
        "HRV_MeanNN",
        "HRV_SDNN",
        "HRV_SDANN1",
        "HRV_SDNN_Index",
        "HRV_SDSD",
        "HRV_RMSSD",
        "HRV_pNN20",
        "HRV_pNN50",
        "HRV_HTI",
        "HRV_TINN",
        "HRV_Triang",
        "HRV_CVNN",
        "HRV_CVSD",
        "HRV_MedianNN",
        "HRV_MadNN",
        "HRV_MCVNN",
        "HRV_IQRNN",
        "HRV_Eliahou",
        "HRV_Shannon",
    ]
    available = [k for k in preferred_order if k in metrics]
    if len(available) < 19:
        others = [k for k in metrics.keys() if k not in available]
        available.extend(others[: 19 - len(available)])

    selected = {k: float(metrics[k]) for k in available[:19]}
    return signals, selected


def dummy_predict(selected_model: str, features: Dict[str, float]) -> Dict[str, float]:
    """Deterministic dummy prediction based on simple heuristic."""
    risk_score = 0.5
    if "HRV_RMSSD" in features and "HRV_SDNN" in features:
        den = max(features["HRV_SDNN"], 1e-6)
        ratio = features["HRV_RMSSD"] / den
        risk_score = float(np.clip(0.5 + 0.3 * (ratio - 1.0), 0.0, 1.0))
    else:
        vals = np.array(list(features.values()), dtype=float)
        if len(vals) > 0:
            risk_score = float(np.clip((vals - vals.min()).mean() / (vals.ptp() + 1e-6), 0.0, 1.0))
    return {"stress_probability": risk_score, "predicted_label": int(risk_score >= 0.5)}


def render_feature_chart(features: Dict[str, float]) -> alt.Chart:
    data = pd.DataFrame({"Feature": list(features.keys()), "Value": list(features.values())})
    data = data.sort_values("Feature")
    chart = (
        alt.Chart(data)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("Value:Q", title="Feature Value"),
            y=alt.Y("Feature:N", sort="-x", title="HRV Time-Domain Feature"),
            tooltip=["Feature", alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(height=28 * len(data), width=720)
    )
    return chart


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("Upload a 30-second ECG at 750 Hz (22500 samples) with columns `ECG` and `Label` (0/1).")

    with st.sidebar:
        st.header("Settings")
        sampling_rate = st.number_input("Sampling rate (Hz)", min_value=100, max_value=4000, value=750, step=50)
        model_choice = st.selectbox("Select model", SUPPORTED_MODELS, index=0)
        st.caption("Models are dummy placeholders for now; bring your trained models later.")

    uploaded = st.file_uploader("Upload ECG CSV/TSV", type=["csv", "tsv", "txt"])

    if uploaded is None:
        st.info("Awaiting file upload. Example format: columns `ECG` and `Label`.")
        st.stop()

    try:
        df = read_ecg_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"Rows: {len(df):,}")

    if len(df) < sampling_rate * 5:
        st.warning("Very short signal detected (<5s). HRV metrics may be unreliable.")

    with st.spinner("Processing ECG and extracting HRV features..."):
        try:
            signals, features = process_hrv_time_domain(df["ECG"].to_numpy(dtype=float), sampling_rate=int(sampling_rate))
        except Exception as e:
            st.error(f"HRV extraction failed: {e}")
            st.stop()

    st.subheader("HRV Time-Domain Features (19)")
    chart = render_feature_chart(features)
    st.altair_chart(chart, use_container_width=True)

    feat_df = pd.DataFrame([features]).T.reset_index()
    feat_df.columns = ["Feature", "Value"]
    st.dataframe(feat_df, use_container_width=True)

    features_csv = feat_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download features CSV", data=features_csv, file_name="hrv_time_features.csv", mime="text/csv")

    st.subheader("Model Prediction (Dummy)")
    pred = dummy_predict(model_choice, features)
    prob = pred["stress_probability"]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stress probability", f"{prob:.2%}")
    with col2:
        st.metric("Predicted label", "Stress" if pred["predicted_label"] == 1 else "No stress")
    st.caption("This is a placeholder. When your models are ready, we will load and run them here.")


if __name__ == "__main__":
    main()

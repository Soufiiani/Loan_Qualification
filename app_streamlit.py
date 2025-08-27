
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import traceback

# Ensure globals exist even if loading fails
model, schema = None, None

st.set_page_config(page_title="Loan Status Predictor", page_icon="💳", layout="centered")

st.title("💳 Loan Status Predictor")
st.write("Enter applicant and loan details to predict the **Loan Status**.")

@st.cache_resource
def load_artifacts():
    """
    Load model & schema without calling st.stop() inside the cached function.
    We search several candidate directories and raise RuntimeError on failure.
    """
    from pathlib import Path
    import json, joblib

    # Candidates: script dir, its parent, current working dir, its parent
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [script_dir, script_dir.parent, cwd, cwd.parent]

    tried = []
    last_exc = None
    for base in candidates:
        model_path = base / "loan_pipeline.pkl"
        schema_path = base / "schema.json"
        tried.append(f"{base} -> ({model_path.exists()}, {schema_path.exists()})")
        if model_path.exists() and schema_path.exists():
            try:
                model = joblib.load(str(model_path))
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                return model, schema
            except Exception as e:
                last_exc = e

    details = "; ".join(tried)
    if last_exc is not None:
        raise RuntimeError(f"Found files but failed to load: {last_exc}. Tried: {details}")
    raise RuntimeError(f"Required files not found in expected locations. Tried: {details}")

def build_default_row(schema):
    row = {}
    for c in schema["categorical_features"]:
        row[c] = schema["defaults"].get(c, "")
    for c in schema["numeric_features"]:
        v = schema["defaults"].get(c, None)
        if v is None or (isinstance(v,float) and (np.isnan(v) or not np.isfinite(v))):
            r = schema["ranges"].get(c, {})
            v = float(r.get("min", 0.0))
        row[c] = float(v)
    return row

def validate_inputs(schema, inputs):
    """Ensure numeric fields are floats and catch NaNs/Infs early. Returns (clean_inputs, warnings)."""
    warnings = []
    clean = {}
    # Categorical
    for c in schema.get("categorical_features", []):
        v = inputs.get(c, "")
        if v is None:
            v = ""
        clean[c] = str(v)
    # Numeric
    for c in schema.get("numeric_features", []):
        v = inputs.get(c, None)
        try:
            v = float(v)
            if not np.isfinite(v):
                warnings.append(f"'{c}' was non-finite; replaced with default.")
                r = schema["ranges"].get(c, {})
                v = float(r.get("min", 0.0))
        except Exception:
            warnings.append(f"'{c}' could not be parsed as a number; using default.")
            r = schema["ranges"].get(c, {})
            v = float(r.get("min", 0.0))
        clean[c] = v
    return clean, warnings

try:
    model, schema = load_artifacts()
except Exception as e:
    st.error(f"Artifact loading failed: {e}")
    st.info("Make sure 'loan_pipeline.pkl' and 'schema.json' are in the same folder as 'app_streamlit.py' or one level above it, or run Streamlit from that folder.")
    st.stop()

with st.expander("About this app", expanded=False):
    tgt = "Loan_Status"
    if schema is not None and isinstance(schema, dict) and "target_col" in schema:
        tgt = schema["target_col"]
    st.markdown(f"""
- **Model**: RandomForestClassifier inside a scikit-learn Pipeline.
- **Preprocessing**: Numeric median imputation; categorical most-frequent imputation + one-hot encoding.
- **Target column**: `{tgt}`
""")


st.header("Input data")

# Allow users to start with a safe sample row
col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("🧪 Use sample values"):
        st.session_state["_prefill"] = build_default_row(schema)
with col_b:
    if st.button("🧹 Clear form"):
        st.session_state["_prefill"] = {}

prefill = st.session_state.get("_prefill", {})

with st.form("input_form"):
    inputs = {}

    # Categorical features
    if schema["categorical_features"]:
        st.subheader("Categorical")
        for c in schema["categorical_features"]:
            opts = schema["categories"].get(c, [])
            default = prefill.get(c, schema["defaults"].get(c, ""))
            if len(opts) == 0:
                val = st.text_input(c, value=str(default) if default is not None else "")
            else:
                # make sure default is first option
                if str(default) not in opts and default is not None:
                    opts = [str(default)] + [o for o in opts if o != str(default)]
                idx = 0 if len(opts) > 0 else None
                val = st.selectbox(c, options=opts, index=idx)
            inputs[c] = val

    # Numeric features
    if schema["numeric_features"]:
        st.subheader("Numeric")
        for c in schema["numeric_features"]:
            r = schema["ranges"].get(c, {})
            mn = float(r.get("min", 0.0))
            mx = float(r.get("max", mn + 1.0))
            default = prefill.get(c, schema["defaults"].get(c, mn))
            try:
                default = float(default)
                if not np.isfinite(default):
                    default = (mn + mx) / 2.0
            except Exception:
                default = (mn + mx) / 2.0
            step = float((mx - mn)/100.0) if mx > mn else 1.0
            val = st.number_input(c, value=default, min_value=mn, max_value=mx, step=step)
            inputs[c] = val

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    # Validate and predict
    X_input_raw = pd.DataFrame([inputs])
    X_input, warns = validate_inputs(schema, inputs)
    X_input = pd.DataFrame([X_input])
    if warns:
        st.info("Input notes:\n- " + "\n- ".join(warns))
    try:
        pred = model.predict(X_input)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_input)[0]
                classes = list(model.classes_)
                if pred in classes:
                    idx = classes.index(pred)
                    confidence = float(proba[idx])
            except Exception:
                pass

        if confidence is not None:
            st.success(f"**Prediction: {pred}**  ·  Confidence: {confidence:.2%}")
        else:
            st.success(f"**Prediction: {pred}**")

        with st.expander("Show raw & cleaned inputs"):
            st.caption("Raw form values:")
            st.dataframe(X_input_raw)
            st.caption("Cleaned values used for prediction:")
            st.dataframe(X_input)

    except Exception as e:
        st.error(f"Prediction failed with error: {e.__class__.__name__}: {e}")
        with st.expander("Debug details"):
            st.code("\n".join(traceback.format_exc().splitlines()[-8:]))  # tail of traceback
        st.info("Tip: Use **🧪 Use sample values** to verify the model runs end-to-end.")

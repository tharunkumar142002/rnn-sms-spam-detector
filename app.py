
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from pathlib import Path
from datetime import datetime


# ==========================================
# STEP 1: PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PulseGuard AI | SMS Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# STEP 2: CUSTOM WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #080d1c 0%,
        #111a35 55%,
        #161232 100%
    );
    color: #f0f4ff;
}

h1, h2, h3 {
    color: #f0f4ff !important;
}

[data-testid="stSidebar"] {
    background-color: #10172b;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid #4c68bf;
    background: linear-gradient(
        90deg, #3655d6, #7044cd
    );
    color: white;
    font-weight: 600;
    min-height: 44px;
    transition: 0.2s;
}

div.stButton > button:hover {
    border-color: #88a6ff;
    color: white;
    box-shadow: 0 0 15px #405b9d;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(
        110deg, #173775, #402d80
    );
    margin-bottom: 25px;
    border: 1px solid #586ac0;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.hero-text {
    font-size: 17px;
    color: #d7e3ff;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# STEP 3: LOAD TRAINED RNN MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "sms_rnn.keras"


@st.cache_resource(show_spinner="Loading AI model...")
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


if not MODEL_PATH.exists():

    st.error(
        "Model file not found. "
        "Please check models/sms_rnn.keras"
    )

    st.stop()


model = load_model()


# ==========================================
# STEP 4: SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "sms_message" not in st.session_state:
    st.session_state.sms_message = ""

if "last_score" not in st.session_state:
    st.session_state.last_score = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# ==========================================
# STEP 5: PREDICTION FUNCTION
# ==========================================

def predict_sms(messages):

    # Convert messages into a string tensor
    input_tensor = tf.constant(
        messages,
        dtype=tf.string
    )

    # Model already includes text vectorization
    predictions = model.predict(
        input_tensor,
        batch_size=32,
        verbose=0
    )

    # Return spam probabilities
    return predictions.reshape(-1)


def choose_sample(message):

    st.session_state.sms_message = message
    st.session_state.last_score = None


# ==========================================
# STEP 6: SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🛡️ PulseGuard AI")

    st.caption("Neural SMS Security Dashboard")

    st.divider()

    st.subheader("⚙️ Detection Settings")

    threshold = st.slider(
        "Spam Detection Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.info(
        f"Messages with a spam score of "
        f"{threshold * 100:.0f}% or higher "
        f"are classified as Spam."
    )

    st.divider()

    st.subheader("🧠 AI Model")

    st.write("Architecture: SimpleRNN")
    st.write("Embedding dimensions: 32")
    st.write("Hidden units: 64")
    st.write("Framework: TensorFlow/Keras")

    st.divider()

    st.success("AI Model Loaded")

    st.caption(
        "Educational AI demonstration. "
        "Predictions are not guaranteed."
    )


# ==========================================
# STEP 7: WEBSITE HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🛡️ PulseGuard AI
    </div>
    <div class="hero-text">
        Intelligent SMS Spam Detection
        powered by Recurrent Neural Networks
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# STEP 8: NAVIGATION TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Detector",
    "📁 Bulk Analyzer",
    "📊 Analytics",
    "🧠 Model Insights"
])


# ==========================================
# TAB 1: LIVE SMS DETECTOR
# ==========================================

with tab1:

    st.header("🔍 Analyze an SMS Message")

    st.write(
        "Enter a message or select one "
        "of the sample messages below."
    )

    st.subheader("⚡ Try Example Messages")

    sample_messages = [
        "Congratulations! You won a free prize. Claim now!",
        "Hi, are you coming to college tomorrow?",
        "URGENT! You have won cash. Click here to claim!",
        "Hello, please call me when you are free."
    ]

    cols = st.columns(4)

    labels = [
        "🎁 Prize Message",
        "💬 Normal Chat",
        "🚨 Urgent Offer",
        "📞 Friendly SMS"
    ]

    for i, col in enumerate(cols):

        with col:

            st.button(
                labels[i],
                key=f"sample_{i}",
                use_container_width=True,
                on_click=choose_sample,
                args=(sample_messages[i],)
            )

    st.divider()

    message = st.text_area(
        "Enter SMS Text",
        key="sms_message",
        height=150,
        placeholder="Type your SMS message here..."
    )

    st.caption(f"Message length: {len(message)} characters")

    analyze = st.button(
        "🚀 Analyze Message",
        key="analyze_sms",
        use_container_width=True,
        type="primary"
    )

    if analyze:

        if not message.strip():

            st.warning("Please enter an SMS message.")

            st.session_state.last_score = None

        else:

            with st.spinner("AI is analyzing your message..."):

                score = float(
                    predict_sms([message])[0]
                )

            st.session_state.last_score = score

            classification = (
                "Spam" if score >= threshold else "Ham"
            )

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": message,
                "classification": classification,
                "spam_score_pct": round(score * 100, 2)
            })

    # Display last prediction
    if st.session_state.last_score is not None:

        spam_score = st.session_state.last_score

        is_spam = spam_score >= threshold

        st.divider()

        st.subheader("📊 AI Prediction Results")

        col1, col2 = st.columns([1, 1])

        with col1:

            if is_spam:

                st.error("🚨 SPAM DETECTED")

                st.write(
                    "This message contains patterns "
                    "the RNN associates with spam."
                )

            else:

                st.success("✅ NORMAL MESSAGE (HAM)")

                st.write(
                    "This message has been classified "
                    "as a normal SMS."
                )

            st.metric(
                "Spam Score",
                f"{spam_score * 100:.2f}%"
            )

            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%"
            )

        with col2:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=spam_score * 100,
                    number={"suffix": "%"},
                    title={"text": "Spam Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#7c8cff"},
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#187b63"
                            },
                            {
                                "range": [30, 70],
                                "color": "#9d772e"
                            },
                            {
                                "range": [70, 100],
                                "color": "#9b334e"
                            }
                        ],
                        "threshold": {
                            "line": {
                                "color": "white",
                                "width": 4
                            },
                            "value": threshold * 100
                        }
                    }
                )
            )

            gauge.update_layout(
                height=300,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=10)
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        st.caption(
            "This score is the RNN's sigmoid output, "
            "not a calibrated guarantee of correctness."
        )


# ==========================================
# TAB 2: BULK CSV ANALYZER
# ==========================================

with tab2:

    st.header("📁 Bulk SMS Analyzer")

    st.write(
        "Upload a CSV containing a column "
        "named 'message' to classify multiple SMS messages."
    )

    st.info("Maximum 1,000 messages per upload.")

    uploaded_file = st.file_uploader(
        "Upload SMS CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            st.write("Uploaded Dataset Preview")

            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )

            message_column = next(
                (
                    col for col in uploaded_df.columns
                    if col.strip().lower() == "message"
                ),
                None
            )

            if message_column is None:

                st.error(
                    "CSV must contain a 'message' column."
                )

            else:

                if st.button(
                    "🚀 Analyze All Messages",
                    use_container_width=True
                ):

                    texts = (
                        uploaded_df[message_column]
                        .dropna()
                        .astype(str)
                    )

                    texts = texts[
                        texts.str.strip() != ""
                    ]

                    if len(texts) == 0:

                        st.warning("No valid messages found.")

                    elif len(texts) > 1000:

                        st.warning(
                            "Please upload at most 1,000 messages."
                        )

                    else:

                        with st.spinner(
                            "Analyzing uploaded messages..."
                        ):

                            scores = predict_sms(
                                texts.tolist()
                            )

                        results = pd.DataFrame({
                            "message": texts.to_list(),
                            "spam_probability": scores,
                            "prediction": np.where(
                                scores >= threshold,
                                "Spam",
                                "Ham"
                            )
                        })

                        st.session_state.batch_results = results

        except Exception as error:

            st.error(
                f"Unable to read CSV: {error}"
            )

    if st.session_state.batch_results is not None:

        results = st.session_state.batch_results

        st.success("Bulk analysis completed!")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Messages",
            len(results)
        )

        col2.metric(
            "Spam Detected",
            int((results["prediction"] == "Spam").sum())
        )

        col3.metric(
            "Ham Detected",
            int((results["prediction"] == "Ham").sum())
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_data,
            file_name="sms_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================
# TAB 3: ANALYTICS DASHBOARD
# ==========================================

with tab3:

    st.header("📊 Prediction Analytics")

    history = st.session_state.history

    if len(history) == 0:

        st.info(
            "No predictions yet. Analyze SMS messages "
            "to see analytics here."
        )

    else:

        history_df = pd.DataFrame(history)

        total = len(history_df)

        spam_count = int(
            (history_df["classification"] == "Spam").sum()
        )

        ham_count = total - spam_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", total)
        col2.metric("Spam Predictions", spam_count)
        col3.metric("Ham Predictions", ham_count)

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("Classification Distribution")

            pie = px.pie(
                names=["Spam", "Ham"],
                values=[spam_count, ham_count],
                hole=0.55,
                color_discrete_sequence=[
                    "#ff657b",
                    "#33cc9b"
                ]
            )

            pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        with right:

            st.subheader("Spam Score Trend")

            history_df["prediction_number"] = range(
                1, len(history_df) + 1
            )

            trend = px.line(
                history_df,
                x="prediction_number",
                y="spam_score_pct",
                markers=True,
                labels={
                    "prediction_number": "Prediction Number",
                    "spam_score_pct": "Spam Score (%)"
                }
            )

            trend.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                trend,
                use_container_width=True
            )

        st.subheader("📜 Prediction History")

        st.dataframe(
            history_df[
                [
                    "time",
                    "message",
                    "classification",
                    "spam_score_pct"
                ]
            ],
            use_container_width=True
        )

        download_csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Session History",
            data=download_csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Prediction History"):

            st.session_state.history = []
            st.session_state.last_score = None
            st.rerun()

    st.caption(
        "History is stored in this Streamlit session "
        "and is not a permanent database."
    )


# ==========================================
# TAB 4: MODEL INSIGHTS
# ==========================================

with tab4:

    st.header("🧠 About Our RNN Model")

    st.markdown("""
    ### Model Architecture

    **Input SMS**
    ↓

    **TextVectorization**
    - Vocabulary learned from training data
    - Sequence length: 50

    ↓

    **Embedding Layer**
    - 32 dimensions per token

    ↓

    **SimpleRNN**
    - 64 hidden units
    - Tanh activation

    ↓

    **Dense Layer**
    - 1 neuron
    - Sigmoid activation

    ↓

    **Prediction**
    - 0 = Ham
    - 1 = Spam
    """)

    st.divider()

    st.subheader("📈 Test Evaluation Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "~99%")
    col2.metric("Spam Precision", "100%")
    col3.metric("Spam Recall", "91%")
    col4.metric("Spam F1-Score", "95%")

    st.caption(
        "These values are rounded from the previously "
        "generated classification report on 517 test messages."
    )

    st.divider()

    st.subheader("⚙️ Framework")

    st.write("Python")
    st.write("TensorFlow / Keras")
    st.write("Simple Recurrent Neural Network")
    st.write("Streamlit")
    st.write("Plotly")

    st.warning(
        "This model is intended for learning and "
        "demonstration, not as a production security system."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "PulseGuard AI | Built with Python, "
    "TensorFlow, SimpleRNN, Streamlit and Plotly"
)

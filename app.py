import os
import re

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Analysis Dashboard: Hate Speech Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f8fb;
        }

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e8eaf0;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        h1, h2, h3 {
            color: #1f2937;
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: 2.1rem;
            font-weight: 750;
            margin-bottom: 0.35rem;
        }

        h2 {
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 0.75rem;
        }

        h3 {
            font-size: 1.1rem;
            font-weight: 650;
        }

        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e6e8ef;
            border-radius: 16px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #6b7280;
            font-size: 0.9rem;
        }

        div[data-testid="stMetricValue"] {
            color: #111827;
            font-size: 1.55rem;
            font-weight: 750;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 10px;
            border: 1px solid #e6e8ef;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.04);
        }

        .section-card {
            background-color: #ffffff;
            border: 1px solid #e6e8ef;
            border-radius: 18px;
            padding: 1.35rem 1.45rem;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.04);
            margin-bottom: 1.1rem;
        }

        .dashboard-subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 0.85rem;
            line-height: 1.6;
        }

        .meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: 0.4rem 0 1.4rem 0;
        }

        .meta-pill {
            background-color: #ffffff;
            color: #374151;
            border: 1px solid #e5e7eb;
            border-radius: 999px;
            padding: 0.55rem 0.9rem;
            font-size: 0.92rem;
            box-shadow: 0 6px 18px rgba(31, 41, 55, 0.04);
        }

        .meta-label {
            color: #6b7280;
            font-weight: 500;
        }

        .meta-value {
            color: #111827;
            font-weight: 650;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .sidebar-title {
            color: #111827;
            font-size: 1.15rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .sidebar-caption {
            color: #6b7280;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }

        .footer {
            color: #6b7280;
            text-align: center;
            font-size: 0.88rem;
            padding: 1rem 0 0.25rem 0;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid #d1d5db;
            background-color: #111827;
            color: white;
            font-weight: 600;
            padding: 0.55rem 1rem;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            border-color: #111827;
            background-color: #374151;
            color: white;
        }

        [data-testid="stFileUploader"] {
            background-color: #ffffff;
            border: 1px dashed #cbd5e1;
            border-radius: 16px;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


LEX_NAME = "Lexicon Based Custom Rule dengan Scoring Threshold"
ZSC_NAME = "Zero Shot Classification Labeling"


@st.cache_data
def get_model_data():
    data = {
        "IndoBERT": {
            LEX_NAME: {
                "Split": pd.DataFrame({
                    "Split Scheme": ["70:30", "80:20", "90:10"],
                    "Accuracy": [0.9874, 0.9829, 0.9848],
                    "Precision (M)": [0.9521, 0.9304, 0.9396],
                    "Recall (M)": [0.9819, 0.9863, 0.9832],
                    "F1-Macro": [0.9663, 0.9561, 0.9600],
                    "ROC-AUC": [0.9981, 0.9969, 0.9996],
                }),
                "KFold_Detail": pd.DataFrame({
                    "Fold": [1, 2, 3, 4, 5],
                    "Accuracy": [0.9923, 0.9885, 0.9846, 0.9840, 0.9827],
                    "Precision (M)": [0.9661, 0.9639, 0.9342, 0.9356, 0.9380],
                    "Recall (M)": [0.9912, 0.9781, 0.9870, 0.9910, 0.9738],
                    "F1-Macro": [0.9783, 0.9709, 0.9587, 0.9611, 0.9550],
                    "ROC-AUC": [0.9995, 0.9984, 0.9972, 0.9990, 0.9860],
                }),
                "KFold_Summary": pd.DataFrame({
                    "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "ROC-AUC"],
                    "Mean": [0.9864, 0.9476, 0.9842, 0.9648, 0.9960],
                    "Std Dev": [0.0039, 0.0160, 0.0079, 0.0096, 0.0057],
                }),
            },
            ZSC_NAME: {
                "Split": pd.DataFrame({
                    "Split Scheme": ["70:30", "80:20", "90:10"],
                    "Accuracy": [0.8895, 0.8795, 0.8672],
                    "Precision (M)": [0.8236, 0.8085, 0.7570],
                    "Recall (M)": [0.7681, 0.7614, 0.8063],
                    "F1-Macro": [0.7911, 0.7816, 0.7781],
                    "ROC-AUC": [0.8864, 0.9067, 0.8928],
                }),
                "KFold_Detail": pd.DataFrame({
                    "Fold": [1, 2, 3, 4, 5],
                    "Accuracy": [0.8749, 0.8822, 0.8791, 0.8899, 0.8739],
                    "Precision (M)": [0.7786, 0.7974, 0.7913, 0.8055, 0.7906],
                    "Recall (M)": [0.7601, 0.8012, 0.7874, 0.7781, 0.7918],
                    "F1-Macro": [0.7688, 0.7993, 0.7893, 0.7907, 0.7912],
                    "ROC-AUC": [0.8942, 0.9200, 0.9207, 0.9114, 0.9128],
                }),
                "KFold_Summary": pd.DataFrame({
                    "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Macro", "ROC-AUC"],
                    "Mean": [0.8800, 0.7927, 0.7837, 0.7878, 0.9118],
                    "Std Dev": [0.0064, 0.0101, 0.0152, 0.0112, 0.0108],
                }),
            },
        },
        "BiLSTM": {
            LEX_NAME: {
                "Split": pd.DataFrame({
                    "Split Scheme": ["70:30", "80:20", "90:10"],
                    "Accuracy": [0.9595, 0.9630, 0.9657],
                    "Precision (M)": [0.8744, 0.8686, 0.8881],
                    "Recall (M)": [0.9193, 0.9711, 0.9391],
                    "F1-Macro": [0.8951, 0.9112, 0.9114],
                    "ROC-AUC": [0.9837, 0.9855, 0.9783],
                }),
                "KFold_Detail": pd.DataFrame({
                    "Fold": [1, 2, 3, 4, 5],
                    "Accuracy": [0.9711, 0.9621, 0.9529, 0.9585, 0.9653],
                    "Precision (M)": [0.9063, 0.8981, 0.8434, 0.8513, 0.8962],
                    "Recall (M)": [0.9524, 0.9343, 0.9478, 0.9583, 0.9197],
                    "F1-Macro": [0.9219, 0.9151, 0.8861, 0.8946, 0.9129],
                    "ROC-AUC": [0.9906, 0.9877, 0.9870, 0.9824, 0.9865],
                }),
                "KFold_Summary": pd.DataFrame({
                    "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Macro", "ROC-AUC"],
                    "Mean": [0.9620, 0.8791, 0.9425, 0.9061, 0.9868],
                    "Std Dev": [0.0082, 0.0293, 0.0155, 0.0150, 0.0029],
                }),
            },
            ZSC_NAME: {
                "Split": pd.DataFrame({
                    "Split Scheme": ["70:30", "80:20", "90:10"],
                    "Accuracy": [0.8451, 0.8698, 0.8750],
                    "Precision (M)": [0.7431, 0.7808, 0.7852],
                    "Recall (M)": [0.8124, 0.7831, 0.8432],
                    "F1-Macro": [0.7675, 0.7820, 0.8083],
                    "ROC-AUC": [0.8968, 0.8857, 0.9032],
                }),
                "KFold_Detail": pd.DataFrame({
                    "Fold": [1, 2, 3, 4, 5],
                    "Accuracy": [0.8407, 0.8295, 0.8221, 0.8147, 0.8537],
                    "Precision (M)": [0.7274, 0.7332, 0.7176, 0.6989, 0.7612],
                    "Recall (M)": [0.7859, 0.8243, 0.7942, 0.7767, 0.8237],
                    "F1-Macro": [0.7491, 0.7592, 0.7412, 0.7219, 0.7844],
                    "ROC-AUC": [0.8742, 0.8999, 0.8829, 0.8553, 0.9037],
                }),
                "KFold_Summary": pd.DataFrame({
                    "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Macro", "ROC-AUC"],
                    "Mean": [0.8321, 0.7277, 0.8010, 0.7512, 0.8832],
                    "Std Dev": [0.0154, 0.0228, 0.0219, 0.0230, 0.0197],
                }),
            },
        },
    }

    return data


def render_page_header(title, subtitle):
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='dashboard-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def render_research_header(model, labeling):
    st.markdown(f"<h1>Evaluasi Performa Model {model}</h1>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="meta-row">
            <div class="meta-pill">
                <span class="meta-label">Metode labeling:</span>
                <span class="meta-value">{labeling}</span>
            </div>
            <div class="meta-pill">
                <span class="meta-label">Fokus analisis:</span>
                <span class="meta-value">Isu Tarif Trump</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meta_info(labeling, evaluation_type):
    st.markdown(
        f"""
        <div class="meta-row">
            <div class="meta-pill">
                <span class="meta-label">Metode labeling:</span>
                <span class="meta-value">{labeling}</span>
            </div>
            <div class="meta-pill">
                <span class="meta-label">Jenis evaluasi:</span>
                <span class="meta-value">{evaluation_type}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title, description=None):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader(title)

    if description:
        st.markdown(f"<div class='small-note'>{description}</div>", unsafe_allow_html=True)


def close_section_card():
    st.markdown("</div>", unsafe_allow_html=True)


def format_metric(value):
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def style_table(styler):
    return (
        styler
        .set_properties(**{
            "text-align": "center",
            "color": "#111827",
            "font-weight": "500",
        })
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("color", "#111827"),
                        ("font-weight", "700"),
                        ("background-color", "#f9fafb"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("color", "#111827"),
                    ],
                },
            ]
        )
    )


def format_percentage_table(df, subset):
    formatter = {
        col: (lambda value: "-" if pd.isna(value) else f"{value * 100:.2f}%")
        for col in subset
    }
    styled = df.style.format(formatter)
    return style_table(styled)


def format_regular_table(df):
    return style_table(df.style)


def display_image_if_exists(path, caption, width=None, use_container_width=False):
    if os.path.exists(path):
        st.image(path, caption=caption, width=width, use_container_width=use_container_width)
    else:
        st.info(f"File belum ditemukan: {path}")


def get_label_slug(labeling_method):
    return "lexicon" if labeling_method == LEX_NAME else "zsc"


def get_best_row(df, metric="F1-Macro"):
    if metric in df.columns:
        return df.loc[df[metric].idxmax()]

    return df.iloc[0]


def get_comparison_data(all_data, labeling_method, evaluation_type):
    if evaluation_type == "Skema Split Data":
        rows = []

        for model_name in ["IndoBERT", "BiLSTM"]:
            split_df = all_data[model_name][labeling_method]["Split"]
            best_row = get_best_row(split_df, metric="F1-Macro")

            row = {
                "Model": model_name,
                "Skema Terbaik": best_row["Split Scheme"],
                "Accuracy": best_row["Accuracy"],
                "Precision": best_row["Precision (M)"],
                "Recall": best_row["Recall (M)"],
                "F1-Macro": best_row["F1-Macro"],
                "ROC-AUC": best_row["ROC-AUC"] if "ROC-AUC" in split_df.columns else pd.NA,
            }

            rows.append(row)

        return pd.DataFrame(rows)

    rows = []

    for model_name in ["IndoBERT", "BiLSTM"]:
        summary_df = all_data[model_name][labeling_method]["KFold_Summary"]

        def get_mean(metric_keyword):
            return summary_df.loc[
                summary_df["Metric"].str.contains(metric_keyword, case=False),
                "Mean"
            ].iloc[0]

        rows.append({
            "Model": model_name,
            "Accuracy": get_mean("Accuracy"),
            "Precision": get_mean("Precision"),
            "Recall": get_mean("Recall"),
            "F1-Macro": get_mean("F1"),
            "ROC-AUC": get_mean("ROC-AUC"),
        })

    return pd.DataFrame(rows)


def get_all_split_comparison(all_data, labeling_method):
    split_rows = []

    for model_name in ["IndoBERT", "BiLSTM"]:
        split_df = all_data[model_name][labeling_method]["Split"].copy()
        best_idx = split_df["F1-Macro"].idxmax()

        split_df.insert(0, "Model", model_name)
        split_df["Status"] = ""
        split_df.loc[best_idx, "Status"] = "Skema Terbaik"

        if "ROC-AUC" not in split_df.columns:
            split_df["ROC-AUC"] = pd.NA

        split_rows.append(split_df)

    return pd.concat(split_rows, ignore_index=True)


def get_best_split_rows(all_data, labeling_method):
    rows = []

    for model_name in ["IndoBERT", "BiLSTM"]:
        split_df = all_data[model_name][labeling_method]["Split"]
        best_row = get_best_row(split_df, metric="F1-Macro")

        rows.append({
            "Model": model_name,
            "Skema Split Terbaik": best_row["Split Scheme"],
            "F1-Macro": best_row["F1-Macro"],
        })

    return pd.DataFrame(rows)


def get_best_model_text(comparison_df):
    best_row = comparison_df.loc[comparison_df["F1-Macro"].idxmax()]
    return (
        f"Berdasarkan nilai F1-Macro, model {best_row['Model']} menunjukkan performa "
        f"terbaik dengan nilai F1-Macro sebesar {format_metric(best_row['F1-Macro'])}. "
        "ROC-AUC tetap ditampilkan sebagai metrik tambahan, tetapi penentuan model terbaik "
        "pada dashboard ini didasarkan pada F1-Macro."
    )


def render_single_metric_chart(df, selected_metric):
    chart_data = df[["Model", selected_metric]].dropna().copy()
    chart_data["Nilai (%)"] = chart_data[selected_metric] * 100

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Model:N", title="Model", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Nilai (%):Q", title=f"{selected_metric} (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Nilai (%):Q", title=f"{selected_metric} (%)", format=".2f"),
            ],
        )
        .properties(height=360)
    )

    st.altair_chart(chart, use_container_width=True)


def render_grouped_bar_chart(df, metric_cols):
    chart_data = df.melt(
        id_vars=["Model"],
        value_vars=metric_cols,
        var_name="Metrik",
        value_name="Nilai",
    ).dropna()

    chart_data["Nilai (%)"] = chart_data["Nilai"] * 100

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metrik:N", title="Metrik", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Model:N"),
            y=alt.Y("Nilai (%):Q", title="Nilai (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Model:N", title="Model"),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Metrik:N", title="Metrik"),
                alt.Tooltip("Nilai (%):Q", title="Nilai (%)", format=".2f"),
            ],
        )
        .properties(height=420)
    )

    st.altair_chart(chart, use_container_width=True)


def clean_label_column(df, text_col, label_col):
    df_result = df[[text_col, label_col]].copy()
    df_result.columns = ["Teks", "Label"]

    jumlah_data_awal = len(df_result)

    df_result = df_result.dropna(subset=["Teks", "Label"])
    df_result["Label"] = df_result["Label"].astype(str).str.strip().str.lower()

    label_map = {
        "hate": "Hate Speech",
        "hate speech": "Hate Speech",
        "hs": "Hate Speech",
        "non_hate": "Non-Hate Speech",
        "non hate": "Non-Hate Speech",
        "non-hate": "Non-Hate Speech",
        "non-hate speech": "Non-Hate Speech",
        "non hate speech": "Non-Hate Speech",
        "nhs": "Non-Hate Speech",
    }

    df_result["Label Tampilan"] = df_result["Label"].map(label_map)
    df_result = df_result.dropna(subset=["Label Tampilan"])

    jumlah_data_valid = len(df_result)
    jumlah_data_tidak_valid = jumlah_data_awal - jumlah_data_valid

    return df_result, jumlah_data_tidak_valid


def get_top_words(text_series, top_n=10):
    stopwords = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada",
        "adalah", "atau", "karena", "dalam", "akan", "tidak", "ada", "jadi", "juga",
        "saya", "kamu", "dia", "mereka", "kita", "kami", "nya", "ya", "kok", "lah",
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
        "http", "https", "t", "co", "rt",
    }

    all_text = " ".join(text_series.dropna().astype(str)).lower()
    all_text = re.sub(r"http\S+|www\S+", " ", all_text)
    all_text = re.sub(r"[^a-zA-ZÀ-ÿ\u00C0-\u024F\u1E00-\u1EFF\u0100-\u017F\u0180-\u024F]+", " ", all_text)

    words = [
        word for word in all_text.split()
        if len(word) > 2 and word not in stopwords
    ]

    if len(words) == 0:
        return pd.DataFrame(columns=["Kata", "Frekuensi"])

    word_count = pd.Series(words).value_counts().head(top_n).reset_index()
    word_count.columns = ["Kata", "Frekuensi"]

    return word_count


with st.sidebar:
    st.markdown("<div class='sidebar-title'>Dashboard Analisis Hate Speech</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-caption'>Evaluasi model deteksi hate speech pada isu Tarif Trump.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Pilih Halaman")
    main_menu = st.radio(
        label="Halaman",
        options=["Perbandingan Model", "Evaluasi Hasil Riset", "Visualisasi Hasil Klasifikasi"],
        label_visibility="collapsed",
    )

    st.divider()

    if main_menu == "Perbandingan Model":
        comparison_labeling = st.radio(
            "Metode Labeling",
            [ZSC_NAME, LEX_NAME],
        )

        comparison_type = st.radio(
            "Jenis Evaluasi",
            ["Skema Split Data", "Cross Validation (5-Fold)"],
        )

    elif main_menu == "Evaluasi Hasil Riset":
        selected_model = st.radio(
            "Arsitektur Model",
            ["IndoBERT", "BiLSTM"],
        )

        selected_labeling = st.radio(
            "Metode Labeling",
            [ZSC_NAME, LEX_NAME],
        )

        analysis_type = st.radio(
            "Jenis Analisis",
            ["Skema Split Data", "Cross Validation (5-Fold)"],
        )


if main_menu == "Perbandingan Model":
    if "all_data" not in st.session_state:
        st.session_state["all_data"] = get_model_data()

    all_data = st.session_state["all_data"]

    render_page_header(
        title="Perbandingan Performa Model",
        subtitle="Halaman ini menampilkan perbandingan performa IndoBERT dan BiLSTM berdasarkan metode labeling dan jenis evaluasi yang dipilih.",
    )

    render_meta_info(comparison_labeling, comparison_type)

    comparison_df = get_comparison_data(
        all_data,
        labeling_method=comparison_labeling,
        evaluation_type=comparison_type,
    )

    table_percentage_cols = [
        col for col in ["Accuracy", "Precision", "Recall", "F1-Macro", "ROC-AUC"]
        if col in comparison_df.columns
    ]

    chart_percentage_cols = [
        col for col in table_percentage_cols
        if comparison_df[col].notna().all()
    ]

    render_section_card(
        "Ringkasan Perbandingan Model",
        "Tabel ini membandingkan performa IndoBERT dan BiLSTM. Model terbaik ditentukan berdasarkan nilai F1-Macro.",
    )

    st.dataframe(
        format_percentage_table(comparison_df, table_percentage_cols),
        use_container_width=True,
        hide_index=True,
    )

    close_section_card()

    if comparison_type == "Skema Split Data":
        render_section_card(
            "Skema Split Terbaik",
            "Bagian ini menunjukkan skema pembagian data terbaik pada masing-masing model berdasarkan nilai F1-Macro.",
        )

        best_split_df = get_best_split_rows(all_data, comparison_labeling)

        col_best1, col_best2 = st.columns(2)

        with col_best1:
            row = best_split_df.iloc[0]
            st.metric(
                label=row["Model"],
                value=row["Skema Split Terbaik"],
                delta=f"F1-Macro {format_metric(row['F1-Macro'])}",
            )

        with col_best2:
            row = best_split_df.iloc[1]
            st.metric(
                label=row["Model"],
                value=row["Skema Split Terbaik"],
                delta=f"F1-Macro {format_metric(row['F1-Macro'])}",
            )

        st.markdown("#### Detail Seluruh Skema Split")

        all_split_df = get_all_split_comparison(all_data, comparison_labeling)

        split_percentage_cols = [
            col for col in ["Accuracy", "Precision (M)", "Recall (M)", "F1-Macro", "ROC-AUC"]
            if col in all_split_df.columns
        ]

        st.dataframe(
            format_percentage_table(all_split_df, split_percentage_cols),
            use_container_width=True,
            hide_index=True,
        )

        close_section_card()

    render_section_card(
        "Grafik Perbandingan Metrik",
        "Grafik berikut menampilkan perbandingan IndoBERT dan BiLSTM secara berdampingan. Model terbaik tetap ditentukan berdasarkan F1-Macro.",
    )

    selected_metric = st.radio(
        "Pilih metrik utama",
        chart_percentage_cols,
        horizontal=True,
    )

    render_single_metric_chart(comparison_df, selected_metric)

    st.markdown("#### Perbandingan Semua Metrik")
    render_grouped_bar_chart(comparison_df, chart_percentage_cols)

    close_section_card()

    render_section_card("Interpretasi Singkat")
    st.info(get_best_model_text(comparison_df))
    close_section_card()


elif main_menu == "Evaluasi Hasil Riset":
    if "all_data" not in st.session_state:
        st.session_state["all_data"] = get_model_data()

    model_data = st.session_state["all_data"][selected_model][selected_labeling]
    label_slug = get_label_slug(selected_labeling)

    render_research_header(selected_model, selected_labeling)

    if analysis_type == "Skema Split Data":
        df_split = model_data["Split"]
        best_row = get_best_row(df_split)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Split Terbaik", best_row["Split Scheme"])
        metric_cols[1].metric("Accuracy", format_metric(best_row["Accuracy"]))
        metric_cols[2].metric("F1-Macro", format_metric(best_row["F1-Macro"]))

        if "ROC-AUC" in best_row.index:
            metric_cols[3].metric("ROC-AUC", format_metric(best_row["ROC-AUC"]))
        else:
            metric_cols[3].metric("Recall", format_metric(best_row["Recall (M)"]))

        st.divider()

        render_section_card(
            "Distribusi Label Dataset",
            "Visualisasi persebaran kelas digunakan untuk membaca komposisi data pada metode labeling yang dipilih.",
        )

        path_space = f"Assets/label distribution_{label_slug}.png"
        path_underscore = f"Assets/label_distribution_{label_slug}.png"

        if os.path.exists(path_space):
            st.image(path_space, caption=f"Distribusi Label: {selected_labeling}", use_container_width=True)
        elif os.path.exists(path_underscore):
            st.image(path_underscore, caption=f"Distribusi Label: {selected_labeling}", use_container_width=True)
        else:
            st.info("File distribusi label belum ditemukan di folder Assets.")

        close_section_card()

        render_section_card("Tabel Performa Skema Split")
        st.dataframe(
            format_percentage_table(df_split, df_split.columns[1:]),
            use_container_width=True,
            hide_index=True,
        )
        close_section_card()

        tab_visual, tab_wordcloud = st.tabs(["Visualisasi Evaluasi", "Analisis Konten Wordcloud"])

        with tab_visual:
            render_section_card(
                "Confusion Matrix dan ROC Curve",
                "Pilih rasio split untuk melihat visualisasi performa model pada skenario pengujian tertentu.",
            )

            split_choice = st.radio(
                "Pilih Rasio Split",
                ["70:30", "80:20", "90:10"],
                horizontal=True,
            )

            split_slug = split_choice.replace(":", "_")

            col1, col2 = st.columns(2)

            with col1:
                img_cm = f"Assets/{selected_model.lower()}_{label_slug}_cm_{split_slug}.png"
                display_image_if_exists(
                    img_cm,
                    caption=f"Confusion Matrix {selected_model} - Split {split_choice}",
                    use_container_width=True,
                )

            with col2:
                img_roc = f"Assets/{selected_model.lower()}_{label_slug}_roc_curve_{split_slug}.png"
                display_image_if_exists(
                    img_roc,
                    caption=f"ROC Curve {selected_model} - Split {split_choice}",
                    use_container_width=True,
                )

            close_section_card()

        with tab_wordcloud:
            render_section_card(
                "Wordcloud Hate Speech dan Non-Hate Speech",
                "Visualisasi ini membantu melihat kata dominan pada masing-masing kelas hasil labeling.",
            )

            col_wc1, col_wc2 = st.columns(2)

            with col_wc1:
                img_h = f"Assets/wordcloud_hate_{label_slug}.png"
                display_image_if_exists(
                    img_h,
                    caption="Wordcloud Hate Speech",
                    use_container_width=True,
                )

            with col_wc2:
                img_nh = f"Assets/wordcloud_non_hate_{label_slug}.png"
                display_image_if_exists(
                    img_nh,
                    caption="Wordcloud Non-Hate Speech",
                    use_container_width=True,
                )

            close_section_card()

    else:
        summary_df = model_data["KFold_Summary"]
        detail_df = model_data["KFold_Detail"].rename(columns={"Fold": "N-Fold"})

        f1_mean = summary_df.loc[summary_df["Metric"].str.contains("F1", case=False), "Mean"].iloc[0]
        acc_mean = summary_df.loc[summary_df["Metric"].eq("Accuracy"), "Mean"].iloc[0]
        recall_mean = summary_df.loc[summary_df["Metric"].str.contains("Recall", case=False), "Mean"].iloc[0]
        roc_mean = summary_df.loc[summary_df["Metric"].eq("ROC-AUC"), "Mean"].iloc[0]

        metric_cols = st.columns(4)
        metric_cols[0].metric("Mean Accuracy", format_metric(acc_mean))
        metric_cols[1].metric("Mean Recall", format_metric(recall_mean))
        metric_cols[2].metric("Mean F1-Macro", format_metric(f1_mean))
        metric_cols[3].metric("Mean ROC-AUC", format_metric(roc_mean))

        st.divider()

        tab_detail, tab_summary = st.tabs(["Detail Per N-Fold", "Statistik Ringkasan"])

        with tab_detail:
            render_section_card(
                "Detail Performa Setiap N-Fold",
                "Tabel ini menampilkan performa model pada masing-masing fold validasi silang.",
            )

            st.dataframe(
                format_percentage_table(detail_df, detail_df.columns[1:]),
                use_container_width=True,
                hide_index=True,
            )

            close_section_card()

        with tab_summary:
            render_section_card(
                "Ringkasan Statistik Cross Validation",
                "Nilai mean menunjukkan performa rata-rata, sedangkan standard deviation menunjukkan stabilitas performa model.",
            )

            st.dataframe(
                format_percentage_table(summary_df, ["Mean", "Std Dev"]),
                use_container_width=True,
                hide_index=True,
            )

            close_section_card()


elif main_menu == "Visualisasi Hasil Klasifikasi":
    render_page_header(
        title="Visualisasi Hasil Klasifikasi",
        subtitle="Unggah file hasil klasifikasi untuk menampilkan ringkasan, distribusi label, sampel hasil klasifikasi, dan karakteristik teks secara interaktif.",
    )

    render_section_card(
        "Unggah File Hasil Klasifikasi",
        "File yang diunggah merupakan CSV atau XLSX yang berisi kolom teks dan kolom hasil klasifikasi.",
    )

    uploaded_file = st.file_uploader("Unggah file CSV atau XLSX", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)

            st.markdown("#### Pratinjau File")
            st.dataframe(
                format_regular_table(df_up.head(10)),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Pilih Kolom Hasil Klasifikasi")

            default_text_index = 0
            default_label_index = 0

            if "full_text" in df_up.columns:
                default_text_index = list(df_up.columns).index("full_text")

            if "label" in df_up.columns:
                default_label_index = list(df_up.columns).index("label")

            col_select1, col_select2 = st.columns(2)

            with col_select1:
                text_col = st.selectbox(
                    "Pilih kolom teks",
                    df_up.columns,
                    index=default_text_index,
                )

            with col_select2:
                label_col = st.selectbox(
                    "Pilih kolom hasil klasifikasi",
                    df_up.columns,
                    index=default_label_index,
                )

            df_result, jumlah_data_tidak_valid = clean_label_column(df_up, text_col, label_col)

            if jumlah_data_tidak_valid > 0:
                st.info(
                    f"Terdapat {jumlah_data_tidak_valid} baris dengan teks atau label kosong/tidak valid, "
                    "sehingga tidak dimasukkan ke dalam perhitungan distribusi hasil klasifikasi."
                )

            st.markdown("#### Ringkasan Hasil Klasifikasi")

            total_data = len(df_result)
            total_hate = (df_result["Label Tampilan"] == "Hate Speech").sum()
            total_non_hate = (df_result["Label Tampilan"] == "Non-Hate Speech").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Jumlah Data Valid", total_data)
            col2.metric("Hate Speech", total_hate)
            col3.metric("Non-Hate Speech", total_non_hate)

            st.markdown("#### Distribusi Hasil Klasifikasi")

            label_count = (
                df_result["Label Tampilan"]
                .value_counts()
                .reindex(["Hate Speech", "Non-Hate Speech"], fill_value=0)
                .reset_index()
            )

            label_count.columns = ["Label", "Jumlah"]

            col_chart, col_table = st.columns([1.6, 1])

            with col_chart:
                st.bar_chart(label_count.set_index("Label"))

            with col_table:
                st.dataframe(
                    format_regular_table(label_count),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("#### Sampel Hasil Klasifikasi")

            st.dataframe(
                format_regular_table(df_result[["Teks", "Label Tampilan"]].head(20)),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Kata Dominan pada Masing-Masing Kelas")

            hate_texts = df_result[df_result["Label Tampilan"] == "Hate Speech"]["Teks"]
            non_hate_texts = df_result[df_result["Label Tampilan"] == "Non-Hate Speech"]["Teks"]

            hate_words = get_top_words(hate_texts, top_n=10)
            non_hate_words = get_top_words(non_hate_texts, top_n=10)

            col_hate, col_non_hate = st.columns(2)

            with col_hate:
                st.markdown("##### Hate Speech")
                if hate_words.empty:
                    st.info("Tidak ada kata dominan yang dapat ditampilkan.")
                else:
                    st.dataframe(
                        format_regular_table(hate_words),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.bar_chart(hate_words.set_index("Kata"))

            with col_non_hate:
                st.markdown("##### Non-Hate Speech")
                if non_hate_words.empty:
                    st.info("Tidak ada kata dominan yang dapat ditampilkan.")
                else:
                    st.dataframe(
                        format_regular_table(non_hate_words),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.bar_chart(non_hate_words.set_index("Kata"))

            csv_result = df_result[["Teks", "Label Tampilan"]].to_csv(index=False).encode("utf-8")

            st.download_button(
                "Unduh Hasil Klasifikasi (.csv)",
                data=csv_result,
                file_name="hasil_klasifikasi.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    close_section_card()


st.divider()
st.markdown(
    "<div class='footer'>Dashboard Analisis Skripsi | Denina Nastiti Putri Amani | 2026</div>",
    unsafe_allow_html=True,
)
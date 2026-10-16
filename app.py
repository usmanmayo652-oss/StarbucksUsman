import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Starbucks Nutrition Explorer",
    page_icon="☕",
    layout="wide",
)

st.title("☕ Starbucks Beverage Nutrition Explorer")
st.caption(
    "Explore the Starbucks beverage dataset, visualize nutrition information, "
    "and try a Decision Tree classifier."
)

DATA_FILE = "starbucks.csv"


def load_and_clean_data(file_obj=None):
    """Load the CSV and convert nutrition columns to usable numeric values."""
    if file_obj is not None:
        data = pd.read_csv(file_obj)
    elif os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)
    else:
        return None

    # Remove accidental spaces in column names from the original CSV.
    data.columns = data.columns.astype(str).str.strip()
    data = data.drop_duplicates().copy()

    # Convert columns that are numeric or percentages stored as strings.
    for col in data.columns:
        if data[col].dtype == "object":
            values = data[col].astype(str).str.strip()
            cleaned = values.str.replace("%", "", regex=False)
            numeric = pd.to_numeric(cleaned, errors="coerce")

            # Convert only when most non-empty values can be interpreted as numbers.
            non_empty = values.ne("") & values.ne("nan") & values.ne("None")
            if non_empty.sum() > 0 and numeric[non_empty].notna().mean() >= 0.80:
                data[col] = numeric

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload a Starbucks CSV (optional)", type=["csv"])
df = load_and_clean_data(uploaded_file)

if df is None:
    st.warning(
        "No dataset file was found. Put the provided CSV in the same folder as "
        "`app.py` and name it `starbucks.csv`, or upload it using the sidebar."
    )
    st.code("app.py\nstarbucks.csv\nrequirements.txt\nREADME.md")
    st.stop()

if "Beverage_category" not in df.columns:
    st.error("The dataset must contain a `Beverage_category` column.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
categories = sorted(df["Beverage_category"].dropna().astype(str).unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Beverage category",
    categories,
    default=categories,
)
search_text = st.sidebar.text_input("Search beverage name").strip().lower()

filtered = df.copy()
filtered = filtered[
    filtered["Beverage_category"].astype(str).isin(selected_categories)
]
if search_text and "Beverage" in filtered.columns:
    filtered = filtered[
        filtered["Beverage"].astype(str).str.lower().str.contains(search_text, na=False)
    ]

if filtered.empty:
    st.info("No rows match these filters. Change the category or search text.")
    st.stop()

# Overview metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beverages / records", f"{len(filtered):,}")
m2.metric("Categories", f"{filtered['Beverage_category'].nunique():,}")
if "Calories" in filtered.columns:
    m3.metric("Average calories", f"{filtered['Calories'].mean():.1f}")
else:
    m3.metric("Average calories", "N/A")
if "Caffeine (mg)" in filtered.columns:
    m4.metric("Average caffeine (mg)", f"{filtered['Caffeine (mg)'].mean():.1f}")
else:
    m4.metric("Average caffeine (mg)", "N/A")

overview_tab, charts_tab, model_tab, data_tab = st.tabs(
    ["Overview", "Visualizations", "Decision Tree", "Dataset"]
)

with overview_tab:
    st.subheader("Dataset overview")
    st.write(
        "Use the sidebar to filter beverage categories or search for a beverage. "
        "The charts and table update to reflect the selected records."
    )
    left, right = st.columns(2)

    with left:
        st.markdown("**Records by beverage category**")
        category_counts = filtered["Beverage_category"].astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        category_counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of records")
        ax.set_ylabel("Category")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Category share**")
        fig, ax = plt.subplots(figsize=(7, 5))
        category_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            ax=ax,
            ylabel="",
        )
        ax.set_title("Share of filtered records")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with charts_tab:
    st.subheader("Explore nutrition data")
    numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        selected_numeric = st.selectbox(
            "Choose a numeric field for its distribution",
            numeric_cols,
            index=numeric_cols.index("Calories") if "Calories" in numeric_cols else 0,
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(filtered[selected_numeric].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_numeric}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox(
                    "Scatter plot — X axis",
                    numeric_cols,
                    index=numeric_cols.index("Calories") if "Calories" in numeric_cols else 0,
                    key="scatter_x",
                )
            with c2:
                y_default = numeric_cols.index("Caffeine (mg)") if "Caffeine (mg)" in numeric_cols else min(1, len(numeric_cols) - 1)
                y_col = st.selectbox(
                    "Scatter plot — Y axis",
                    numeric_cols,
                    index=y_default,
                    key="scatter_y",
                )

            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_data = filtered[[x_col, y_col, "Beverage_category"]].dropna()
            sns.scatterplot(
                data=plot_data,
                x=x_col,
                y=y_col,
                hue="Beverage_category",
                ax=ax,
            )
            ax.set_title(f"{y_col} vs {x_col}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if len(numeric_cols) > 1:
            st.markdown("**Correlation heatmap**")
            corr = filtered[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(min(14, 1 + len(numeric_cols)), 8))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation between numeric nutrition fields")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No numeric columns are available to chart.")

with model_tab:
    st.subheader("Decision Tree classification")
    st.write(
        "Train a Decision Tree to predict a selected category using numeric nutrition "
        "fields. This is a demonstration model; results depend on the selected target "
        "and the small size of this dataset."
    )

    target_candidates = [
        col for col in filtered.columns
        if filtered[col].nunique(dropna=True) >= 2
        and filtered[col].nunique(dropna=True) <= 30
        and filtered[col].dtype == "object"
    ]
    if "Beverage_category" not in target_candidates:
        target_candidates.insert(0, "Beverage_category")
    target_candidates = list(dict.fromkeys(target_candidates))

    target = st.selectbox(
        "Target to predict",
        target_candidates,
        index=target_candidates.index("Beverage_category"),
    )
    model_numeric_cols = [
        col for col in filtered.select_dtypes(include=np.number).columns
        if col != target
    ]

    if len(model_numeric_cols) == 0:
        st.warning("At least one numeric feature is needed to train the model.")
    else:
        default_features = [
            col for col in [
                "Calories", "Total Fat (g)", "Trans Fat (g)",
                "Saturated Fat (g)", "Sodium (mg)",
                "Total Carbohydrates (g)", "Cholesterol (mg)",
                "Dietary Fibre (g)", "Sugars (g)", "Protein (g)",
                "Vitamin A (% DV)", "Vitamin C (% DV)",
                "Calcium (% DV)", "Iron (% DV)", "Caffeine (mg)"
            ] if col in model_numeric_cols
        ]
        chosen_features = st.multiselect(
            "Numeric features",
            model_numeric_cols,
            default=default_features or model_numeric_cols,
        )
        max_depth = st.slider("Maximum tree depth", min_value=1, max_value=12, value=5)
        test_fraction = st.slider(
            "Test set size (%)", min_value=10, max_value=40, value=20, step=5
        )

        if st.button("Train Decision Tree", type="primary"):
            if not chosen_features:
                st.error("Select at least one feature.")
            else:
                model_df = filtered[chosen_features + [target]].dropna(subset=[target]).copy()
                model_df[target] = model_df[target].astype(str)
                counts = model_df[target].value_counts()
                keep_classes = counts[counts >= 2].index
                model_df = model_df[model_df[target].isin(keep_classes)]

                if model_df[target].nunique() < 2:
                    st.error(
                        "The selected target needs at least two classes with two or more records each."
                    )
                else:
                    X = model_df[chosen_features].apply(pd.to_numeric, errors="coerce")
                    y_raw = model_df[target].astype(str)
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(y_raw)

                    n_classes = len(encoder.classes_)
                    test_count = int(np.ceil(len(model_df) * test_fraction / 100))
                    train_count = len(model_df) - test_count
                    class_counts = pd.Series(y).value_counts()
                    can_stratify = (
                        class_counts.min() >= 2
                        and test_count >= n_classes
                        and train_count >= n_classes
                    )

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_fraction / 100,
                        random_state=42,
                        stratify=y if can_stratify else None,
                    )

                    model = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("tree", DecisionTreeClassifier(
                            criterion="gini",
                            max_depth=max_depth,
                            random_state=42,
                            class_weight="balanced",
                        )),
                    ])
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)

                    st.metric("Test accuracy", f"{accuracy * 100:.2f}%")
                    st.caption(
                        f"Training records: {len(X_train)} · Test records: {len(X_test)} · "
                        f"Target classes: {n_classes}"
                    )

                    labels = np.arange(n_classes)
                    cm = confusion_matrix(y_test, predictions, labels=labels)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=encoder.classes_,
                        yticklabels=encoder.classes_,
                        ax=ax,
                    )
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel("Actual label")
                    ax.set_title("Confusion matrix")
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    report = classification_report(
                        y_test,
                        predictions,
                        labels=labels,
                        target_names=encoder.classes_,
                        output_dict=True,
                        zero_division=0,
                    )
                    st.markdown("**Classification report**")
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                    tree_model = model.named_steps["tree"]
                    importances = pd.DataFrame({
                        "Feature": chosen_features,
                        "Importance": tree_model.feature_importances_,
                    }).sort_values("Importance", ascending=False)
                    st.markdown("**Feature importance**")
                    st.dataframe(importances, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(18, 9))
                    plot_tree(
                        tree_model,
                        feature_names=chosen_features,
                        class_names=[str(c) for c in encoder.classes_],
                        filled=True,
                        rounded=True,
                        fontsize=7,
                        ax=ax,
                    )
                    ax.set_title("Decision Tree")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

with data_tab:
    st.subheader("Filtered data")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Missing values")
    missing = filtered.isna().sum().rename("Missing values").to_frame()
    missing["Missing %"] = (filtered.isna().mean() * 100).round(2)
    st.dataframe(missing, use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_bytes,
        file_name="starbucks_filtered.csv",
        mime="text/csv",
    )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, pandas, scikit-learn, Matplotlib and Seaborn.")

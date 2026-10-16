# Starbucks Beverage Nutrition Explorer

A Streamlit web app built from the Starbucks beverage nutrition dataset. It lets users filter and explore beverages, view nutrition charts, inspect the dataset, download filtered rows, and train a Decision Tree classification model.

## Features

- Dataset overview and category counts
- Category filters and beverage-name search
- Pie chart, histogram, scatter plot, and correlation heatmap
- Missing-value overview and filtered data preview
- Interactive Decision Tree classifier
- Adjustable target, numeric features, tree depth, and test-set size
- Accuracy, confusion matrix, classification report, feature importance, and tree visualization
- CSV download for filtered data

## Project files

```text
starbucks-streamlit-app/
├── app.py
├── requirements.txt
├── README.md
└── starbucks.csv
```

Keep the supplied CSV in the same folder as `app.py` and name it `starbucks.csv`. Alternatively, the app lets you upload a CSV from the sidebar.

## Run locally

### 1. Install Python

Use Python 3.10 or newer.

### 2. Open a terminal in the project folder

Create and activate a virtual environment if desired.

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deploy on GitHub and Streamlit Community Cloud

1. Create a new repository on GitHub, for example `starbucks-streamlit-app`.
2. Add `app.py`, `requirements.txt`, `README.md`, and `starbucks.csv` to the repository.
3. Commit and push the files.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
5. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
6. Deploy the app. After the build completes, Streamlit provides a public app URL.

### Git commands

Run these commands from the project folder after creating an empty GitHub repository. Replace the URL with your own repository URL.

```bash
git init
git add app.py README.md requirements.txt starbucks.csv
git commit -m "Add Starbucks Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/starbucks-streamlit-app.git
git push -u origin main
```

## Model notes

The Decision Tree is a demonstration classification model. The default target is `Beverage_category`; it uses numeric nutrition columns as input features. The dataset is relatively small, so the reported test accuracy may vary with the target and selected features. A good accuracy score alone does not guarantee that the model generalizes to new data.

## Dataset

The app is designed for the supplied Starbucks beverage nutrition CSV. Check the dataset's original source and license before publishing or redistributing it publicly.

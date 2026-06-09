# IMDb Movie Rating Predictor

A web app that predicts a movie's IMDb rating using a Machine Learning model trained in Part 2 of the project.

You fill in basic movie details → the app runs them through the model → you get a predicted rating between **0 and 10**.

---

## Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd partthree_pred
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate it
```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
python api.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## Input Fields

| Field | Valid Values |
|---|---|
| Release Year | 1895 – 2024 |
| Runtime (minutes) | 60 – 300 |
| Number of Actors | 1 – 5 |
| Number of Genres | 1 – 5 |
| Genres | Pick from buttons (e.g. Drama, Comedy) |
| Language | Full name (e.g. English, Hebrew) |
| Country | Full name (e.g. United States, Israel) |

---

## Project Files

| File | What it does |
|---|---|
| `api.py` | Flask backend — 2 endpoints (`/` and `/predict`) |
| `index.html` | Web form the user interacts with |
| `assets_data_prep.py` | `prepare_data()` function — copied from Part 2 |
| `trained_model.pkl` | Trained Random Forest model from Part 2 (best-performing model) |
| `requirements.txt` | All required Python packages |

---

## Team Members

| Name | ID |
|---|---|
|  |  |
|  |  |

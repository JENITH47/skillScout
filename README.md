# SkillScout

SkillScout is a Streamlit app for ranking resumes against a job description using:

- structured resume/JD feature extraction
- semantic similarity scoring
- a stacked ML inference pipeline
- recruiter-style feedback, requirement fit, and confidence signals

## What Is Included

This repo is set up as a lightweight app repository.

Included:

- `app.py`
- `ranking_feedback_nlp.py`
- `inference_pipeline.py`
- `requirement.txt`
- the production stack model artifacts used by the app

Excluded via `.gitignore`:

- local virtual environment files
- huge training datasets
- raw embedding dumps
- sample resumes
- research / backup model folders
- generated CSV reports

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirement.txt
```

3. Start the app:

```powershell
streamlit run app.py
```

4. Open:

`http://localhost:8501`

## GitHub Upload

If Git is installed:

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

If Git is not installed on your machine, install **Git for Windows** or use **GitHub Desktop** and publish this folder as a new repository.

## Notes

- The app currently expects the stack model files in the repo root.
- If you want a fully reproducible training repo too, create a second separate repository for datasets, notebooks, and experiments instead of mixing them into the app repo.

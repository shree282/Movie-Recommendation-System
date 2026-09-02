# Movie Recommendation System

## How to run
1. Install packages:
   pip install -r requirements.txt
2. Run:
   streamlit run app.py

## Method
The project uses content-based filtering. Movie genres are converted into TF-IDF vectors and cosine similarity is used to find movies with similar genre profiles.

## Files
- movies.csv: dataset
- app.py: Streamlit application
- Movie_Recommendation_Notebook.ipynb: notebook version
- Movie_Recommendation_Report.pdf: 2-page report

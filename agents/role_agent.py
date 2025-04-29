#Define role- and level-specific keywords used to filter relevant documents
role_level_keywords = {
    "Data Scientist": {
        "Junior": ["basics", "intro", "python", "pandas", "EDA", "linear regression"],
        "Mid": ["feature engineering", "cross-validation", "metrics", "XGBoost"],
        "Senior": ["model deployment", "MLOps", "architecture", "scaling", "optimization"]
    },
    "Data Engineer": {
        "Junior": ["SQL", "ETL basics", "data pipeline", "batch job"],
        "Mid": ["Airflow", "Spark", "data lakes", "streaming"],
        "Senior": ["architecture", "data mesh", "BigQuery optimization", "DevOps"]
    },
    "Data Analyst": {
        "Junior": ["excel", "charts", "basic SQL", "Power BI"],
        "Mid": ["advanced SQL", "dashboard", "KPIs", "insights"],
        "Senior": ["strategy", "business modeling", "forecasting", "stakeholder"]
    }
}

# Function to filter documents based on the user's role and experience level
def filter_docs_by_role_and_level(docs, role, level):
    # Get the list of keywords for the given role and level; return empty list if not found
    keywords = role_level_keywords.get(role, {}).get(level, [])

    # Return documents that contain at least one of the keywords (case-insensitive match)
    return [
        doc for doc in docs 
        if any(k.lower() in doc.page_content.lower() for k in keywords)
    ]

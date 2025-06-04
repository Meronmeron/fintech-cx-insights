# Ethiopian Mobile Banking Apps Customer Satisfaction Analysis

## Project Overview

This project analyzes customer satisfaction with mobile banking apps by collecting and processing user reviews from the Google Play Store for three major Ethiopian banks:

1. **Commercial Bank of Ethiopia (CBE)**
2. **Bank of Abyssinia (BOA)**
3. **Dashen Bank**

The analysis simulates the role of a Data Analyst at Omega Consultancy, a firm advising banks on customer satisfaction insights.

## Project Structure

```
├── data/
│   ├── raw/                    # Raw scraped data
│   ├── processed/              # Cleaned and processed data
│   └── final/                  # Final analysis-ready datasets
├── src/
│   ├── scraping/              # Web scraping modules
│   ├── preprocessing/         # Data cleaning and preprocessing
│   ├── analysis/              # Sentiment analysis and insights
│   └── visualization/         # Charts and visualizations
├── notebooks/                 # Jupyter notebooks for exploration
├── results/                   # Final reports and insights
└── requirements.txt          # Project dependencies
```

## Task 1: Data Collection & Preprocessing

### Objectives

- Scrape reviews from Google Play Store for three banks' mobile apps
- Collect minimum 400 reviews per bank (1,200 total)
- Extract: Review Text, Rating (1-5), Date, Bank/App Name, Source
- Preprocess data: remove duplicates, handle missing data, normalize dates
- Save as structured CSV format

### Data Schema

| Column | Description         | Example                             |
| ------ | ------------------- | ----------------------------------- |
| review | User feedback text  | "Love the UI, but it crashes often" |
| rating | Star rating (1-5)   | 4                                   |
| date   | Review posting date | 2024-01-15                          |
| bank   | Bank/App name       | "Commercial Bank of Ethiopia"       |
| source | Data source         | "Google Play"                       |

## Setup Instructions

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Collection**

   ```bash
   python src/scraping/play_store_scraper.py
   ```

3. **Process Data**
   ```bash
   python src/preprocessing/data_cleaner.py
   ```

## Banks & Apps to Analyze

### Commercial Bank of Ethiopia (CBE)

- App: CBE Mobile Banking
- Package: com.cbe.mobile

### Bank of Abyssinia (BOA)

- App: BOA Mobile Banking
- Package: com.boa.mobile

### Dashen Bank

- App: Dashen Mobile Banking
- Package: com.dashen.mobile

## Expected Deliverables

- Clean dataset with 1,200+ reviews
- Sentiment analysis results
- Customer satisfaction insights
- Visual dashboards and reports
- Recommendations for banking improvements

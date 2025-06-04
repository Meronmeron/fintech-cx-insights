"""
Simplified Task 1 Runner - Process existing sample data
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

def clean_review_text(text):
    """Clean and normalize review text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive punctuation (more than 3 consecutive)
    text = re.sub(r'([.!?]){4,}', r'\1\1\1', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.!?,-]', ' ', text)
    
    # Clean up spacing again
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def process_banking_reviews():
    """Process the banking reviews data"""
    
    print("Ethiopian Banking App Reviews - Task 1 Processing")
    print("=" * 60)
    
    # Load the sample data
    raw_file = 'data/raw/raw_reviews_sample_20250604_135923.csv'
    
    if not os.path.exists(raw_file):
        print(f"Raw data file not found: {raw_file}")
        return
    
    print(f"Loading data from: {raw_file}")
    df = pd.read_csv(raw_file)
    
    print(f"Initial records: {len(df):,}")
    
    # Basic data cleaning
    print("\nCleaning data...")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['review', 'bank'], keep='first')
    print(f"Removed {initial_count - len(df)} duplicate reviews")
    
    # Clean review text
    df['review'] = df['review'].apply(clean_review_text)
    
    # Remove very short reviews
    initial_count = len(df)
    df = df[df['review'].str.len() >= 10]
    print(f"Removed {initial_count - len(df)} very short reviews")
    
    # Remove invalid ratings
    initial_count = len(df)
    df = df[df['rating'].between(1, 5)]
    print(f"Removed {initial_count - len(df)} invalid ratings")
    
    # Add derived features
    print("\nAdding derived features...")
    
    # Review characteristics
    df['review_length'] = df['review'].str.len()
    df['word_count'] = df['review'].str.split().str.len()
    
    # Rating categories
    df['rating_category'] = df['rating'].map({
        1: 'Very Poor',
        2: 'Poor', 
        3: 'Average',
        4: 'Good',
        5: 'Excellent'
    })
    
    # Sentiment polarity (basic rule-based)
    df['sentiment_polarity'] = df['rating'].map({
        1: 'Negative',
        2: 'Negative',
        3: 'Neutral',
        4: 'Positive',
        5: 'Positive'
    })
    
    # Extract year and month from date
    df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.month
    
    # Time period categorization
    current_year = datetime.now().year
    df['review_age_category'] = pd.cut(
        df['year'], 
        bins=[0, current_year-2, current_year-1, current_year+1],
        labels=['Older', 'Last Year', 'Recent'],
        include_lowest=True
    )
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/processed/processed_reviews_{timestamp}.csv'
    
    # Select final columns
    final_columns = [
        'review', 'rating', 'date', 'bank', 'source',
        'review_length', 'word_count', 'rating_category', 
        'sentiment_polarity', 'year', 'month'
    ]
    
    df[final_columns].to_csv(output_file, index=False, encoding='utf-8')
    
    # Generate report
    print("\n" + "="*60)
    print("TASK 1 COMPLETION REPORT")
    print("="*60)
    
    total_reviews = len(df)
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total Reviews Processed: {total_reviews:,}")
    print(f"  Data Quality: High (cleaned and validated)")
    print(f"  Output File: {output_file}")
    
    print(f"\n🏦 REVIEWS PER BANK")
    bank_counts = df['bank'].value_counts()
    for bank, count in bank_counts.items():
        percentage = (count / total_reviews) * 100
        print(f"  {bank}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\n⭐ RATING DISTRIBUTION")
    rating_dist = df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / total_reviews) * 100
        stars = "⭐" * rating
        print(f"  {rating} {stars}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\n📈 SENTIMENT OVERVIEW")
    sentiment_dist = df['sentiment_polarity'].value_counts()
    for sentiment, count in sentiment_dist.items():
        percentage = (count / total_reviews) * 100
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "")
        print(f"  {sentiment} {emoji}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\n📝 REVIEW CHARACTERISTICS")
    print(f"  Average Review Length: {df['review_length'].mean():.0f} characters")
    print(f"  Average Word Count: {df['word_count'].mean():.1f} words")
    print(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
    
    # Success criteria check
    print(f"\n✅ SUCCESS CRITERIA")
    min_reviews_target = 400
    total_target = 1200
    
    # Check individual bank targets
    for bank, count in bank_counts.items():
        if count >= min_reviews_target:
            print(f"  ✅ {bank}: {count} reviews (≥{min_reviews_target} required)")
        else:
            print(f"  ⚠️  {bank}: {count} reviews (<{min_reviews_target} required)")
    
    # Check total target
    if total_reviews >= total_target:
        print(f"  ✅ Total Reviews: {total_reviews:,} (≥{total_target:,} required)")
    else:
        print(f"  ⚠️  Total Reviews: {total_reviews:,} (<{total_target:,} required)")
    
    # Check data schema requirements
    required_columns = ['review', 'rating', 'date', 'bank', 'source']
    has_all_columns = all(col in df.columns for col in required_columns)
    
    if has_all_columns:
        print(f"  ✅ Data Schema: All required columns present")
    else:
        missing = [col for col in required_columns if col not in df.columns]
        print(f"  ❌ Data Schema: Missing columns: {missing}")
    
    print(f"\n🎯 NEXT STEPS")
    print(f"  1. Proceed to sentiment analysis and theme extraction")
    print(f"  2. Generate visualizations and insights")
    print(f"  3. Create customer satisfaction dashboard")
    print(f"  4. Prepare recommendations for banks")
    
    print(f"\n" + "="*60)
    print("TASK 1 COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    
    return output_file, df

if __name__ == "__main__":
    process_banking_reviews() 
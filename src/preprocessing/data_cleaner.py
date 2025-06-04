"""
Data Preprocessing Module for Ethiopian Banking App Reviews
Cleans, normalizes, and prepares scraped data for analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dateutil import parser

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing for review data
    """
    
    def __init__(self):
        """Initialize the data cleaner"""
        self.cleaned_data = None
        self.cleaning_stats = {}
        
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw scraped data from CSV file
        """
        try:
            logger.info(f"Loading raw data from: {filepath}")
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Loaded {len(df)} raw reviews")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_review_text(self, text: str) -> str:
        """
        Clean and normalize review text
        """
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
    
    def normalize_date(self, date_value: Any) -> Optional[str]:
        """
        Normalize date to YYYY-MM-DD format
        """
        if pd.isna(date_value):
            return None
        
        try:
            # If it's already a datetime object
            if isinstance(date_value, datetime):
                return date_value.strftime('%Y-%m-%d')
            
            # If it's a string, try to parse it
            if isinstance(date_value, str):
                # Handle various date formats
                parsed_date = parser.parse(date_value)
                return parsed_date.strftime('%Y-%m-%d')
            
            # If it's a timestamp
            if isinstance(date_value, (int, float)):
                parsed_date = datetime.fromtimestamp(date_value)
                return parsed_date.strftime('%Y-%m-%d')
                
        except Exception as e:
            logger.warning(f"Could not parse date: {date_value} - {e}")
            return None
        
        return None
    
    def validate_rating(self, rating: Any) -> int:
        """
        Validate and normalize rating values (1-5 scale)
        """
        if pd.isna(rating):
            return 0
        
        try:
            rating = float(rating)
            
            # Ensure rating is within valid range
            if rating < 1:
                return 1
            elif rating > 5:
                return 5
            else:
                return int(round(rating))
                
        except (ValueError, TypeError):
            logger.warning(f"Invalid rating value: {rating}")
            return 0
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate reviews based on multiple criteria
        """
        logger.info("Removing duplicate reviews...")
        
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicates based on review text and bank (case-insensitive)
        df['review_lower'] = df['review'].str.lower()
        df = df.drop_duplicates(subset=['review_lower', 'bank'], keep='first')
        df = df.drop('review_lower', axis=1)
        
        # Remove very similar reviews (same first 50 characters)
        df['review_prefix'] = df['review'].str[:50].str.lower()
        df = df.drop_duplicates(subset=['review_prefix', 'bank'], keep='first')
        df = df.drop('review_prefix', axis=1)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        logger.info(f"Removed {removed_count} duplicate reviews ({initial_count} -> {final_count})")
        self.cleaning_stats['duplicates_removed'] = removed_count
        
        return df
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the dataset
        """
        logger.info("Handling missing data...")
        
        # Track missing data statistics
        missing_stats = df.isnull().sum()
        logger.info(f"Missing data before cleaning:\n{missing_stats}")
        
        # Remove rows with missing review text
        initial_count = len(df)
        df = df.dropna(subset=['review'])
        df = df[df['review'].str.strip() != '']
        reviews_removed = initial_count - len(df)
        
        if reviews_removed > 0:
            logger.info(f"Removed {reviews_removed} rows with missing/empty review text")
        
        # Fill missing ratings with 0 (will be handled later)
        df['rating'] = df['rating'].fillna(value=0)
        
        # Fill missing dates with None (will be handled later)
        df['date'] = df['date'].fillna(value=None)
        
        # Fill missing bank names (shouldn't happen, but just in case)
        df['bank'] = df['bank'].fillna(value='Unknown')
        
        # Fill missing source
        df['source'] = df['source'].fillna(value='Google Play')
        
        self.cleaning_stats['rows_with_missing_reviews'] = reviews_removed
        
        return df
    
    def filter_quality_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out low-quality reviews
        """
        logger.info("Filtering low-quality reviews...")
        
        initial_count = len(df)
        
        # Remove very short reviews (less than 10 characters)
        df = df[df['review'].str.len() >= 10]
        
        # Remove reviews that are just ratings without text content
        df = df[~df['review'].str.match(r'^[1-5]\s*stars?\.?$', case=False, na=False)]
        
        # Remove reviews with only special characters or numbers
        df = df[df['review'].str.contains(r'[a-zA-Z]', regex=True, na=False)]
        
        # Remove invalid ratings
        df = df[df['rating'].between(1, 5)]
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        logger.info(f"Removed {removed_count} low-quality reviews ({initial_count} -> {final_count})")
        self.cleaning_stats['low_quality_removed'] = removed_count
        
        return df
    
    def standardize_bank_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize bank names for consistency
        """
        logger.info("Standardizing bank names...")
        
        # Define bank name mappings for consistency
        bank_mappings = {
            'commercial bank of ethiopia': 'Commercial Bank of Ethiopia',
            'cbe': 'Commercial Bank of Ethiopia',
            'bank of abyssinia': 'Bank of Abyssinia',
            'boa': 'Bank of Abyssinia',
            'dashen bank': 'Dashen Bank',
            'dashen': 'Dashen Bank'
        }
        
        # Store original bank names
        original_banks = df['bank'].copy()
        
        # Apply mappings
        df_bank_lower = df['bank'].str.lower()
        df['bank'] = df_bank_lower.map(bank_mappings)
        
        # Fill unmapped values with original bank names
        mask = df['bank'].isna()
        df.loc[mask, 'bank'] = original_banks.loc[mask]
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for analysis
        """
        logger.info("Adding derived features...")
        
        # Review length
        df['review_length'] = df['review'].str.len()
        
        # Review word count
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
        
        # Time period
        current_year = datetime.now().year
        df['review_age_category'] = pd.cut(
            df['year'], 
            bins=[0, current_year-2, current_year-1, current_year+1],
            labels=['Older', 'Last Year', 'Recent'],
            include_lowest=True
        )
        
        return df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline
        """
        logger.info("Starting comprehensive data cleaning...")
        
        # Initialize cleaning stats
        self.cleaning_stats = {
            'initial_records': len(df),
            'duplicates_removed': 0,
            'low_quality_removed': 0,
            'rows_with_missing_reviews': 0
        }
        
        # Step 1: Handle missing data
        df = self.handle_missing_data(df)
        
        # Step 2: Clean review text
        logger.info("Cleaning review text...")
        df['review'] = df['review'].apply(self.clean_review_text)
        
        # Step 3: Normalize dates
        logger.info("Normalizing dates...")
        df['date'] = df['date'].apply(self.normalize_date)
        
        # Step 4: Validate ratings
        logger.info("Validating ratings...")
        df['rating'] = df['rating'].apply(self.validate_rating)
        
        # Step 5: Standardize bank names
        df = self.standardize_bank_names(df)
        
        # Step 6: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 7: Filter low-quality reviews
        df = self.filter_quality_reviews(df)
        
        # Step 8: Add derived features
        df = self.add_derived_features(df)
        
        # Final statistics
        self.cleaning_stats['final_records'] = len(df)
        self.cleaning_stats['total_removed'] = (
            self.cleaning_stats['initial_records'] - self.cleaning_stats['final_records']
        )
        
        logger.info("Data cleaning completed!")
        self.print_cleaning_summary()
        
        self.cleaned_data = df
        return df
    
    def print_cleaning_summary(self):
        """
        Print a summary of the cleaning process
        """
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Initial records: {self.cleaning_stats['initial_records']:,}")
        print(f"Duplicates removed: {self.cleaning_stats['duplicates_removed']:,}")
        print(f"Low-quality reviews removed: {self.cleaning_stats['low_quality_removed']:,}")
        print(f"Rows with missing reviews: {self.cleaning_stats['rows_with_missing_reviews']:,}")
        print(f"Final records: {self.cleaning_stats['final_records']:,}")
        print(f"Total removed: {self.cleaning_stats['total_removed']:,}")
        print(f"Retention rate: {(self.cleaning_stats['final_records']/self.cleaning_stats['initial_records']*100):.1f}%")
        print("="*60)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save processed data to CSV
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_reviews_{timestamp}.csv"
        
        # Ensure processed directory exists
        os.makedirs('data/processed', exist_ok=True)
        
        filepath = os.path.join('data', 'processed', filename)
        
        # Select final columns for the dataset
        final_columns = [
            'review', 'rating', 'date', 'bank', 'source',
            'review_length', 'word_count', 'rating_category', 
            'sentiment_polarity', 'year', 'month'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in final_columns if col in df.columns]
        
        df[available_columns].to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Processed data saved to: {filepath}")
        
        return filepath

def main():
    """
    Main function to run the data cleaning process
    """
    import glob
    
    print("Ethiopian Banking App Reviews Data Cleaner")
    print("=" * 50)
    
    # Find the most recent raw data file
    raw_files = glob.glob('data/raw/raw_reviews_*.csv')
    
    if not raw_files:
        print("No raw data files found. Please run the scraper first.")
        return
    
    # Use the most recent file
    latest_file = max(raw_files, key=os.path.getctime)
    print(f"Processing file: {latest_file}")
    
    try:
        # Initialize cleaner
        cleaner = DataCleaner()
        
        # Load and clean data
        raw_df = cleaner.load_raw_data(latest_file)
        clean_df = cleaner.clean_dataset(raw_df)
        
        # Save processed data
        output_file = cleaner.save_processed_data(clean_df)
        
        # Print final statistics
        print(f"\nProcessed data saved to: {output_file}")
        
        print("\nFinal Dataset Statistics:")
        print(f"Total reviews: {len(clean_df):,}")
        print("\nReviews per bank:")
        bank_counts = clean_df['bank'].value_counts()
        for bank, count in bank_counts.items():
            print(f"  {bank}: {count:,} reviews")
        
        print(f"\nRating distribution:")
        rating_dist = clean_df['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"  {rating} stars: {count:,} reviews")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
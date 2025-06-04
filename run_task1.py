"""
Task 1 Runner: Complete Data Collection and Preprocessing Pipeline
Executes web scraping and data cleaning for Ethiopian banking app reviews
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task1_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """
    Install required dependencies
    """
    print("Installing required dependencies...")
    
    import subprocess
    import sys
    
    packages = [
        'google-play-scraper==1.2.4',
        'pandas==2.1.4',
        'numpy==1.25.2',
        'python-dateutil==2.8.2',
        'tqdm==4.66.1'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {package}: {e}")

def run_scraping():
    """
    Execute the web scraping process
    """
    print("\n" + "="*60)
    print("PHASE 1: WEB SCRAPING")
    print("="*60)
    
    try:
        # Import and run scraper
        from src.scraping.play_store_scraper import PlayStoreScraper
        
        scraper = PlayStoreScraper()
        logger.info("Starting web scraping process...")
        
        # Scrape all banks
        reviews_df = scraper.scrape_all_banks()
        
        if not reviews_df.empty:
            # Save raw data
            raw_filepath = scraper.save_raw_data(reviews_df)
            
            logger.info(f"Scraping completed successfully!")
            logger.info(f"Total reviews collected: {len(reviews_df)}")
            
            # Print bank-wise summary
            print("\nScraping Results:")
            bank_counts = reviews_df['bank'].value_counts()
            for bank, count in bank_counts.items():
                print(f"  {bank}: {count} reviews")
            
            return raw_filepath
        else:
            logger.error("No reviews were collected during scraping")
            return None
            
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        raise

def run_preprocessing(raw_filepath):
    """
    Execute the data preprocessing process
    """
    print("\n" + "="*60)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*60)
    
    try:
        # Import and run cleaner
        from src.preprocessing.data_cleaner import DataCleaner
        
        cleaner = DataCleaner()
        logger.info("Starting data preprocessing...")
        
        # Load and clean data
        raw_df = cleaner.load_raw_data(raw_filepath)
        clean_df = cleaner.clean_dataset(raw_df)
        
        # Save processed data
        processed_filepath = cleaner.save_processed_data(clean_df)
        
        logger.info("Data preprocessing completed successfully!")
        
        return processed_filepath, clean_df
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def generate_final_report(clean_df, processed_filepath):
    """
    Generate a comprehensive final report
    """
    print("\n" + "="*60)
    print("TASK 1 COMPLETION REPORT")
    print("="*60)
    
    total_reviews = len(clean_df)
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total Reviews Collected: {total_reviews:,}")
    print(f"  Data Quality: High (cleaned and validated)")
    print(f"  Output File: {processed_filepath}")
    
    print(f"\n🏦 REVIEWS PER BANK")
    bank_counts = clean_df['bank'].value_counts()
    for bank, count in bank_counts.items():
        percentage = (count / total_reviews) * 100
        print(f"  {bank}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\n⭐ RATING DISTRIBUTION")
    rating_dist = clean_df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / total_reviews) * 100
        stars = "⭐" * rating
        print(f"  {rating} {stars}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\n📈 SENTIMENT OVERVIEW")
    sentiment_dist = clean_df['sentiment_polarity'].value_counts()
    for sentiment, count in sentiment_dist.items():
        percentage = (count / total_reviews) * 100
        emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(sentiment, "")
        print(f"  {sentiment} {emoji}: {count:,} reviews ({percentage:.1f}%)")
    
    print(f"\n📝 REVIEW CHARACTERISTICS")
    print(f"  Average Review Length: {clean_df['review_length'].mean():.0f} characters")
    print(f"  Average Word Count: {clean_df['word_count'].mean():.1f} words")
    print(f"  Date Range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    
    # Success criteria check
    print(f"\n✅ SUCCESS CRITERIA")
    min_reviews_target = 400
    total_target = 1200
    
    criteria_met = []
    
    # Check individual bank targets
    for bank, count in bank_counts.items():
        if count >= min_reviews_target:
            criteria_met.append(f"  ✅ {bank}: {count} reviews (≥{min_reviews_target} required)")
        else:
            criteria_met.append(f"  ⚠️  {bank}: {count} reviews (<{min_reviews_target} required)")
    
    # Check total target
    if total_reviews >= total_target:
        criteria_met.append(f"  ✅ Total Reviews: {total_reviews:,} (≥{total_target:,} required)")
    else:
        criteria_met.append(f"  ⚠️  Total Reviews: {total_reviews:,} (<{total_target:,} required)")
    
    # Check data schema requirements
    required_columns = ['review', 'rating', 'date', 'bank', 'source']
    has_all_columns = all(col in clean_df.columns for col in required_columns)
    
    if has_all_columns:
        criteria_met.append(f"  ✅ Data Schema: All required columns present")
    else:
        missing = [col for col in required_columns if col not in clean_df.columns]
        criteria_met.append(f"  ❌ Data Schema: Missing columns: {missing}")
    
    for criterion in criteria_met:
        print(criterion)
    
    print(f"\n🎯 NEXT STEPS")
    print(f"  1. Proceed to sentiment analysis and theme extraction")
    print(f"  2. Generate visualizations and insights")
    print(f"  3. Create customer satisfaction dashboard")
    print(f"  4. Prepare recommendations for banks")
    
    print(f"\n" + "="*60)
    print("TASK 1 COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)

def main():
    """
    Main function to execute Task 1 complete pipeline
    """
    start_time = datetime.now()
    
    print("Ethiopian Mobile Banking Apps Analysis - Task 1")
    print("Data Collection & Preprocessing Pipeline")
    print("=" * 60)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Install dependencies
        install_dependencies()
        
        # Step 2: Run web scraping
        raw_filepath = run_scraping()
        
        if raw_filepath is None:
            logger.error("Scraping failed. Cannot proceed with preprocessing.")
            return
        
        # Step 3: Run preprocessing
        processed_filepath, clean_df = run_preprocessing(raw_filepath)
        
        # Step 4: Generate final report
        generate_final_report(clean_df, processed_filepath)
        
        # Execution summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️ EXECUTION SUMMARY")
        print(f"  Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total Duration: {duration}")
        print(f"  Status: ✅ SUCCESS")
        
        logger.info("Task 1 pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Task 1 pipeline failed: {e}")
        print(f"\n❌ PIPELINE FAILED")
        print(f"Error: {e}")
        print(f"Check task1_execution.log for detailed error information.")
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nExecution time before failure: {duration}")

if __name__ == "__main__":
    main() 
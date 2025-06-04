"""
Google Play Store Reviews Scraper for Ethiopian Banking Apps
Collects reviews, ratings, dates, and app information for analysis
"""

import os
import sys
import time
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from google_play_scraper import app, reviews_all, Sort
    from google_play_scraper.exceptions import NotFoundError
except ImportError:
    print("google-play-scraper not installed. Installing...")
    os.system("pip install google-play-scraper")
    from google_play_scraper import app, reviews_all, Sort
    from google_play_scraper.exceptions import NotFoundError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlayStoreScraper:
    """
    Scraper for collecting reviews from Google Play Store for Ethiopian banking apps
    """
    
    def __init__(self):
        """Initialize the scraper with bank app configurations"""
        self.banks_config = {
            'Commercial Bank of Ethiopia': {
                'package_names': [
                    'com.combanketh.mobilebanking',
                    'com.cbe.mobile',
                    'et.com.cbe.cbe_mobile',
                    'com.cbe.mobilebanking',
                    'com.cbe.ethiopia'
                ],
                'search_terms': ['CBE Mobile', 'Commercial Bank Ethiopia Mobile']
            },
            'Bank of Abyssinia': {
                'package_names': [
                    'com.boa.boaMobileBanking',
                    'com.boa.mobile',
                    'et.com.boa.mobile',
                    'com.bankofabyssinia.mobile',
                    'com.boa.ethiopia'
                ],
                'search_terms': ['BOA Mobile', 'Bank of Abyssinia Mobile']
            },
            'Dashen Bank': {
                'package_names': [
                    'com.dashen.dashensuperapp',
                    'com.dashen.mobile',
                    'et.com.dashen.mobile',
                    'com.dashenbank.mobile',
                    'com.dashen.ethiopia'
                ],
                'search_terms': ['Dashen Mobile', 'Dashen Bank Mobile']
            }
        }
        
        self.all_reviews = []
        self.target_reviews_per_bank = 400
        
    def find_app_package(self, bank_name: str) -> str:
        """
        Find the correct package name for a bank's app
        """
        logger.info(f"Searching for {bank_name} app package...")
        
        config = self.banks_config[bank_name]
        
        # Try each potential package name
        for package_name in config['package_names']:
            try:
                app_info = app(package_name)
                logger.info(f"Found app: {app_info['title']} ({package_name})")
                return package_name
            except NotFoundError:
                logger.debug(f"Package {package_name} not found")
                continue
        
        logger.warning(f"No package found for {bank_name}")
        return None
    
    def scrape_app_reviews(self, package_name: str, bank_name: str, max_reviews: int = 500) -> List[Dict]:
        """
        Scrape reviews for a specific app package
        """
        logger.info(f"Scraping reviews for {bank_name} ({package_name})...")
        
        try:
            # Get app information first
            app_info = app(package_name)
            app_title = app_info.get('title', bank_name)
            
            logger.info(f"App found: {app_title}")
            logger.info(f"Target reviews: {max_reviews}")
            
            # Scrape reviews with different sorting methods to get variety
            all_reviews = []
            
            # Try to get reviews sorted by newest first
            try:
                reviews_newest = reviews_all(
                    package_name,
                    sleep_milliseconds=0,
                    sort=Sort.NEWEST,
                    count=max_reviews // 2
                )
                all_reviews.extend(reviews_newest)
                logger.info(f"Collected {len(reviews_newest)} newest reviews")
            except Exception as e:
                logger.warning(f"Error getting newest reviews: {e}")
            
            # Try to get reviews sorted by most helpful
            try:
                reviews_helpful = reviews_all(
                    package_name,
                    sleep_milliseconds=0,
                    sort=Sort.MOST_RELEVANT,
                    count=max_reviews // 2
                )
                all_reviews.extend(reviews_helpful)
                logger.info(f"Collected {len(reviews_helpful)} most relevant reviews")
            except Exception as e:
                logger.warning(f"Error getting most relevant reviews: {e}")
            
            # Remove duplicates based on review content and user
            seen_reviews = set()
            unique_reviews = []
            
            for review in all_reviews:
                review_key = (review.get('content', ''), review.get('userName', ''))
                if review_key not in seen_reviews and review.get('content'):
                    seen_reviews.add(review_key)
                    
                    # Clean and format the review data
                    clean_review = {
                        'review': review.get('content', '').strip(),
                        'rating': review.get('score', 0),
                        'date': review.get('at', ''),
                        'bank': bank_name,
                        'app_name': app_title,
                        'source': 'Google Play',
                        'package_name': package_name,
                        'user_name': review.get('userName', ''),
                        'thumbs_up': review.get('thumbsUpCount', 0)
                    }
                    
                    unique_reviews.append(clean_review)
            
            logger.info(f"Collected {len(unique_reviews)} unique reviews for {bank_name}")
            return unique_reviews
            
        except NotFoundError:
            logger.error(f"App not found: {package_name}")
            return []
        except Exception as e:
            logger.error(f"Error scraping {package_name}: {e}")
            return []
    
    def scrape_all_banks(self) -> pd.DataFrame:
        """
        Scrape reviews for all configured banks
        """
        logger.info("Starting to scrape reviews for all banks...")
        
        all_reviews = []
        
        for bank_name in self.banks_config.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {bank_name}")
            logger.info(f"{'='*50}")
            
            package_name = self.find_app_package(bank_name)
            
            if package_name:
                bank_reviews = self.scrape_app_reviews(
                    package_name, 
                    bank_name, 
                    self.target_reviews_per_bank
                )
                all_reviews.extend(bank_reviews)
                
                logger.info(f"Collected {len(bank_reviews)} reviews for {bank_name}")
                
                # Add delay to be respectful to the API
                time.sleep(2)
            else:
                logger.warning(f"Could not find app for {bank_name}")
        
        logger.info(f"\nTotal reviews collected: {len(all_reviews)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_reviews)
        
        if not df.empty:
            # Remove any remaining duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=['review', 'bank'], keep='first')
            final_count = len(df)
            
            logger.info(f"Removed {initial_count - final_count} duplicate reviews")
            logger.info(f"Final dataset: {final_count} reviews")
        
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save raw scraped data to CSV
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_reviews_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs('data/raw', exist_ok=True)
        
        filepath = os.path.join('data', 'raw', filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Raw data saved to: {filepath}")
        
        return filepath

def main():
    """
    Main function to run the scraping process
    """
    print("Ethiopian Mobile Banking Apps Review Scraper")
    print("=" * 50)
    
    scraper = PlayStoreScraper()
    
    try:
        # Scrape all reviews
        reviews_df = scraper.scrape_all_banks()
        
        if not reviews_df.empty:
            # Save raw data
            filepath = scraper.save_raw_data(reviews_df)
            
            # Print summary statistics
            print("\n" + "="*50)
            print("SCRAPING SUMMARY")
            print("="*50)
            print(f"Total reviews collected: {len(reviews_df)}")
            print("\nReviews per bank:")
            
            bank_counts = reviews_df['bank'].value_counts()
            for bank, count in bank_counts.items():
                print(f"  {bank}: {count} reviews")
            
            print(f"\nRating distribution:")
            rating_dist = reviews_df['rating'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                print(f"  {rating} stars: {count} reviews")
            
            print(f"\nData saved to: {filepath}")
            
        else:
            print("No reviews were collected. Please check the app package names.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
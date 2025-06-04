"""
Sample Data Generator for Ethiopian Banking App Reviews
Creates realistic demo data for testing when live scraping is limited
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

class SampleDataGenerator:
    """
    Generate realistic sample review data for Ethiopian banking apps
    """
    
    def __init__(self):
        self.banks = [
            'Commercial Bank of Ethiopia',
            'Bank of Abyssinia', 
            'Dashen Bank'
        ]
        
        # Sample review templates with various sentiments
        self.positive_reviews = [
            "Excellent mobile banking app! Very user-friendly interface and fast transactions.",
            "Love the new features. Makes banking so much easier than going to the branch.",
            "Quick and secure payments. The app works perfectly on my phone.",
            "Outstanding customer service through the app. Highly recommended!",
            "Best banking app I've used. Transaction history is very clear and detailed.",
            "Fast money transfer feature is amazing. No more long queues at the bank!",
            "Very convenient for paying bills and checking account balance anytime.",
            "The app is stable and reliable. Never had any technical issues.",
            "Great user experience! The design is modern and intuitive.",
            "Perfect for daily banking needs. All features work as expected.",
            "Smooth and efficient. Customer support is very responsive.",
            "Amazing app! Makes mobile banking incredibly easy and secure.",
        ]
        
        self.negative_reviews = [
            "App keeps crashing when I try to make payments. Very frustrating!",
            "Login issues persist for weeks. Cannot access my account properly.",
            "Very slow loading times. Takes forever to complete simple transactions.",
            "Poor customer service. No response to my complaints about app errors.",
            "The app frequently logs me out during transactions. Needs major improvements.",
            "Cannot transfer money - app shows error messages constantly.",
            "Outdated interface and confusing navigation. Needs complete redesign.",
            "Security concerns - app doesn't feel safe for financial transactions.",
            "Bill payment feature never works properly. Always shows server errors.",
            "Worst banking app experience! Constantly freezes and crashes.",
            "Unable to check account balance. App is unreliable and buggy.",
            "Terrible user experience. The app is slow and unresponsive.",
        ]
        
        self.neutral_reviews = [
            "App works fine but could use some improvements in the user interface.",
            "Decent banking app with basic features. Nothing special but gets the job done.",
            "Average mobile banking experience. Some features work better than others.",
            "The app is okay for simple transactions but lacks advanced features.",
            "Mixed experience - some features are good while others need work.",
            "Basic functionality is there but the design could be more modern.",
            "App works as expected. Would like to see more payment options added.",
            "Standard banking app with room for improvement in user experience.",
            "Functional but not exceptional. Meets basic banking needs adequately.",
            "Acceptable app performance though it could be faster and more responsive.",
            "Regular banking app - works fine for checking balance and simple transfers.",
            "Ordinary mobile banking experience. Nothing outstanding but functional.",
        ]
        
    def generate_review_text(self, rating):
        """Generate review text based on rating"""
        if rating >= 4:
            return random.choice(self.positive_reviews)
        elif rating <= 2:
            return random.choice(self.negative_reviews)
        else:
            return random.choice(self.neutral_reviews)
    
    def generate_sample_data(self, reviews_per_bank=450) -> pd.DataFrame:
        """
        Generate comprehensive sample dataset
        """
        all_reviews = []
        
        for bank in self.banks:
            print(f"Generating {reviews_per_bank} reviews for {bank}...")
            
            for i in range(reviews_per_bank):
                # Generate rating with realistic distribution
                rating = np.random.choice(
                    [1, 2, 3, 4, 5], 
                    p=[0.1, 0.15, 0.25, 0.35, 0.15]  # Slightly skewed toward positive
                )
                
                # Generate review text based on rating
                review_text = self.generate_review_text(rating)
                
                # Add some variation to reviews
                if random.random() < 0.3:  # 30% chance to modify
                    variations = [
                        f"{review_text} Update: Still having the same experience.",
                        f"{review_text} Been using this app for months now.",
                        f"{review_text} Comparing to other banks, this is my experience.",
                        f"{review_text} Overall satisfied with the service.",
                        f"{review_text} Hope they continue to improve.",
                    ]
                    review_text = random.choice(variations)
                
                # Generate random date within last 2 years
                start_date = datetime.now() - timedelta(days=730)
                end_date = datetime.now() - timedelta(days=1)
                random_date = start_date + timedelta(
                    days=random.randint(0, (end_date - start_date).days)
                )
                
                review_data = {
                    'review': review_text,
                    'rating': rating,
                    'date': random_date.strftime('%Y-%m-%d'),
                    'bank': bank,
                    'app_name': f"{bank} Mobile Banking",
                    'source': 'Google Play',
                    'package_name': f"com.{bank.lower().replace(' ', '').replace('of', '').replace('bank', '')}.mobile",
                    'user_name': f"User{random.randint(1000, 9999)}",
                    'thumbs_up': random.randint(0, 20)
                }
                
                all_reviews.append(review_data)
        
        df = pd.DataFrame(all_reviews)
        
        # Add some realistic noise and variations
        df = self._add_realistic_variations(df)
        
        print(f"\nGenerated {len(df)} total sample reviews")
        return df
    
    def _add_realistic_variations(self, df):
        """Add realistic variations to make data more authentic"""
        
        # Add some duplicate-like reviews (but slightly different)
        duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        for idx in duplicate_indices:
            original_review = df.loc[idx, 'review']
            variations = [
                f"{original_review} Posting again to emphasize.",
                f"Update: {original_review}",
                f"{original_review} Still my opinion.",
            ]
            df.loc[idx, 'review'] = random.choice(variations)
        
        # Add some reviews with mixed ratings (edge cases)
        edge_indices = np.random.choice(df.index, size=int(len(df) * 0.1), replace=False)
        for idx in edge_indices:
            # Sometimes positive text with lower rating or vice versa
            if random.random() < 0.5:
                df.loc[idx, 'review'] = random.choice(self.positive_reviews)
                df.loc[idx, 'rating'] = np.random.choice([2, 3])  # Mixed signal
        
        return df
    
    def save_sample_data(self, df, filename=None):
        """Save sample data as raw data file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_reviews_sample_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs('data/raw', exist_ok=True)
        
        filepath = os.path.join('data', 'raw', filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Sample data saved to: {filepath}")
        return filepath

def main():
    """Generate sample data for testing"""
    print("Ethiopian Banking App Sample Data Generator")
    print("=" * 50)
    
    generator = SampleDataGenerator()
    
    # Generate sample data (450 reviews per bank = 1350 total)
    sample_df = generator.generate_sample_data(reviews_per_bank=450)
    
    # Save the data
    filepath = generator.save_sample_data(sample_df)
    
    # Print summary
    print("\nSample Data Summary:")
    print(f"Total reviews: {len(sample_df)}")
    
    print("\nReviews per bank:")
    bank_counts = sample_df['bank'].value_counts()
    for bank, count in bank_counts.items():
        print(f"  {bank}: {count} reviews")
    
    print("\nRating distribution:")
    rating_dist = sample_df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"  {rating} stars: {count} reviews")
    
    print(f"\nDate range: {sample_df['date'].min()} to {sample_df['date'].max()}")
    print(f"\nFile saved to: {filepath}")
    
    return filepath

if __name__ == "__main__":
    main() 
"""
Sentiment Analysis Module for Ethiopian Banking App Reviews
Implements multiple sentiment analysis approaches for comprehensive analysis
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis using multiple approaches:
    1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
    2. TextBlob (Pattern-based sentiment analysis)
    3. DistilBERT (Transformer-based deep learning model)
    """
    
    def __init__(self):
        """Initialize sentiment analysis models"""
        self.vader_analyzer = None
        self.distilbert_pipeline = None
        self.models_loaded = False
        
        # Initialize sentiment categories
        self.sentiment_mapping = {
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative', 
            'NEUTRAL': 'Neutral'
        }
        
    def _load_models(self):
        """Load all sentiment analysis models"""
        try:
            logger.info("Loading sentiment analysis models...")
            
            # Load VADER
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("✅ VADER sentiment analyzer loaded")
            except ImportError:
                logger.warning("VADER not available. Installing...")
                import subprocess
                subprocess.check_call(["pip", "install", "vaderSentiment"])
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Load TextBlob (no explicit loading needed)
            try:
                from textblob import TextBlob
                # Test TextBlob
                test_blob = TextBlob("test")
                logger.info("✅ TextBlob sentiment analyzer ready")
            except ImportError:
                logger.warning("TextBlob not available")
            
            # Load DistilBERT
            try:
                from transformers import pipeline
                self.distilbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                logger.info("✅ DistilBERT sentiment model loaded")
            except Exception as e:
                logger.warning(f"DistilBERT not available: {e}")
                logger.info("Will proceed with VADER and TextBlob only")
            
            self.models_loaded = True
            logger.info("Sentiment analysis models initialization complete")
            
        except Exception as e:
            logger.error(f"Error loading sentiment models: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning while preserving sentiment-relevant features
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        if not self.vader_analyzer:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0, 'label': 'Neutral'}
        
        text = self._preprocess_text(text)
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Classify based on compound score
        if scores['compound'] >= 0.05:
            label = 'Positive'
        elif scores['compound'] <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'vader_compound': scores['compound'],
            'vader_positive': scores['pos'],
            'vader_neutral': scores['neu'],
            'vader_negative': scores['neg'],
            'vader_label': label
        }
    
    def analyze_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        try:
            from textblob import TextBlob
            
            text = self._preprocess_text(text)
            blob = TextBlob(text)
            
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify based on polarity
            if polarity > 0.1:
                label = 'Positive'
            elif polarity < -0.1:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            return {
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity,
                'textblob_label': label
            }
            
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'textblob_label': 'Neutral'
            }
    
    def analyze_distilbert_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using DistilBERT"""
        if not self.distilbert_pipeline:
            return {
                'distilbert_positive': 0.0,
                'distilbert_negative': 0.0,
                'distilbert_label': 'Neutral',
                'distilbert_confidence': 0.0
            }
        
        try:
            text = self._preprocess_text(text)
            
            # Handle empty text
            if not text.strip():
                return {
                    'distilbert_positive': 0.0,
                    'distilbert_negative': 0.0,
                    'distilbert_label': 'Neutral',
                    'distilbert_confidence': 0.0
                }
            
            # Truncate very long texts to avoid model limits
            if len(text) > 512:
                text = text[:512]
            
            results = self.distilbert_pipeline(text)
            
            # Extract scores
            positive_score = 0.0
            negative_score = 0.0
            
            for result in results[0]:
                if result['label'] == 'POSITIVE':
                    positive_score = result['score']
                elif result['label'] == 'NEGATIVE':
                    negative_score = result['score']
            
            # Determine label and confidence
            if positive_score > negative_score:
                label = 'Positive'
                confidence = positive_score
            elif negative_score > positive_score:
                label = 'Negative'
                confidence = negative_score
            else:
                label = 'Neutral'
                confidence = max(positive_score, negative_score)
            
            return {
                'distilbert_positive': positive_score,
                'distilbert_negative': negative_score,
                'distilbert_label': label,
                'distilbert_confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"DistilBERT analysis failed: {e}")
            return {
                'distilbert_positive': 0.0,
                'distilbert_negative': 0.0,
                'distilbert_label': 'Neutral',
                'distilbert_confidence': 0.0
            }
    
    def analyze_comprehensive_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis using all available methods"""
        if not self.models_loaded:
            self._load_models()
        
        # Get results from all methods
        vader_results = self.analyze_vader_sentiment(text)
        textblob_results = self.analyze_textblob_sentiment(text)
        distilbert_results = self.analyze_distilbert_sentiment(text)
        
        # Combine results
        combined_results = {**vader_results, **textblob_results, **distilbert_results}
        
        # Create ensemble prediction
        labels = [
            vader_results.get('vader_label', 'Neutral'),
            textblob_results.get('textblob_label', 'Neutral'),
            distilbert_results.get('distilbert_label', 'Neutral')
        ]
        
        # Simple majority voting for ensemble
        from collections import Counter
        label_counts = Counter(labels)
        ensemble_label = label_counts.most_common(1)[0][0]
        
        # Calculate ensemble confidence
        ensemble_confidence = label_counts[ensemble_label] / len(labels)
        
        combined_results.update({
            'ensemble_label': ensemble_label,
            'ensemble_confidence': ensemble_confidence
        })
        
        return combined_results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'review') -> pd.DataFrame:
        """Analyze sentiment for entire dataframe"""
        logger.info(f"Starting sentiment analysis for {len(df)} reviews...")
        
        if not self.models_loaded:
            self._load_models()
        
        # Create results lists
        sentiment_results = []
        
        # Progress tracking
        from tqdm import tqdm
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiment"):
            text = row[text_column]
            results = self.analyze_comprehensive_sentiment(text)
            results['review_id'] = idx
            sentiment_results.append(results)
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        
        # Merge with original dataframe
        result_df = df.copy()
        for col in sentiment_df.columns:
            if col != 'review_id':
                result_df[col] = sentiment_df[col].values
        
        logger.info("Sentiment analysis completed successfully!")
        return result_df
    
    def aggregate_sentiment_by_bank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment results by bank"""
        logger.info("Aggregating sentiment by bank...")
        
        aggregation_cols = [
            'vader_compound', 'vader_positive', 'vader_neutral', 'vader_negative',
            'textblob_polarity', 'textblob_subjectivity',
            'distilbert_positive', 'distilbert_negative', 'distilbert_confidence',
            'ensemble_confidence'
        ]
        
        # Filter existing columns
        existing_agg_cols = [col for col in aggregation_cols if col in df.columns]
        
        bank_sentiment = df.groupby('bank')[existing_agg_cols].mean().round(3)
        
        # Add label distributions
        for label_col in ['vader_label', 'textblob_label', 'distilbert_label', 'ensemble_label']:
            if label_col in df.columns:
                label_dist = df.groupby('bank')[label_col].value_counts(normalize=True).unstack(fill_value=0)
                label_dist.columns = [f"{label_col}_{col.lower()}" for col in label_dist.columns]
                bank_sentiment = bank_sentiment.join(label_dist, how='left')
        
        return bank_sentiment
    
    def aggregate_sentiment_by_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment results by rating"""
        logger.info("Aggregating sentiment by rating...")
        
        aggregation_cols = [
            'vader_compound', 'vader_positive', 'vader_neutral', 'vader_negative',
            'textblob_polarity', 'textblob_subjectivity',
            'distilbert_positive', 'distilbert_negative', 'distilbert_confidence',
            'ensemble_confidence'
        ]
        
        # Filter existing columns
        existing_agg_cols = [col for col in aggregation_cols if col in df.columns]
        
        rating_sentiment = df.groupby(['bank', 'rating'])[existing_agg_cols].mean().round(3)
        
        return rating_sentiment
    
    def generate_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive sentiment analysis summary"""
        logger.info("Generating sentiment analysis summary...")
        
        summary = {
            'total_reviews': len(df),
            'analysis_timestamp': datetime.now().isoformat(),
            'models_used': []
        }
        
        # Check which models were used
        if 'vader_label' in df.columns:
            summary['models_used'].append('VADER')
        if 'textblob_label' in df.columns:
            summary['models_used'].append('TextBlob')
        if 'distilbert_label' in df.columns:
            summary['models_used'].append('DistilBERT')
        
        # Overall sentiment distribution
        if 'ensemble_label' in df.columns:
            sentiment_dist = df['ensemble_label'].value_counts(normalize=True)
            summary['overall_sentiment_distribution'] = sentiment_dist.to_dict()
        
        # Bank-wise sentiment
        if 'bank' in df.columns and 'ensemble_label' in df.columns:
            bank_sentiment = df.groupby('bank')['ensemble_label'].value_counts(normalize=True).unstack(fill_value=0)
            summary['bank_sentiment_distribution'] = bank_sentiment.to_dict('index')
        
        # Rating correlation
        if 'rating' in df.columns and 'vader_compound' in df.columns:
            rating_correlation = df[['rating', 'vader_compound']].corr().iloc[0, 1]
            summary['rating_sentiment_correlation'] = rating_correlation
        
        return summary

def main():
    """Main function to test sentiment analysis"""
    print("Ethiopian Banking Apps Sentiment Analysis - Task 2")
    print("=" * 60)
    
    try:
        # Load processed data from Task 1
        import glob
        processed_files = glob.glob('data/processed/processed_reviews_*.csv')
        
        if not processed_files:
            print("No processed data found. Please run Task 1 first.")
            return
        
        # Use the most recent file
        latest_file = max(processed_files, key=lambda x: x.split('_')[-1])
        print(f"Loading data from: {latest_file}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} reviews for sentiment analysis")
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Perform sentiment analysis
        result_df = analyzer.analyze_dataframe(df)
        
        # Generate aggregations
        bank_sentiment = analyzer.aggregate_sentiment_by_bank(result_df)
        rating_sentiment = analyzer.aggregate_sentiment_by_rating(result_df)
        summary = analyzer.generate_sentiment_summary(result_df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        output_file = f'data/final/sentiment_analysis_{timestamp}.csv'
        result_df.to_csv(output_file, index=False)
        print(f"Detailed sentiment analysis saved to: {output_file}")
        
        # Save aggregations
        bank_file = f'data/final/bank_sentiment_summary_{timestamp}.csv'
        bank_sentiment.to_csv(bank_file)
        print(f"Bank sentiment summary saved to: {bank_file}")
        
        rating_file = f'data/final/rating_sentiment_summary_{timestamp}.csv'
        rating_sentiment.to_csv(rating_file)
        print(f"Rating sentiment summary saved to: {rating_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total reviews analyzed: {summary['total_reviews']}")
        print(f"Models used: {', '.join(summary['models_used'])}")
        
        if 'overall_sentiment_distribution' in summary:
            print("\nOverall Sentiment Distribution:")
            for sentiment, percentage in summary['overall_sentiment_distribution'].items():
                print(f"  {sentiment}: {percentage:.1%}")
        
        if 'rating_sentiment_correlation' in summary:
            print(f"\nRating-Sentiment Correlation: {summary['rating_sentiment_correlation']:.3f}")
        
        print("\n✅ Sentiment Analysis Completed Successfully!")
        
        return result_df, bank_sentiment, rating_sentiment, summary
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
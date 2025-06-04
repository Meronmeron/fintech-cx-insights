"""
Thematic Analysis Module for Ethiopian Banking App Reviews
Extracts keywords, n-grams, and groups them into meaningful themes
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Set, Any
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThemeAnalyzer:
    """
    Comprehensive thematic analysis for banking app reviews
    Features:
    1. Keyword extraction using TF-IDF
    2. N-gram extraction for phrase identification
    3. Theme clustering and categorization
    4. Bank-specific theme analysis
    """
    
    def __init__(self):
        """Initialize thematic analysis components"""
        self.vectorizer = None
        self.nlp = None
        self.stop_words = None
        self.banking_stopwords = None
        self.models_loaded = False
        
        # Pre-defined banking themes framework
        self.theme_keywords = {
            'Account Access Issues': [
                'login', 'password', 'authentication', 'access', 'account', 'lock', 'unlock',
                'signin', 'sign in', 'cant login', 'failed login', 'unable to access',
                'locked out', 'forgot password', 'reset password', 'otp', 'verification'
            ],
            'Transaction Performance': [
                'transfer', 'payment', 'transaction', 'money', 'send', 'receive', 'balance',
                'slow transfer', 'failed transaction', 'pending', 'processing', 'speed',
                'fast', 'instant', 'delay', 'timeout', 'error', 'success', 'complete'
            ],
            'User Interface & Experience': [
                'ui', 'interface', 'design', 'layout', 'menu', 'navigation', 'easy',
                'difficult', 'confusing', 'clear', 'simple', 'complicated', 'user friendly',
                'intuitive', 'beautiful', 'ugly', 'modern', 'outdated', 'responsive'
            ],
            'Technical Issues': [
                'crash', 'bug', 'error', 'freeze', 'hang', 'laggy', 'slow', 'glitch',
                'problem', 'issue', 'broken', 'fix', 'update', 'version', 'install',
                'loading', 'connection', 'network', 'server', 'maintenance'
            ],
            'Customer Support': [
                'support', 'help', 'service', 'customer', 'staff', 'agent', 'response',
                'helpful', 'unhelpful', 'rude', 'polite', 'quick', 'slow response',
                'contact', 'call', 'email', 'chat', 'assistance', 'resolve'
            ],
            'Security & Trust': [
                'security', 'safe', 'secure', 'trust', 'privacy', 'protection', 'fraud',
                'scam', 'hack', 'biometric', 'fingerprint', 'pin', 'encryption',
                'worried', 'concern', 'confident', 'reliable', 'trustworthy'
            ],
            'Features & Functionality': [
                'feature', 'function', 'service', 'bill pay', 'airtime', 'utility',
                'loan', 'savings', 'investment', 'statement', 'history', 'notification',
                'alert', 'sms', 'email notification', 'mobile banking', 'ATM', 'card'
            ]
        }
        
        # Banking-specific stopwords to add to standard ones
        self.banking_stopwords = {
            'app', 'bank', 'banking', 'mobile', 'phone', 'application', 'system',
            'commercial', 'abyssinia', 'dashen', 'ethiopia', 'ethiopian', 'birr'
        }
    
    def _load_models(self):
        """Load NLP models and components"""
        try:
            logger.info("Loading thematic analysis models...")
            
            # Load TF-IDF Vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Initialize stop words
            try:
                import nltk
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                from nltk.corpus import stopwords
                self.stop_words = set(stopwords.words('english'))
                logger.info("✅ NLTK stopwords loaded")
            except:
                # Fallback to basic English stopwords
                self.stop_words = {
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                    'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
                    'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                    'further', 'then', 'once'
                }
                logger.info("✅ Basic stopwords loaded")
            
            # Add banking-specific stopwords
            self.stop_words.update(self.banking_stopwords)
            
            # Initialize TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=list(self.stop_words),
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8,
                lowercase=True
            )
            logger.info("✅ TF-IDF vectorizer initialized")
            
            # Try to load spaCy
            try:
                import spacy
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("✅ spaCy model loaded")
            except (ImportError, OSError):
                logger.warning("spaCy not available, using basic tokenization")
                self.nlp = None
            
            self.models_loaded = True
            logger.info("Thematic analysis models initialization complete")
            
        except Exception as e:
            logger.error(f"Error loading thematic models: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for thematic analysis"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_keywords_tfidf(self, texts: List[str], max_features: int = 50) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        if not self.models_loaded:
            self._load_models()
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text.strip()]
        
        if not processed_texts:
            return []
        
        try:
            # Fit TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, mean_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:max_features]
            
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            return []
    
    def extract_keywords_spacy(self, texts: List[str], max_features: int = 50) -> List[Tuple[str, float]]:
        """Extract keywords using spaCy NER and POS tagging"""
        if not self.nlp:
            return []
        
        keyword_counts = Counter()
        total_docs = len(texts)
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                continue
                
            try:
                doc = self.nlp(self._preprocess_text(text))
                
                # Extract meaningful tokens
                for token in doc:
                    if (token.is_alpha and 
                        not token.is_stop and 
                        len(token.text) > 2 and
                        token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV'] and
                        token.text.lower() not in self.stop_words):
                        keyword_counts[token.lemma_.lower()] += 1
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) >= 2 and len(chunk.text.split()) <= 3:
                        clean_chunk = ' '.join([token.lemma_.lower() for token in chunk 
                                              if not token.is_stop and token.is_alpha])
                        if clean_chunk and len(clean_chunk) > 3:
                            keyword_counts[clean_chunk] += 1
                            
            except Exception as e:
                logger.warning(f"spaCy processing failed for text: {e}")
                continue
        
        # Convert to relative frequencies
        keyword_scores = [(word, count/total_docs) for word, count in keyword_counts.most_common(max_features)]
        
        return keyword_scores
    
    def extract_ngrams(self, texts: List[str], n: int = 2, max_features: int = 30) -> List[Tuple[str, int]]:
        """Extract n-grams from texts"""
        if not texts:
            return []
        
        ngram_counts = Counter()
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                continue
            
            processed = self._preprocess_text(text)
            words = processed.split()
            
            # Filter out stop words
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            # Generate n-grams
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if len(ngram) > 3:  # Filter very short ngrams
                    ngram_counts[ngram] += 1
        
        return ngram_counts.most_common(max_features)
    
    def classify_themes(self, keywords: List[str], confidence_threshold: float = 0.3) -> Dict[str, Dict[str, Any]]:
        """Classify keywords into predefined themes"""
        theme_scores = defaultdict(lambda: {'keywords': [], 'score': 0.0, 'matches': 0})
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            best_theme = None
            best_score = 0.0
            
            for theme, theme_keywords in self.theme_keywords.items():
                # Calculate similarity score
                score = 0.0
                
                # Exact matches
                if keyword_lower in [tk.lower() for tk in theme_keywords]:
                    score += 1.0
                
                # Partial matches
                for theme_keyword in theme_keywords:
                    if theme_keyword.lower() in keyword_lower or keyword_lower in theme_keyword.lower():
                        score += 0.5
                
                # Word overlap for multi-word keywords
                if ' ' in keyword_lower:
                    keyword_words = set(keyword_lower.split())
                    for theme_keyword in theme_keywords:
                        theme_words = set(theme_keyword.lower().split())
                        overlap = len(keyword_words.intersection(theme_words))
                        if overlap > 0:
                            score += overlap * 0.3
                
                if score > best_score:
                    best_score = score
                    best_theme = theme
            
            # Assign to theme if above threshold
            if best_theme and best_score >= confidence_threshold:
                theme_scores[best_theme]['keywords'].append(keyword)
                theme_scores[best_theme]['score'] += best_score
                theme_scores[best_theme]['matches'] += 1
        
        # Normalize scores
        for theme in theme_scores:
            if theme_scores[theme]['matches'] > 0:
                theme_scores[theme]['score'] /= theme_scores[theme]['matches']
        
        return dict(theme_scores)
    
    def analyze_themes_by_bank(self, df: pd.DataFrame, text_column: str = 'review') -> Dict[str, Any]:
        """Analyze themes for each bank separately"""
        logger.info("Starting bank-specific thematic analysis...")
        
        if not self.models_loaded:
            self._load_models()
        
        bank_themes = {}
        
        for bank in df['bank'].unique():
            logger.info(f"Analyzing themes for {bank}...")
            
            bank_data = df[df['bank'] == bank]
            bank_texts = bank_data[text_column].dropna().tolist()
            
            if not bank_texts:
                continue
            
            # Extract keywords using multiple methods
            tfidf_keywords = self.extract_keywords_tfidf(bank_texts, max_features=30)
            spacy_keywords = self.extract_keywords_spacy(bank_texts, max_features=30) if self.nlp else []
            
            # Extract n-grams
            bigrams = self.extract_ngrams(bank_texts, n=2, max_features=20)
            trigrams = self.extract_ngrams(bank_texts, n=3, max_features=15)
            
            # Combine all keywords
            all_keywords = []
            all_keywords.extend([kw[0] for kw in tfidf_keywords[:20]])
            all_keywords.extend([kw[0] for kw in spacy_keywords[:20]])
            all_keywords.extend([kw[0] for kw in bigrams[:15]])
            all_keywords.extend([kw[0] for kw in trigrams[:10]])
            
            # Remove duplicates while preserving order
            unique_keywords = []
            seen = set()
            for kw in all_keywords:
                if kw not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw)
            
            # Classify into themes
            theme_classification = self.classify_themes(unique_keywords)
            
            # Store results
            bank_themes[bank] = {
                'total_reviews': len(bank_texts),
                'tfidf_keywords': tfidf_keywords,
                'spacy_keywords': spacy_keywords,
                'bigrams': bigrams,
                'trigrams': trigrams,
                'themes': theme_classification,
                'top_keywords': unique_keywords[:25]
            }
        
        logger.info("Bank-specific thematic analysis completed")
        return bank_themes
    
    def analyze_sentiment_themes(self, df: pd.DataFrame, text_column: str = 'review') -> Dict[str, Any]:
        """Analyze themes by sentiment to understand satisfaction drivers"""
        if 'ensemble_label' not in df.columns:
            logger.warning("No sentiment labels found. Run sentiment analysis first.")
            return {}
        
        logger.info("Analyzing themes by sentiment...")
        
        sentiment_themes = {}
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            sentiment_data = df[df['ensemble_label'] == sentiment]
            
            if len(sentiment_data) == 0:
                continue
            
            sentiment_texts = sentiment_data[text_column].dropna().tolist()
            
            # Extract keywords for this sentiment
            tfidf_keywords = self.extract_keywords_tfidf(sentiment_texts, max_features=25)
            bigrams = self.extract_ngrams(sentiment_texts, n=2, max_features=15)
            
            # Combine keywords
            all_keywords = [kw[0] for kw in tfidf_keywords[:15]] + [kw[0] for kw in bigrams[:10]]
            
            # Classify themes
            theme_classification = self.classify_themes(all_keywords)
            
            sentiment_themes[sentiment] = {
                'review_count': len(sentiment_texts),
                'top_keywords': tfidf_keywords[:15],
                'top_bigrams': bigrams[:10],
                'themes': theme_classification
            }
        
        return sentiment_themes
    
    def generate_theme_insights(self, bank_themes: Dict[str, Any], sentiment_themes: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate actionable insights from thematic analysis"""
        logger.info("Generating thematic insights...")
        
        insights = {
            'analysis_timestamp': datetime.now().isoformat(),
            'banks_analyzed': list(bank_themes.keys()),
            'total_themes_identified': len(self.theme_keywords),
            'bank_specific_insights': {},
            'cross_bank_patterns': {},
            'recommendations': []
        }
        
        # Bank-specific insights
        for bank, data in bank_themes.items():
            bank_insights = {
                'review_count': data['total_reviews'],
                'primary_themes': [],
                'concern_areas': [],
                'positive_aspects': []
            }
            
            # Identify primary themes (those with most keywords)
            theme_scores = {}
            for theme, theme_data in data['themes'].items():
                theme_scores[theme] = theme_data['matches']
            
            # Sort themes by prevalence
            sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
            bank_insights['primary_themes'] = sorted_themes[:3]
            
            insights['bank_specific_insights'][bank] = bank_insights
        
        # Cross-bank patterns
        all_themes = defaultdict(int)
        for bank_data in bank_themes.values():
            for theme, theme_data in bank_data['themes'].items():
                all_themes[theme] += theme_data['matches']
        
        insights['cross_bank_patterns'] = {
            'most_common_themes': sorted(all_themes.items(), key=lambda x: x[1], reverse=True)[:5],
            'industry_wide_concerns': [theme for theme, count in all_themes.items() if count >= len(bank_themes)]
        }
        
        # Generate recommendations
        recommendations = []
        
        # Check for common issues
        if 'Technical Issues' in all_themes and all_themes['Technical Issues'] > 5:
            recommendations.append("Priority: Address app stability and technical issues across all platforms")
        
        if 'Account Access Issues' in all_themes and all_themes['Account Access Issues'] > 3:
            recommendations.append("Focus: Improve authentication and login experience")
        
        if 'User Interface & Experience' in all_themes:
            recommendations.append("Enhancement: Continue UI/UX improvements based on user feedback")
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def save_theme_analysis(self, bank_themes: Dict[str, Any], sentiment_themes: Dict[str, Any], 
                           insights: Dict[str, Any], timestamp: str = None) -> Dict[str, str]:
        """Save thematic analysis results"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        import json
        import os
        
        # Create results directory if it doesn't exist
        os.makedirs('data/final', exist_ok=True)
        
        saved_files = {}
        
        # Save bank themes
        bank_themes_file = f'data/final/bank_themes_{timestamp}.json'
        with open(bank_themes_file, 'w', encoding='utf-8') as f:
            json.dump(bank_themes, f, indent=2, ensure_ascii=False, default=str)
        saved_files['bank_themes'] = bank_themes_file
        
        # Save sentiment themes if available
        if sentiment_themes:
            sentiment_themes_file = f'data/final/sentiment_themes_{timestamp}.json'
            with open(sentiment_themes_file, 'w', encoding='utf-8') as f:
                json.dump(sentiment_themes, f, indent=2, ensure_ascii=False, default=str)
            saved_files['sentiment_themes'] = sentiment_themes_file
        
        # Save insights
        insights_file = f'data/final/theme_insights_{timestamp}.json'
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
        saved_files['insights'] = insights_file
        
        logger.info(f"Thematic analysis results saved: {list(saved_files.keys())}")
        return saved_files

def main():
    """Main function to test thematic analysis"""
    print("Ethiopian Banking Apps Thematic Analysis - Task 2")
    print("=" * 60)
    
    try:
        # Load processed data
        import glob
        processed_files = glob.glob('data/processed/processed_reviews_*.csv')
        
        if not processed_files:
            print("No processed data found. Please run Task 1 first.")
            return
        
        # Use the most recent file
        latest_file = max(processed_files, key=lambda x: x.split('_')[-1])
        print(f"Loading data from: {latest_file}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} reviews for thematic analysis")
        
        # Initialize analyzer
        analyzer = ThemeAnalyzer()
        
        # Perform thematic analysis
        bank_themes = analyzer.analyze_themes_by_bank(df)
        
        # Analyze sentiment themes if sentiment data exists
        sentiment_themes = analyzer.analyze_sentiment_themes(df)
        
        # Generate insights
        insights = analyzer.generate_theme_insights(bank_themes, sentiment_themes)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = analyzer.save_theme_analysis(bank_themes, sentiment_themes, insights, timestamp)
        
        # Print summary
        print("\n" + "="*60)
        print("THEMATIC ANALYSIS SUMMARY")
        print("="*60)
        print(f"Banks analyzed: {len(bank_themes)}")
        print(f"Themes framework: {len(analyzer.theme_keywords)} predefined themes")
        
        for bank, data in bank_themes.items():
            print(f"\n{bank}:")
            print(f"  Reviews: {data['total_reviews']}")
            print(f"  Top themes: {len(data['themes'])}")
            if data['themes']:
                top_theme = max(data['themes'].items(), key=lambda x: x[1]['matches'])
                print(f"  Primary theme: {top_theme[0]} ({top_theme[1]['matches']} matches)")
        
        print(f"\nResults saved to: {list(saved_files.values())}")
        print("\n✅ Thematic Analysis Completed Successfully!")
        
        return bank_themes, sentiment_themes, insights
        
    except Exception as e:
        logger.error(f"Error in thematic analysis: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
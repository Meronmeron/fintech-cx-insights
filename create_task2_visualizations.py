"""
Task 2 Visualization Script: Sentiment and Thematic Analysis Dashboard
Creates comprehensive visualizations and reports for Ethiopian Banking Apps analysis
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_latest_results():
    """Load the most recent Task 2 analysis results"""
    try:
        # Load sentiment analysis results
        sentiment_files = glob.glob('data/final/sentiment_analysis_*.csv')
        if not sentiment_files:
            raise FileNotFoundError("No sentiment analysis results found")
        
        latest_sentiment = max(sentiment_files, key=lambda x: x.split('_')[-1])
        df = pd.read_csv(latest_sentiment)
        print(f"✅ Loaded sentiment data from: {latest_sentiment}")
        
        # Load bank sentiment summary
        bank_sentiment_files = glob.glob('data/final/bank_sentiment_summary_*.csv')
        if bank_sentiment_files:
            latest_bank_sentiment = max(bank_sentiment_files, key=lambda x: x.split('_')[-1])
            bank_sentiment = pd.read_csv(latest_bank_sentiment, index_col=0)
            print(f"✅ Loaded bank sentiment summary from: {latest_bank_sentiment}")
        else:
            bank_sentiment = None
        
        # Load thematic analysis results
        theme_files = glob.glob('data/final/bank_themes_*.json')
        if theme_files:
            latest_themes = max(theme_files, key=lambda x: x.split('_')[-1])
            with open(latest_themes, 'r', encoding='utf-8') as f:
                bank_themes = json.load(f)
            print(f"✅ Loaded thematic analysis from: {latest_themes}")
        else:
            bank_themes = None
        
        # Load integrated insights
        insights_files = glob.glob('data/final/integrated_insights_*.json')
        if insights_files:
            latest_insights = max(insights_files, key=lambda x: x.split('_')[-1])
            with open(latest_insights, 'r', encoding='utf-8') as f:
                insights = json.load(f)
            print(f"✅ Loaded integrated insights from: {latest_insights}")
        else:
            insights = None
        
        return df, bank_sentiment, bank_themes, insights
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None, None, None

def create_sentiment_overview(df):
    """Create sentiment analysis overview visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 Sentiment Analysis Overview - Ethiopian Banking Apps', fontsize=16, fontweight='bold')
    
    # 1. Overall Sentiment Distribution
    if 'ensemble_label' in df.columns:
        sentiment_counts = df['ensemble_label'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Gold, Red
        
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 0].set_title('Overall Sentiment Distribution', fontweight='bold')
    
    # 2. Sentiment by Bank
    if 'ensemble_label' in df.columns and 'bank' in df.columns:
        sentiment_bank = pd.crosstab(df['bank'], df['ensemble_label'], normalize='index') * 100
        sentiment_bank.plot(kind='bar', ax=axes[0, 1], color=colors)
        axes[0, 1].set_title('Sentiment Distribution by Bank (%)', fontweight='bold')
        axes[0, 1].set_xlabel('Bank')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].legend(title='Sentiment')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Rating vs Sentiment Correlation
    if 'rating' in df.columns and 'vader_compound' in df.columns:
        axes[1, 0].scatter(df['rating'], df['vader_compound'], alpha=0.6, c=df['rating'], cmap='viridis')
        axes[1, 0].set_xlabel('Rating (Stars)')
        axes[1, 0].set_ylabel('VADER Sentiment Score')
        axes[1, 0].set_title('Rating vs Sentiment Score Correlation', fontweight='bold')
        
        # Add trend line
        z = np.polyfit(df['rating'], df['vader_compound'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(df['rating'], p(df['rating']), "r--", alpha=0.8)
    
    # 4. Sentiment Evolution Over Time
    if 'date' in df.columns and 'ensemble_label' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month_year'] = df['date'].dt.to_period('M')
        
        sentiment_time = df.groupby(['month_year', 'ensemble_label']).size().unstack(fill_value=0)
        sentiment_time_pct = sentiment_time.div(sentiment_time.sum(axis=1), axis=0) * 100
        
        sentiment_time_pct.plot(ax=axes[1, 1], color=colors)
        axes[1, 1].set_title('Sentiment Trends Over Time', fontweight='bold')
        axes[1, 1].set_xlabel('Month-Year')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].legend(title='Sentiment')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/task2_sentiment_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_bank_comparison(df, bank_sentiment):
    """Create detailed bank comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏦 Bank-by-Bank Sentiment Comparison', fontsize=16, fontweight='bold')
    
    banks = df['bank'].unique()
    
    # 1. Average Sentiment Scores by Bank
    if bank_sentiment is not None and 'vader_compound' in bank_sentiment.columns:
        bank_sentiment['vader_compound'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Average VADER Sentiment Score by Bank', fontweight='bold')
        axes[0, 0].set_xlabel('Bank')
        axes[0, 0].set_ylabel('Average Sentiment Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 2. Sentiment Distribution Heatmap
    if 'ensemble_label' in df.columns:
        sentiment_matrix = pd.crosstab(df['bank'], df['ensemble_label'], normalize='index') * 100
        sns.heatmap(sentiment_matrix, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0, 1])
        axes[0, 1].set_title('Sentiment Distribution Heatmap (%)', fontweight='bold')
    
    # 3. Rating Distribution by Bank
    if 'rating' in df.columns:
        rating_dist = df.groupby(['bank', 'rating']).size().unstack(fill_value=0)
        rating_dist.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='viridis')
        axes[1, 0].set_title('Rating Distribution by Bank', fontweight='bold')
        axes[1, 0].set_xlabel('Bank')
        axes[1, 0].set_ylabel('Number of Reviews')
        axes[1, 0].legend(title='Rating (Stars)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Review Volume and Sentiment
    bank_stats = df.groupby('bank').agg({
        'ensemble_label': lambda x: (x == 'Positive').mean() * 100,
        'rating': 'count'
    }).round(1)
    
    # Create dual-axis plot
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    x_pos = range(len(bank_stats))
    bars1 = ax1.bar([x - 0.2 for x in x_pos], bank_stats['rating'], 0.4, 
                    label='Review Count', color='lightblue', alpha=0.7)
    bars2 = ax2.bar([x + 0.2 for x in x_pos], bank_stats['ensemble_label'], 0.4, 
                    label='% Positive', color='green', alpha=0.7)
    
    ax1.set_xlabel('Bank')
    ax1.set_ylabel('Review Count', color='blue')
    ax2.set_ylabel('% Positive Sentiment', color='green')
    ax1.set_title('Review Volume vs Positive Sentiment %', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([bank.split(' ')[-1] for bank in bank_stats.index], rotation=45)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/task2_bank_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_thematic_analysis_viz(bank_themes):
    """Create thematic analysis visualizations"""
    if not bank_themes:
        print("⚠️ No thematic analysis data available for visualization")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎪 Thematic Analysis - Key Topics and Themes', fontsize=16, fontweight='bold')
    
    # 1. Top Keywords Across All Banks
    all_keywords = {}
    for bank, data in bank_themes.items():
        for kw, score in data.get('tfidf_keywords', [])[:10]:
            all_keywords[kw] = all_keywords.get(kw, 0) + score
    
    if all_keywords:
        top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:15]
        keywords, scores = zip(*top_keywords)
        
        axes[0, 0].barh(range(len(keywords)), scores, color='lightcoral')
        axes[0, 0].set_yticks(range(len(keywords)))
        axes[0, 0].set_yticklabels(keywords)
        axes[0, 0].set_xlabel('TF-IDF Score')
        axes[0, 0].set_title('Top Keywords Across All Banks', fontweight='bold')
        axes[0, 0].invert_yaxis()
    
    # 2. Theme Distribution by Bank
    theme_data = []
    for bank, data in bank_themes.items():
        for theme, theme_info in data.get('themes', {}).items():
            theme_data.append({
                'bank': bank,
                'theme': theme,
                'matches': theme_info.get('matches', 0)
            })
    
    if theme_data:
        theme_df = pd.DataFrame(theme_data)
        theme_pivot = theme_df.pivot(index='bank', columns='theme', values='matches').fillna(0)
        
        sns.heatmap(theme_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 1])
        axes[0, 1].set_title('Theme Distribution by Bank (Keyword Matches)', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Most Common Themes Overall
    theme_totals = {}
    for bank, data in bank_themes.items():
        for theme, theme_info in data.get('themes', {}).items():
            theme_totals[theme] = theme_totals.get(theme, 0) + theme_info.get('matches', 0)
    
    if theme_totals:
        sorted_themes = sorted(theme_totals.items(), key=lambda x: x[1], reverse=True)
        themes, counts = zip(*sorted_themes)
        
        axes[1, 0].pie(counts, labels=themes, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Most Common Themes Across All Banks', fontweight='bold')
    
    # 4. Bank-Specific Primary Themes
    bank_primary_themes = {}
    for bank, data in bank_themes.items():
        themes = data.get('themes', {})
        if themes:
            primary_theme = max(themes.items(), key=lambda x: x[1].get('matches', 0))
            bank_primary_themes[bank.split(' ')[-1]] = primary_theme[1].get('matches', 0)
    
    if bank_primary_themes:
        banks = list(bank_primary_themes.keys())
        matches = list(bank_primary_themes.values())
        
        bars = axes[1, 1].bar(banks, matches, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('Primary Theme Strength by Bank', fontweight='bold')
        axes[1, 1].set_xlabel('Bank')
        axes[1, 1].set_ylabel('Keyword Matches')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/task2_thematic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_interactive_dashboard(df, bank_themes):
    """Create interactive Plotly dashboard"""
    print("🚀 Creating interactive dashboard...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Distribution by Bank', 'Rating vs Sentiment Correlation',
                       'Sentiment Trends Over Time', 'Theme Analysis'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Sentiment by Bank
    if 'ensemble_label' in df.columns:
        sentiment_bank = pd.crosstab(df['bank'], df['ensemble_label'], normalize='index') * 100
        
        for i, sentiment in enumerate(['Positive', 'Neutral', 'Negative']):
            if sentiment in sentiment_bank.columns:
                fig.add_trace(
                    go.Bar(
                        x=sentiment_bank.index,
                        y=sentiment_bank[sentiment],
                        name=sentiment,
                        marker_color=['green', 'orange', 'red'][i]
                    ),
                    row=1, col=1
                )
    
    # 2. Rating vs Sentiment
    if 'rating' in df.columns and 'vader_compound' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['rating'],
                y=df['vader_compound'],
                mode='markers',
                name='Reviews',
                marker=dict(color=df['rating'], colorscale='viridis', size=8, opacity=0.6),
                text=df['bank'],
                hovertemplate='<b>%{text}</b><br>Rating: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Sentiment Over Time
    if 'date' in df.columns and 'ensemble_label' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month_year'] = df['date'].dt.to_period('M').astype(str)
        
        sentiment_time = df.groupby(['month_year', 'ensemble_label']).size().unstack(fill_value=0)
        
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment in sentiment_time.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_time.index,
                        y=sentiment_time[sentiment],
                        mode='lines+markers',
                        name=f'{sentiment} (Time)',
                        line=dict(color={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'}[sentiment])
                    ),
                    row=2, col=1
                )
    
    # 4. Theme Analysis
    if bank_themes:
        theme_data = []
        for bank, data in bank_themes.items():
            for theme, theme_info in data.get('themes', {}).items():
                theme_data.append({
                    'bank': bank.split(' ')[-1],  # Shortened bank name
                    'theme': theme.replace(' & ', ' '),  # Shortened theme name
                    'matches': theme_info.get('matches', 0)
                })
        
        if theme_data:
            theme_df = pd.DataFrame(theme_data)
            theme_totals = theme_df.groupby('theme')['matches'].sum().sort_values(ascending=True)
            
            fig.add_trace(
                go.Bar(
                    x=theme_totals.values,
                    y=theme_totals.index,
                    orientation='h',
                    name='Theme Frequency',
                    marker_color='lightblue'
                ),
                row=2, col=2
            )
    
    # Update layout
    fig.update_layout(
        title_text="📊 Ethiopian Banking Apps Analysis - Interactive Dashboard",
        title_font_size=20,
        showlegend=True,
        height=800
    )
    
    # Save interactive dashboard
    fig.write_html('results/task2_interactive_dashboard.html')
    print("✅ Interactive dashboard saved to: results/task2_interactive_dashboard.html")
    
    return fig

def generate_summary_report(df, bank_sentiment, bank_themes, insights):
    """Generate a comprehensive summary report"""
    print("📝 Generating comprehensive summary report...")
    
    report = []
    report.append("# 📊 TASK 2 ANALYSIS SUMMARY REPORT")
    report.append("## Ethiopian Banking Apps - Sentiment and Thematic Analysis")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Overview Statistics
    report.append("\n## 🔍 OVERVIEW STATISTICS")
    report.append(f"- **Total Reviews Analyzed:** {len(df):,}")
    report.append(f"- **Banks Covered:** {df['bank'].nunique()}")
    report.append(f"- **Date Range:** {df['date'].min()} to {df['date'].max()}")
    report.append(f"- **Average Rating:** {df['rating'].mean():.2f} stars")
    
    # Sentiment Analysis Results
    if 'ensemble_label' in df.columns:
        report.append("\n## 🎭 SENTIMENT ANALYSIS RESULTS")
        sentiment_dist = df['ensemble_label'].value_counts(normalize=True) * 100
        
        for sentiment, percentage in sentiment_dist.items():
            emoji = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😞'}.get(sentiment, '⚪')
            report.append(f"- **{sentiment} {emoji}:** {percentage:.1f}%")
        
        # Sentiment by bank
        report.append("\n### 🏦 Sentiment by Bank:")
        bank_sentiment_summary = df.groupby('bank')['ensemble_label'].value_counts(normalize=True).unstack(fill_value=0) * 100
        
        for bank in bank_sentiment_summary.index:
            report.append(f"\n**{bank}:**")
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in bank_sentiment_summary.columns:
                    pct = bank_sentiment_summary.loc[bank, sentiment]
                    report.append(f"  - {sentiment}: {pct:.1f}%")
    
    # Rating-Sentiment Correlation
    if 'rating' in df.columns and 'vader_compound' in df.columns:
        correlation = df[['rating', 'vader_compound']].corr().iloc[0, 1]
        report.append(f"\n**🔗 Rating-Sentiment Correlation:** {correlation:.3f}")
    
    # Thematic Analysis Results
    if bank_themes:
        report.append("\n## 🎪 THEMATIC ANALYSIS RESULTS")
        
        # Overall themes
        all_themes = {}
        for bank, data in bank_themes.items():
            for theme, theme_info in data.get('themes', {}).items():
                all_themes[theme] = all_themes.get(theme, 0) + theme_info.get('matches', 0)
        
        sorted_themes = sorted(all_themes.items(), key=lambda x: x[1], reverse=True)
        
        report.append("\n### 🏆 Most Common Themes (Keyword Matches):")
        for i, (theme, matches) in enumerate(sorted_themes[:5], 1):
            report.append(f"{i}. **{theme}:** {matches} matches")
        
        # Bank-specific themes
        report.append("\n### 🏦 Primary Themes by Bank:")
        for bank, data in bank_themes.items():
            themes = data.get('themes', {})
            if themes:
                primary_theme = max(themes.items(), key=lambda x: x[1].get('matches', 0))
                report.append(f"- **{bank}:** {primary_theme[0]} ({primary_theme[1].get('matches', 0)} matches)")
                
                # Top keywords
                if data.get('tfidf_keywords'):
                    top_keywords = [kw[0] for kw in data['tfidf_keywords'][:5]]
                    report.append(f"  - Top keywords: {', '.join(top_keywords)}")
    
    # Key Insights
    if insights and 'key_findings' in insights:
        report.append("\n## 💡 KEY INSIGHTS")
        for finding in insights['key_findings']:
            report.append(f"- {finding}")
    
    # Recommendations
    if insights and 'recommendations' in insights:
        report.append("\n## 🎯 RECOMMENDATIONS")
        for i, rec in enumerate(insights['recommendations'], 1):
            report.append(f"{i}. {rec}")
    
    # Save report
    report_text = '\n'.join(report)
    
    with open('results/Task2_Analysis_Summary.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✅ Summary report saved to: results/Task2_Analysis_Summary.md")
    
    return report_text

def main():
    """Main function to create all visualizations and reports"""
    print("🎨 CREATING TASK 2 VISUALIZATIONS AND REPORTS")
    print("=" * 60)
    
    # Load data
    df, bank_sentiment, bank_themes, insights = load_latest_results()
    
    if df is None:
        print("❌ No data loaded. Please run Task 2 analysis first.")
        return
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Create visualizations
    print("\n🎨 Creating sentiment analysis overview...")
    sentiment_fig = create_sentiment_overview(df)
    
    print("\n🏦 Creating bank comparison visualizations...")
    bank_fig = create_bank_comparison(df, bank_sentiment)
    
    if bank_themes:
        print("\n🎪 Creating thematic analysis visualizations...")
        theme_fig = create_thematic_analysis_viz(bank_themes)
    
    print("\n🚀 Creating interactive dashboard...")
    dashboard = create_interactive_dashboard(df, bank_themes)
    
    print("\n📝 Generating comprehensive summary report...")
    report = generate_summary_report(df, bank_sentiment, bank_themes, insights)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ TASK 2 VISUALIZATIONS COMPLETE!")
    print("=" * 60)
    print(f"📊 Analyzed {len(df):,} reviews from {df['bank'].nunique()} banks")
    
    if 'ensemble_label' in df.columns:
        positive_pct = (df['ensemble_label'] == 'Positive').mean() * 100
        print(f"😊 Overall positive sentiment: {positive_pct:.1f}%")
    
    print(f"\n📁 Generated Files:")
    print(f"   🖼️  results/task2_sentiment_overview.png")
    print(f"   🏦 results/task2_bank_comparison.png")
    if bank_themes:
        print(f"   🎪 results/task2_thematic_analysis.png")
    print(f"   🌐 results/task2_interactive_dashboard.html")
    print(f"   📄 results/Task2_Analysis_Summary.md")
    
    print(f"\n🎯 Ready for presentation and decision-making!")

if __name__ == "__main__":
    main() 
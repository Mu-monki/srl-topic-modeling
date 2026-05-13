"""
============================================================
COMPLETE NMF TOPIC MODELING PIPELINE (FIXED)
============================================================
Start-to-finish process with ultra-minimal preprocessing
optimized for full-text documents that are already cleaned.
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = 'topic_modeling_with_fulltext.csv'
TEXT_COLUMN = 'Full_Text_Cleaned'
N_TOPICS = 4  # Based on optimization results
RANDOM_STATE = 42
OUTPUT_DIR = 'nmf_fulltext_results'

TOPIC_LABELS = {
    1: "Disaster Governance &\nVolunteer Dynamics",
    2: "Psychology &\nVolunteer Motivations",
    3: "Operations Research &\nVolunteer Optimization",
    4: "Philippine Disaster Studies &\nCommunity Resilience"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# STEP 1: ULTRA-MINIMAL TEXT CLEANING
# ============================================================

def safe_clean_text(text):
    """
    Very gentle cleaning that preserves nearly all content.
    The Full_Text_Cleaned is already preprocessed, so we only
    need to do very light additional cleaning.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs (essential)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove DOIs
    text = re.sub(r'doi\s*:\s*\S+', ' ', text)
    
    # Remove citation brackets [1], [2,3], [1-5]
    text = re.sub(r'\[\d+(?:[,-]\d+)*\]', ' ', text)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Remove punctuation and special characters (keep only letters)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove single characters
    text = re.sub(r'\b[a-z]\b', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords_and_lemmatize(text, stop_words_set):
    """Remove stop words and lemmatize"""
    if not text:
        return ""
    
    lemmatizer = WordNetLemmatizer()
    
    # Split (already clean, no need for word_tokenize)
    tokens = text.split()
    
    # Remove stop words and short tokens
    tokens = [t for t in tokens if t not in stop_words_set and len(t) > 2]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)


# ============================================================
# STEP 2: NMF TOPIC MODELING
# ============================================================

class NMFTopicModeler:
    """NMF topic modeling with full-text optimized settings"""
    
    def __init__(self, n_topics=4, random_state=42):
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.doc_topic_matrix = None
        
    def fit(self, documents):
        """Fit NMF with full-text optimized parameters"""
        print(f"\n📊 Vectorizing {len(documents)} documents...")
        
        # Calculate average document length
        avg_len = np.mean([len(doc.split()) for doc in documents])
        print(f"   Average document length: {avg_len:.0f} words")
        
        # Adjust max_features based on corpus size
        vocab_size = min(5000, int(len(documents) * 20))
        
        self.vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            min_df=3,                # Must appear in at least 3 docs
            max_df=0.95,             # Can appear in up to 95% of docs
            ngram_range=(1, 2),      # Unigrams and bigrams
            stop_words='english',    # Basic English stop words
            sublinear_tf=True,       # Log scaling
            strip_accents='unicode',
            lowercase=True
        )
        
        dtm = self.vectorizer.fit_transform(documents)
        print(f"   Document-term matrix: {dtm.shape}")
        print(f"   Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        print(f"   Matrix sparsity: {dtm.nnz / (dtm.shape[0] * dtm.shape[1]):.4f}")
        
        if dtm.nnz < 500:
            print("\n   ⚠ CRITICAL: Document-term matrix is nearly empty!")
            print("   The text column may not contain usable content.")
            return None
        
        print(f"\n🔧 Fitting NMF with {self.n_topics} topics...")
        
        self.model = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            alpha_W=0.0,           # No regularization (let data speak)
            alpha_H=0.0,
            l1_ratio=0.0,
            max_iter=400,          # Plenty of iterations
            solver='mu',           # Multiplicative update
            init='nndsvda',        # Better initialization
            tol=1e-4
        )
        
        self.doc_topic_matrix = self.model.fit_transform(dtm)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"   ✓ Model fitted successfully")
        
        return self.doc_topic_matrix
    
    def get_topic_words(self, n_words=20):
        """Get top words for each topic"""
        if self.model is None:
            return {}
            
        topics = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]
            top_weights = topic[top_indices]
            topics[f'Topic_{topic_idx+1}'] = {
                'words': top_words,
                'weights': top_weights
            }
        return topics
    
    def display_topics(self, n_words=20):
        """Display topic results"""
        if self.model is None:
            print("No model fitted yet!")
            return None
            
        topics = self.get_topic_words(n_words)
        
        print(f"\n{'='*70}")
        print(f"NMF TOPIC RESULTS ({self.n_topics} topics)")
        print(f"{'='*70}")
        
        for topic_name, data in topics.items():
            topic_num = int(topic_name.split('_')[1])
            label = TOPIC_LABELS.get(topic_num, topic_name)
            
            # Calculate topic size
            if self.doc_topic_matrix is not None:
                n_docs = (self.doc_topic_matrix.argmax(axis=1) == topic_num - 1).sum()
                pct = n_docs / self.doc_topic_matrix.shape[0] * 100
                print(f"\n{'─'*70}")
                print(f"Topic {topic_num}: {label}")
                print(f"Documents: {n_docs} ({pct:.1f}%)")
            else:
                print(f"\n{'─'*70}")
                print(f"Topic {topic_num}: {label}")
            
            print(f"\n{'Rank':<5} {'Term':<30} {'Weight':>10}  {'Importance'}")
            print(f"{'─'*60}")
            
            for i, (word, weight) in enumerate(zip(data['words'][:n_words], 
                                                   data['weights'][:n_words]), 1):
                bar = '█' * min(int(weight * 80), 40)
                print(f"{i:<5} {word:<30} {weight:>10.4f}  {bar}")
        
        return topics
    
    def diagnose(self):
        """Diagnose topic quality"""
        if self.doc_topic_matrix is None:
            print("No model fitted yet!")
            return None, None
            
        doc_weights = self.doc_topic_matrix
        
        print(f"\n{'='*70}")
        print(f"TOPIC DISTRIBUTION DIAGNOSIS")
        print(f"{'='*70}")
        
        avg_weights = doc_weights.mean(axis=0)
        for i, w in enumerate(avg_weights):
            pct = w * 100
            bar = '█' * int(pct * 2)
            label = TOPIC_LABELS.get(i+1, f"Topic {i+1}")
            print(f"  Topic {i+1}: {w:.4f} ({pct:5.1f}%) {bar}  {label.replace(chr(10), ' ')}")
        
        max_w = avg_weights.max()
        min_w = avg_weights.min()
        
        if max_w > 0.8:
            print(f"\n  ⚠ WARNING: One topic dominates strongly")
            print(f"    Consider reducing n_topics or checking preprocessing")
        elif max_w / (min_w + 0.001) > 20:
            print(f"\n  ⚠ NOTE: Topics are somewhat imbalanced")
            print(f"    This is normal for focused corpora")
        else:
            print(f"\n  ✓ Topic distribution is healthy")
        
        # Topic similarity
        similarity = cosine_similarity(self.model.components_)
        
        print(f"\n  Topic Similarity Matrix (lower = more distinct):")
        print(f"  {'':>10}", end="")
        for i in range(self.n_topics):
            print(f"{'T'+str(i+1):>10}", end="")
        print()
        
        for i in range(self.n_topics):
            print(f"  {'T'+str(i+1):>10}", end="")
            for j in range(self.n_topics):
                if i == j:
                    print(f"{'─':>10}", end="")
                else:
                    sim = similarity[i][j]
                    symbol = "✓" if sim < 0.5 else "⚠" if sim < 0.7 else "❌"
                    print(f"{sim:>8.4f}{symbol}", end="")
            print()
        
        # Average distinctiveness
        mask = ~np.eye(self.n_topics, dtype=bool)
        avg_sim = similarity[mask].mean()
        print(f"\n  Average inter-topic similarity: {avg_sim:.4f}")
        if avg_sim < 0.3:
            print(f"  ✓ Excellent - topics are very distinct")
        elif avg_sim < 0.5:
            print(f"  ✓ Good - topics are reasonably distinct")
        elif avg_sim < 0.7:
            print(f"  ⚠ Moderate - some overlap between topics")
        else:
            print(f"  ❌ High - topics share significant vocabulary")
        
        return avg_weights, similarity
    
    def get_document_topics_table(self, df, n=5):
        """Show top documents for each topic"""
        print(f"\n{'='*70}")
        print(f"TOP {n} DOCUMENTS PER TOPIC")
        print(f"{'='*70}")
        
        for topic_num in range(1, self.n_topics + 1):
            label = TOPIC_LABELS.get(topic_num, f"Topic {topic_num}")
            topic_mask = df['Dominant_Topic'] == topic_num
            topic_docs = df[topic_mask].nlargest(n, f'Topic_{topic_num}_Weight')
            
            print(f"\n{'─'*70}")
            print(f"Topic {topic_num}: {label}")
            print(f"Total documents: {topic_mask.sum()}")
            
            for i, (_, row) in enumerate(topic_docs.iterrows(), 1):
                title = str(row.get('Title', 'N/A'))[:100]
                year = row.get('Year', 'N/A')
                weight = row[f'Topic_{topic_num}_Weight']
                print(f"\n  {i}. [{year}] {title}")
                print(f"     Weight: {weight:.4f}")


# ============================================================
# STEP 3: VISUALIZATION
# ============================================================

class Visualizer:
    """Create visualizations for NMF results"""
    
    def __init__(self, modeler, df, output_dir):
        self.modeler = modeler
        self.df = df
        self.output_dir = output_dir
        self.topics = modeler.get_topic_words(30)
        self.n = modeler.n_topics
    
    def create_wordclouds(self):
        """Generate wordclouds for all topics"""
        n = self.n
        
        fig, axes = plt.subplots(1, n, figsize=(6*n, 7))
        if n == 1:
            axes = [axes]
        
        colors = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys']
        
        for topic_idx in range(n):
            topic_key = f'Topic_{topic_idx+1}'
            data = self.topics.get(topic_key)
            
            if not data or len(data['words']) == 0:
                axes[topic_idx].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[topic_idx].axis('off')
                continue
            
            word_freq = {}
            max_w = max(data['weights']) if len(data['weights']) > 0 else 1
            min_w = min(data['weights']) if len(data['weights']) > 0 else 0
            
            for word, weight in zip(data['words'][:50], data['weights'][:50]):
                if max_w > min_w:
                    scaled = 10 + (weight - min_w) / (max_w - min_w) * 90
                else:
                    scaled = 50
                word_freq[word] = scaled
            
            if word_freq:
                try:
                    wordcloud = WordCloud(
                        width=600, height=500,
                        background_color='white',
                        colormap=colors[topic_idx % len(colors)],
                        max_words=50,
                        min_font_size=8,
                        max_font_size=150,
                        relative_scaling=0.5,
                        random_state=42,
                        collocations=False
                    ).generate_from_frequencies(word_freq)
                    
                    axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
                except Exception as e:
                    axes[topic_idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            
            axes[topic_idx].axis('off')
            
            # Topic label and stats
            label = TOPIC_LABELS.get(topic_idx+1, f"Topic {topic_idx+1}")
            n_docs = len(self.df[self.df['Dominant_Topic'] == topic_idx+1])
            pct = n_docs / len(self.df) * 100 if len(self.df) > 0 else 0
            axes[topic_idx].set_title(
                f'Topic {topic_idx+1}: {label}\n{n_docs} documents ({pct:.1f}%)',
                fontsize=12, fontweight='bold', pad=15
            )
        
        plt.suptitle('NMF Topic Wordclouds', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'wordclouds.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"✓ Saved: wordclouds.png")
    
    def create_topic_barchart(self):
        """
        Comprehensive interactive bar chart visualization for topics.
        Includes multiple views and detailed information.
        """
        n = self.n
        
        # ============================================================
        # VIEW 1: Horizontal Bar Charts (Main View)
        # ============================================================
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Top Terms by Topic Weight</b>',
                '<b>Topic Size Comparison</b>',
                '<b>Term Distinctiveness vs Frequency</b>',
                '<b>Topic Summary Statistics</b>'
            ),
            specs=[
                [{'colspan': 2}, None],
                [{'type': 'scatter'}, {'type': 'table'}]
            ],
            row_heights=[0.55, 0.45],
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#C39BD3']
        lighter_colors = ['#FFB3B3', '#A8E6CF', '#8ED1E8', '#C4E6D0', '#FFF4C4', '#E8C8F0', '#D7BDE2']
        
        # ─── SUBPLOT 1: Top Terms Horizontal Bars ───────────────────
        
        bar_height = 0.8
        all_terms = []
        all_weights = []
        all_topics = []
        all_colors_bar = []
        
        for topic_idx in range(n):
            data = self.topics.get(f'Topic_{topic_idx+1}')
            if not data:
                continue
            
            # Get top 8 terms for clarity
            words = data['words'][:8]
            weights = data['weights'][:8]
            
            # Store for the term comparison plot
            all_terms.extend(words)
            all_weights.extend(weights)
            all_topics.extend([f'Topic {topic_idx+1}'] * len(words))
            all_colors_bar.extend([colors[topic_idx]] * len(words))
        
        # Create grouped horizontal bars
        y_positions = list(range(len(all_terms)))
        
        fig.add_trace(
            go.Bar(
                x=all_weights,
                y=all_terms,
                orientation='h',
                marker=dict(
                    color=all_colors_bar,
                    line=dict(color='white', width=0.5)
                ),
                text=[f'{w:.3f}' for w in all_weights],
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'Weight: %{x:.4f}<br>'
                    '<extra></extra>'
                ),
                showlegend=False,
                name='Terms'
            ),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="Weight", row=1, col=1, gridcolor='#F0F0F0')
        fig.update_yaxes(title_text="", row=1, col=1, automargin=True)
        
        # ─── SUBPLOT 2: Topic Size Comparison (Donut + Stats) ────────
        # This is now merged into the top row with the term bars
        # Instead, we'll add topic size info as annotations
        
        # ─── SUBPLOT 3: Term Distinctiveness vs Frequency ────────────
        
        for topic_idx in range(n):
            data = self.topics.get(f'Topic_{topic_idx+1}')
            if not data:
                continue
            
            words = data['words'][:15]
            weights = data['weights'][:15]
            
            # Calculate distinctiveness (normalized weight * uniqueness score)
            max_w = max(weights) if weights else 1
            min_w = min(weights) if weights else 0
            
            # Jitter for visibility
            x_jitter = np.random.normal(0, 0.02, len(words))
            
            fig.add_trace(
                go.Scatter(
                    x=(np.array(weights) + x_jitter),
                    y=list(range(len(words)))[::-1],
                    mode='markers+text',
                    name=f'Topic {topic_idx+1}',
                    marker=dict(
                        size=[max(10, min(40, w * 300)) for w in weights],
                        color=colors[topic_idx],
                        line=dict(color='white', width=1),
                        opacity=0.85
                    ),
                    text=words,
                    textposition='middle right',
                    textfont=dict(size=11, color=colors[topic_idx]),
                    hovertemplate=(
                        f'<b>Topic {topic_idx+1}</b><br>'
                        f'Term: %{{text}}<br>'
                        f'Weight: %{{x:.4f}}<br>'
                        f'<extra></extra>'
                    ),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(
            title_text="Term Weight (size = importance)", 
            row=2, col=1, 
            gridcolor='#F0F0F0',
            zeroline=True,
            zerolinecolor='#E0E0E0'
        )
        fig.update_yaxes(
            title_text="Term Rank",
            row=2, col=1,
            showticklabels=False,
            gridcolor='#F0F0F0'
        )
        
        # ─── SUBPLOT 4: Topic Summary Statistics Table ──────────────
        
        # Calculate statistics for each topic
        table_data = {
            'Topic': [],
            'Documents': [],
            '%': [],
            'Avg Weight': [],
            'Top 3 Terms': [],
            'Coherence': []
        }
        
        # Simple coherence: how well top terms relate
        for topic_idx in range(n):
            data = self.topics.get(f'Topic_{topic_idx+1}')
            if not data:
                continue
            
            label = TOPIC_LABELS.get(topic_idx+1, f"Topic {topic_idx+1}")
            n_docs = len(self.df[self.df['Dominant_Topic'] == topic_idx+1])
            pct = n_docs / len(self.df) * 100 if len(self.df) > 0 else 0
            avg_w = self.df[f'Topic_{topic_idx+1}_Weight'].mean() if f'Topic_{topic_idx+1}_Weight' in self.df.columns else 0
            top3 = ', '.join(data['words'][:3])
            
            # Simple coherence: ratio of top word to 5th word
            weights = data['weights'][:5]
            coherence = weights[0] / weights[-1] if len(weights) >= 5 and weights[-1] > 0 else 0
            
            table_data['Topic'].append(f"T{topic_idx+1}: {label.split(chr(10))[0]}")
            table_data['Documents'].append(str(n_docs))
            table_data['%'].append(f"{pct:.1f}%")
            table_data['Avg Weight'].append(f"{avg_w:.4f}")
            table_data['Top 3 Terms'].append(top3)
            table_data['Coherence'].append(f"{coherence:.2f}x")
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Topic</b>', '<b>Docs</b>', '<b>%</b>', 
                        '<b>Avg Weight</b>', '<b>Top 3 Terms</b>', '<b>Coherence</b>'],
                    fill_color='#2C3E50',
                    font=dict(color='white', size=12),
                    align=['left', 'center', 'center', 'center', 'left', 'center'],
                    height=35
                ),
                cells=dict(
                    values=[
                        table_data['Topic'],
                        table_data['Documents'],
                        table_data['%'],
                        table_data['Avg Weight'],
                        table_data['Top 3 Terms'],
                        table_data['Coherence']
                    ],
                    fill_color=[
                        ['#ECF0F1', '#F8F9FA'] * (len(table_data['Topic'])//2 + 1)
                    ],
                    font=dict(size=11),
                    align=['left', 'center', 'center', 'center', 'left', 'center'],
                    height=30
                )
            ),
            row=2, col=2
        )
        
        # ============================================================
        # LAYOUT
        # ============================================================
        
        fig.update_layout(
            title_text="<b>NMF Topic Modeling: Comprehensive Topic Analysis</b>",
            title_font_size=22,
            height=900,
            width=1400,
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                orientation='v',
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=1.01,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#E0E0E0',
                borderwidth=1
            ),
            margin=dict(t=100, b=40, l=80, r=200)
        )
        
        # Add annotations explaining the views
        annotations = [
            dict(
                x=0.01, y=1.02,
                xref='paper', yref='paper',
                text='<b>📊 Term Weights:</b> Top terms ranked by importance in each topic',
                showarrow=False,
                font=dict(size=11, color='#7F8C8D'),
                align='left'
            ),
            dict(
                x=0.01, y=0.48,
                xref='paper', yref='paper',
                text='<b>🎯 Term Map:</b> Bubble size = importance. Right-leaning = higher weight',
                showarrow=False,
                font=dict(size=11, color='#7F8C8D'),
                align='left'
            ),
            dict(
                x=0.52, y=0.48,
                xref='paper', yref='paper',
                text='<b>📋 Summary:</b> Coherence = ratio of top term to 5th term weight',
                showarrow=False,
                font=dict(size=11, color='#7F8C8D'),
                align='left'
            ),
        ]
        
        for ann in annotations:
            fig.add_annotation(ann)
        
        path = os.path.join(self.output_dir, 'topic_barchart.html')
        fig.write_html(path)
        print(f"✓ Saved: topic_barchart.html")
        return fig
   
    def create_distribution_pie(self):
        """
        Comprehensive topic distribution visualization with multiple perspectives.
        Includes donut chart, sunburst, treemap, and detailed statistics.
        """
        topic_counts = self.df['Dominant_Topic'].value_counts().sort_index()
        n = self.n
        
        # Calculate statistics for each topic
        topic_stats = {}
        for topic_num in range(1, n + 1):
            mask = self.df['Dominant_Topic'] == topic_num
            n_docs = mask.sum()
            pct = (n_docs / len(self.df)) * 100 if len(self.df) > 0 else 0
            
            # Average weight of documents in this topic
            avg_weight = self.df.loc[mask, f'Topic_{topic_num}_Weight'].mean() if mask.sum() > 0 else 0
            
            # Average year for this topic
            avg_year = self.df.loc[mask, 'Year'].mean() if 'Year' in self.df.columns and mask.sum() > 0 else 0
            
            # Top 3 terms
            data = self.topics.get(f'Topic_{topic_num}')
            top_terms = ', '.join(data['words'][:3]) if data else 'N/A'
            
            # Most cited paper in this topic
            if 'Cited by' in self.df.columns:
                max_cited = self.df.loc[mask, 'Cited by'].max() if mask.sum() > 0 else 0
            else:
                max_cited = 0
            
            # Weight spread (consistency of topic assignment)
            weights = self.df.loc[mask, f'Topic_{topic_num}_Weight']
            weight_std = weights.std() if len(weights) > 1 else 0
            
            topic_stats[topic_num] = {
                'n_docs': n_docs,
                'pct': pct,
                'avg_weight': avg_weight,
                'avg_year': avg_year,
                'top_terms': top_terms,
                'max_cited': max_cited,
                'weight_std': weight_std,
                'label': TOPIC_LABELS.get(topic_num, f'Topic {topic_num}')
            }
        
        # ============================================================
        # CREATE MULTI-PANEL FIGURE
        # ============================================================
        
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'table'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]
            ],
            subplot_titles=(
                '<b>Topic Distribution</b>',
                '<b>Document Count by Topic</b>',
                '<b>Topic Summary Statistics</b>',
                '<b>Average Topic Weight (higher = more focused)</b>',
                '<b>Topic Consistency (lower std = more consistent)</b>',
                '<b>Average Publication Year by Topic</b>'
            ),
            row_heights=[0.55, 0.45],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#C39BD3']
        
        # ─── PANEL 1: Donut Chart ───────────────────────────────────
        
        labels = [f'<b>Topic {i}</b><br>{topic_stats[i]["label"].split(chr(10))[0]}' 
                for i in topic_counts.index]
        values = topic_counts.values
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(
                    colors=colors[:len(topic_counts)],
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent+value',
                textfont=dict(size=11),
                textposition='outside',
                pull=[0.02] * len(topic_counts),
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Documents: %{value}<br>'
                    'Percentage: %{percent}<br>'
                    '<extra></extra>'
                ),
                rotation=90,
                sort=False,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Center annotation
        fig.add_annotation(
            text=f'<b>{len(self.df)}</b><br>total<br>documents',
            x=0.16, y=0.78,
            xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=16, color='#2C3E50'),
            align='center'
        )
        
        # ─── PANEL 2: Horizontal Bar Chart ─────────────────────────
        
        bar_labels = [f'Topic {i}' for i in topic_counts.index]
        bar_values = topic_counts.values
        bar_colors = [colors[i-1] for i in topic_counts.index]
        
        fig.add_trace(
            go.Bar(
                x=bar_values,
                y=bar_labels,
                orientation='h',
                marker=dict(
                    color=bar_colors,
                    line=dict(color='white', width=1.5)
                ),
                text=[f'{v} docs ({topic_stats[i]["pct"]:.1f}%)' for i, v in zip(topic_counts.index, bar_values)],
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'Documents: %{x}<br>'
                    '<extra></extra>'
                ),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Number of Documents", row=1, col=2, gridcolor='#F0F0F0')
        
        # ─── PANEL 3: Summary Statistics Table ──────────────────────
        
        table_header = ['<b>Topic</b>', '<b>Docs</b>', '<b>%</b>', 
                        '<b>Avg Weight</b>', '<b>Top 2 Terms</b>']
        table_cells = [[], [], [], [], []]
        
        for topic_num in topic_counts.index:
            stats = topic_stats[topic_num]
            short_label = stats['label'].split('\n')[0][:40]
            table_cells[0].append(f"T{topic_num}: {short_label}")
            table_cells[1].append(str(stats['n_docs']))
            table_cells[2].append(f"{stats['pct']:.1f}%")
            table_cells[3].append(f"{stats['avg_weight']:.3f}")
            
            # Top 2 terms with weights
            data = self.topics.get(f'Topic_{topic_num}')
            if data and len(data['words']) >= 2:
                terms_str = f"{data['words'][0]} ({data['weights'][0]:.3f}), {data['words'][1]} ({data['weights'][1]:.3f})"
            else:
                terms_str = 'N/A'
            table_cells[4].append(terms_str)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_header,
                    fill_color='#2C3E50',
                    font=dict(color='white', size=11),
                    align=['left', 'center', 'center', 'center', 'left'],
                    height=32
                ),
                cells=dict(
                    values=table_cells,
                    fill_color=[['#ECF0F1', '#F8F9FA'] * (len(topic_counts)//2 + 1)],
                    font=dict(size=10),
                    align=['left', 'center', 'center', 'center', 'left'],
                    height=28
                )
            ),
            row=1, col=3
        )
        
        # ─── PANEL 4: Average Topic Weight ─────────────────────────
        
        weight_values = [topic_stats[i]['avg_weight'] for i in topic_counts.index]
        
        fig.add_trace(
            go.Bar(
                x=[f'Topic {i}' for i in topic_counts.index],
                y=weight_values,
                marker=dict(
                    color=bar_colors,
                    line=dict(color='white', width=1.5)
                ),
                text=[f'{w:.3f}' for w in weight_values],
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Avg Weight: %{y:.4f}<br>'
                    '<extra></extra>'
                ),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Avg Weight", row=2, col=1, gridcolor='#F0F0F0')
        
        # ─── PANEL 5: Topic Consistency (Std Dev) ──────────────────
        
        std_values = [topic_stats[i]['weight_std'] for i in topic_counts.index]
        
        fig.add_trace(
            go.Bar(
                x=[f'Topic {i}' for i in topic_counts.index],
                y=std_values,
                marker=dict(
                    color=bar_colors,
                    line=dict(color='white', width=1.5)
                ),
                text=[f'{s:.3f}' for s in std_values],
                textposition='outside',
                textfont=dict(size=11),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Weight Std Dev: %{y:.4f}<br>'
                    '<i>Lower = more consistent topic assignment</i><br>'
                    '<extra></extra>'
                ),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text="Weight Std Dev", row=2, col=2, gridcolor='#F0F0F0')
        
        # ─── PANEL 6: Average Publication Year ─────────────────────
        
        if 'Year' in self.df.columns:
            year_values = [topic_stats[i]['avg_year'] for i in topic_counts.index]
            
            fig.add_trace(
                go.Bar(
                    x=[f'Topic {i}' for i in topic_counts.index],
                    y=year_values,
                    marker=dict(
                        color=bar_colors,
                        line=dict(color='white', width=1.5)
                    ),
                    text=[f'{y:.1f}' for y in year_values],
                    textposition='outside',
                    textfont=dict(size=11),
                    hovertemplate=(
                        '<b>%{x}</b><br>'
                        'Avg Year: %{y:.1f}<br>'
                        '<i>Higher = more recent research focus</i><br>'
                        '<extra></extra>'
                    ),
                    showlegend=False
                ),
                row=2, col=3
            )
            
            fig.update_yaxes(title_text="Avg Year", row=2, col=3, gridcolor='#F0F0F0')
        
        # ============================================================
        # LAYOUT
        # ============================================================
        
        fig.update_layout(
            title_text="<b>NMF Topic Distribution: Comprehensive Analysis</b>",
            title_font_size=22,
            height=900,
            width=1600,
            template='plotly_white',
            hovermode='closest',
            margin=dict(t=100, b=40, l=60, r=60)
        )
        
        # Add explanatory annotations
        annotations = [
            dict(
                x=0.16, y=-0.05,
                xref='paper', yref='paper',
                text='<i>Donut chart: Inner hole shows total documents</i>',
                showarrow=False,
                font=dict(size=9, color='#95A5A6')
            ),
            dict(
                x=0.50, y=-0.05,
                xref='paper', yref='paper',
                text='<i>Bar chart: Document distribution with percentages</i>',
                showarrow=False,
                font=dict(size=9, color='#95A5A6')
            ),
            dict(
                x=0.83, y=-0.05,
                xref='paper', yref='paper',
                text='<i>Table: Quick reference with top terms</i>',
                showarrow=False,
                font=dict(size=9, color='#95A5A6')
            ),
        ]
        
        for ann in annotations:
            fig.add_annotation(ann)
        
        path = os.path.join(self.output_dir, 'distribution.html')
        fig.write_html(path)
        print(f"✓ Saved: distribution.html")
        return fig

    def create_temporal(self):
        """Temporal evolution of topics with trend lines"""
        if 'Year' not in self.df.columns:
            print("  ⚠ No Year column - skipping temporal analysis")
            return None
        
        topic_cols = [f'Topic_{i}_Weight' for i in range(1, self.n + 1)]
        
        # Filter years with enough data
        year_counts = self.df['Year'].value_counts().sort_index()
        valid_years = year_counts[year_counts >= 3].index
        
        yearly = self.df[self.df['Year'].isin(valid_years)].groupby('Year')[topic_cols].mean()
        
        # Also calculate yearly counts for context
        yearly_counts = self.df[self.df['Year'].isin(valid_years)].groupby('Year').size()
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25],
            subplot_titles=('Topic Evolution Over Time', 'Documents per Year'),
            vertical_spacing=0.12
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # ============================================================
        # PANEL 1: Topic Trends with Trend Lines
        # ============================================================
        
        for i, col in enumerate(topic_cols):
            label = TOPIC_LABELS.get(i+1, f"Topic {i+1}").split('\n')[0]
            color = colors[i % len(colors)]
            
            # Actual data line
            fig.add_trace(
                go.Scatter(
                    x=yearly.index,
                    y=yearly[col],
                    mode='lines+markers',
                    name=f'Topic {i+1}: {label}',
                    line=dict(color=color, width=3, shape='spline', smoothing=0.3),
                    marker=dict(size=10, symbol='circle', 
                            line=dict(color='white', width=1.5)),
                    hovertemplate=(
                        f'<b>Topic {i+1}</b><br>'
                        f'Year: %{{x}}<br>'
                        f'Weight: %{{y:.4f}}<br>'
                        f'<extra></extra>'
                    ),
                    legendgroup=f'topic{i}',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Trend line (linear regression)
            if len(yearly) >= 3:
                x_numeric = yearly.index.values
                y_values = yearly[col].values
                
                # Calculate linear trend
                z = np.polyfit(x_numeric, y_values, 1)
                p = np.poly1d(z)
                trend_line = p(x_numeric)
                
                # Calculate R²
                y_mean = np.mean(y_values)
                ss_tot = np.sum((y_values - y_mean) ** 2)
                ss_res = np.sum((y_values - trend_line) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate trend direction and strength
                slope = z[0]
                if slope > 0.0005:
                    trend_label = '📈 Growing'
                elif slope < -0.0005:
                    trend_label = '📉 Declining'
                else:
                    trend_label = '➡️ Stable'
                
                fig.add_trace(
                    go.Scatter(
                        x=yearly.index,
                        y=trend_line,
                        mode='lines',
                        name=f'T{i+1} trend (R²={r_squared:.2f})',
                        line=dict(color=color, width=2, dash='dash'),
                        opacity=0.5,
                        hovertemplate=(
                            f'<b>Topic {i+1} Trend</b><br>'
                            f'Year: %{{x}}<br>'
                            f'Predicted: %{{y:.4f}}<br>'
                            f'{trend_label}<br>'
                            f'<extra></extra>'
                        ),
                        legendgroup=f'topic{i}',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add annotation for trend
                mid_year = x_numeric[len(x_numeric)//2]
                mid_trend = trend_line[len(trend_line)//2]
                
                fig.add_annotation(
                    x=yearly.index[-1],
                    y=trend_line[-1],
                    text=f'{trend_label}',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=color,
                    font=dict(size=9, color=color),
                    ax=30,
                    ay=0,
                    row=1, col=1
                )
        
        # ============================================================
        # PANEL 2: Document Count per Year
        # ============================================================
        
        fig.add_trace(
            go.Bar(
                x=yearly_counts.index,
                y=yearly_counts.values,
                name='Documents',
                marker=dict(
                    color='#95A5A6',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>%{x}</b><br>Documents: %{y}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # ============================================================
        # LAYOUT
        # ============================================================
        
        fig.update_xaxes(title_text="Year", row=1, col=1, gridcolor='#E5E5E5')
        fig.update_xaxes(title_text="Year", row=2, col=1, gridcolor='#E5E5E5')
        
        fig.update_yaxes(title_text="Average Topic Weight", row=1, col=1, gridcolor='#E5E5E5')
        fig.update_yaxes(title_text="Number of Documents", row=2, col=1, gridcolor='#E5E5E5')
        
        fig.update_layout(
            title_text="Topic Evolution Over Time with Trend Analysis",
            title_font_size=20,
            height=750,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(t=80, b=40, l=60, r=40)
        )
        
        # Add summary annotation
        summary_text = "Dashed lines = linear trends<br>R² indicates trend fit (1.0 = perfect)"
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=summary_text,
            showarrow=False,
            font=dict(size=10, color='#7F8C8D'),
            align='left',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E5E5E5',
            borderwidth=1
        )
        
        path = os.path.join(self.output_dir, 'temporal.html')
        fig.write_html(path)
        print(f"✓ Saved: temporal.html")
        return fig
    
    def create_document_explorer(self):
        """Interactive document explorer table"""
        display_cols = ['Title', 'Year', 'Authors', 'Dominant_Topic', 'Dominant_Topic_Weight']
        available_cols = [c for c in display_cols if c in self.df.columns]
        
        display_df = self.df[available_cols].copy()
        display_df = display_df.sort_values(['Dominant_Topic', 'Dominant_Topic_Weight'],
                                           ascending=[True, False])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Title</b>', '<b>Year</b>', '<b>Authors</b>', 
                       '<b>Topic</b>', '<b>Weight</b>'],
                fill_color='#2C3E50',
                font=dict(color='white', size=13),
                align=['left', 'center', 'left', 'center', 'center'],
                height=35
            ),
            cells=dict(
                values=[
                    display_df['Title'].str[:120] + '...' if 'Title' in display_df else [],
                    display_df['Year'] if 'Year' in display_df else [],
                    display_df['Authors'].str[:60] if 'Authors' in display_df else [],
                    display_df['Dominant_Topic'],
                    display_df['Dominant_Topic_Weight'].round(4)
                ],
                fill_color=[['#ECF0F1', '#F8F9FA'] * (len(display_df)//2 + 1)],
                font=dict(size=11),
                align=['left', 'center', 'left', 'center', 'center'],
                height=30
            )
        )])
        
        fig.update_layout(
            title_text="📋 Document-Topic Assignment Explorer",
            title_font_size=20,
            height=800,
            template='plotly_white'
        )
        
        path = os.path.join(self.output_dir, 'documents.html')
        fig.write_html(path)
        print(f"✓ Saved: documents.html")
        return fig
    
    def create_all(self):
        """Create all visualizations"""
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        try:
            self.create_wordclouds()
        except Exception as e:
            print(f"  ⚠ Wordclouds failed: {e}")
        
        try:
            self.create_topic_barchart()
        except Exception as e:
            print(f"  ⚠ Barchart failed: {e}")
        
        try:
            self.create_distribution_pie()
        except Exception as e:
            print(f"  ⚠ Distribution failed: {e}")
        
        try:
            self.create_temporal()
        except Exception as e:
            print(f"  ⚠ Temporal failed: {e}")
        
        try:
            self.create_document_explorer()
        except Exception as e:
            print(f"  ⚠ Document explorer failed: {e}")
        
        print(f"\n✓ Visualizations saved to: {self.output_dir}/")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("NMF TOPIC MODELING - FULL TEXT PIPELINE")
    print("="*70)
    
    # ─── STEP 1: Load Data ───────────────────────────
    print(f"\n📂 Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   {len(df)} documents loaded")
    print(f"   Columns: {list(df.columns)}")
    
    # ─── STEP 2: Check Text Column ───────────────────
    print(f"\n📝 Checking text column: '{TEXT_COLUMN}'")
    
    if TEXT_COLUMN not in df.columns:
        print(f"\n   ❌ Column '{TEXT_COLUMN}' not found!")
        print(f"   Available text columns:")
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = str(df[col].iloc[0])[:80] if len(df) > 0 else 'N/A'
                print(f"     - {col}: {sample}...")
        return None
    
    # Fill NaN
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
    
    # Check content
    non_empty = (df[TEXT_COLUMN].str.strip().str.len() > 0).sum()
    print(f"   Non-empty documents: {non_empty}/{len(df)}")
    
    if non_empty == 0:
        print(f"\n   ❌ All documents are empty!")
        return None
    
    # Show sample
    sample = df[TEXT_COLUMN].iloc[0]
    print(f"   Sample length: {len(str(sample))} chars")
    print(f"   First 150 chars: {str(sample)[:150]}...")
    
    # ─── STEP 3: Clean Text ──────────────────────────
    print(f"\n🧹 Cleaning text...")
    
    # Step 3a: Basic cleaning (remove URLs, numbers, punctuation)
    df['cleaned_text'] = df[TEXT_COLUMN].apply(safe_clean_text)
    
    # Step 3b: Remove stop words and lemmatize
    stop_set = set(stopwords.words('english'))
    # Add only absolutely necessary extra stops
    extra_stops = {'copyright', 'elsevier', 'springer', 'wiley', 'doi', 'http', 'https', 'www',
            'copyright', 'elsevier', 'springer', 'wiley', 'taylor', 'francis',
            'doi', 'http', 'https', 'www', 'et', 'al', 'downloaded', 'university', 'online library',
            'et', 'al', 'also', 'using', 'based', 'paper', 'research',
            'study', 'results', 'show', 'approach', 'method', 'data',
            'analysis', 'used', 'present', 'proposed', 'model', 'system',
            'new', 'two', 'one', 'different', 'well', 'however', 'may',
            'first', 'second', 'order', 'findings', 'found', 'use',
            'within', 'among', 'provide', 'important', 'significant',
            'due', 'thus', 'therefore', 'can', 'analysis', 'studies',
            'several', 'presented', 'discussed', 'conclusion', 'abstract',
            'introduction', 'related', 'work', 'problem', 'elsevier',
            'ltd', 'rights', 'reserved', 'springer', 'wiley', 'ieee',
            'copyright', 'license', 'cc', 'creative', 'commons',  'cid', 'online library', 
            'article', 'use article', 'library term', 'engineering online'
            'rst', 'hurricane', 'signi', 'orts', 'ected', 'university of the philippines', 'library rule', 'diliman', 'diliman college'
        }
    stop_set = stop_set.union(extra_stops)
    
    df['processed_text'] = df['cleaned_text'].apply(
        lambda x: remove_stopwords_and_lemmatize(x, stop_set)
    )
    
    # ─── STEP 4: Filter Valid Documents ──────────────
    valid_mask = df['processed_text'].str.len() >= 50
    df = df[valid_mask].reset_index(drop=True)
    documents = df['processed_text'].tolist()
    
    print(f"   Valid documents (≥50 chars): {len(documents)}")
    
    if len(documents) < 10:
        print(f"\n   ❌ Too few valid documents ({len(documents)})!")
        print(f"   Checking text lengths:")
        df['text_len'] = df['processed_text'].str.len()
        print(df['text_len'].describe())
        
        # Try even lower threshold
        valid_mask = df['processed_text'].str.len() >= 20
        df = df[valid_mask].reset_index(drop=True)
        documents = df['processed_text'].tolist()
        print(f"   Valid documents (≥20 chars): {len(documents)}")
        
        if len(documents) < 10:
            return None
    
    print(f"   Average length: {np.mean([len(d.split()) for d in documents]):.0f} words")
    
    # ─── STEP 5: Fit NMF Model ───────────────────────
    print(f"\n🔧 Fitting NMF model with {N_TOPICS} topics...")
    
    modeler = NMFTopicModeler(n_topics=N_TOPICS, random_state=RANDOM_STATE)
    doc_topic_matrix = modeler.fit(documents)
    
    if doc_topic_matrix is None:
        print("\n❌ Model fitting failed!")
        return None
    
    # ─── STEP 6: Display Results ─────────────────────
    topics = modeler.display_topics(20)
    avg_weights, similarity = modeler.diagnose()
    
    # ─── STEP 7: Assign Topics to DataFrame ──────────
    for i in range(N_TOPICS):
        df[f'Topic_{i+1}_Weight'] = doc_topic_matrix[:, i]
    
    df['Dominant_Topic'] = doc_topic_matrix.argmax(axis=1) + 1
    df['Dominant_Topic_Weight'] = doc_topic_matrix.max(axis=1)
    
    # Add topic keywords
    if topics:
        topic_keywords = {}
        for topic_idx in range(N_TOPICS):
            words = topics.get(f'Topic_{topic_idx+1}', {}).get('words', [])[:5]
            topic_keywords[topic_idx + 1] = ', '.join(words) if words else 'N/A'
        df['Topic_Keywords'] = df['Dominant_Topic'].map(topic_keywords)
    
    # ─── STEP 8: Show Top Documents ──────────────────
    modeler.get_document_topics_table(df, n=3)
    
    # ─── STEP 9: Save Results ────────────────────────
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save full results
    results_path = os.path.join(OUTPUT_DIR, 'nmf_fulltext_results.csv')
    df.to_csv(results_path, index=False)
    print(f"   ✓ {results_path}")
    
    # Save model and vectorizer
    model_path = os.path.join(OUTPUT_DIR, 'nmf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(modeler.model, f)
    print(f"   ✓ {model_path}")
    
    vec_path = os.path.join(OUTPUT_DIR, 'vectorizer.pkl')
    with open(vectorizer_path := vec_path, 'wb') as f:
        pickle.dump(modeler.vectorizer, f)
    print(f"   ✓ {vec_path}")
    
    # ─── STEP 10: Create Visualizations ──────────────
    viz = Visualizer(modeler, df, OUTPUT_DIR)
    viz.create_all()
    
    # ─── STEP 11: Summary ────────────────────────────
    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 All results saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"\n📊 Key files:")
    print(f"   • nmf_fulltext_results.csv  - Full data with topic assignments")
    print(f"   • nmf_model.pkl             - Trained NMF model")
    print(f"   • vectorizer.pkl            - TF-IDF vectorizer")
    print(f"   • wordclouds.png            - Topic wordclouds")
    print(f"   • topic_barchart.html       - Interactive term charts")
    print(f"   • distribution.html         - Topic proportions")
    print(f"   • documents.html            - Document explorer")
    if 'Year' in df.columns:
        print(f"   • temporal.html             - Topic trends over time")
    
    return df, modeler, viz


if __name__ == "__main__":
    result = main()
    if result is not None:
        df, modeler, viz = result
    else:
        print("\n❌ Pipeline failed. Please check the errors above.")
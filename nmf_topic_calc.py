"""
============================================================
OPTIMAL TOPIC NUMBER FINDER FOR NMF
============================================================
Tests different topic numbers and recommends the best one
based on multiple evaluation metrics.
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV

# Download NLTK
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
OUTPUT_DIR = 'nmf_optimization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Topic range to test
MIN_TOPICS = 2
MAX_TOPICS = 10


# ============================================================
# LIGHTWEIGHT PREPROCESSING (same as before)
# ============================================================

class LightweightPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.extra_stops = {
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
            'rst', 'hurricane', 'signi', 'orts', 'ected'
        }
        self.all_stops = self.stop_words.union(self.extra_stops)
    
    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'doi\s*:\s*\S+', '', text)
        text = re.sub(r'\[\d+(?:[,-]\d+)*\]', '', text)
        
        boilerplate = [
            r'downloaded\s+(?:from|by)\s+[\w\s]+(?:library|university|college)[^.]*\.',
            r'(?:online|digital)\s+library\s+of\s+[\w\s]+[^.]*\.',
            r'this\s+article\s+is\s+(?:governed|protected)\s+by[^.]*\.',
            r'(?:terms|conditions)\s+of\s+(?:use|access|service)[^.]*\.',
            r'all\s+rights\s+reserved[^.]*\.',
            r'copyright\s+©\s*\d{4}[^.]*\.',
            r'creative\s+commons[^.]*\.',
            r'(?:university|college)\s+of\s+[\w\s]+(?:library|press)[^.]*\.',
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        tokens = [t for t in tokens if t not in self.all_stops and len(t) > 2]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)


# ============================================================
# EVALUATION METRICS
# ============================================================

def topic_coherence(model, feature_names, documents, top_n=10):
    """
    Calculate topic coherence based on word co-occurrence.
    Higher = better (more coherent topics).
    """
    doc_word_sets = [set(doc.split()) for doc in documents]
    
    coherence_scores = []
    
    for topic in model.components_:
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        
        pair_scores = []
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                w1, w2 = top_words[i], top_words[j]
                
                # Count co-occurrence
                cooccur = sum(1 for doc_set in doc_word_sets 
                             if w1 in doc_set and w2 in doc_set)
                w1_count = sum(1 for doc_set in doc_word_sets if w1 in doc_set)
                
                if w1_count > 0:
                    pair_scores.append(cooccur / w1_count)
        
        if pair_scores:
            coherence_scores.append(np.mean(pair_scores))
    
    return np.mean(coherence_scores) if coherence_scores else 0


def topic_diversity(model, feature_names, top_n=15):
    """
    Calculate topic diversity - how unique are the top words across topics?
    Higher = better (more diverse topics).
    """
    top_words_per_topic = []
    
    for topic in model.components_:
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = set([feature_names[i] for i in top_indices])
        top_words_per_topic.append(top_words)
    
    # Calculate average Jaccard distance (1 - similarity)
    n_topics = len(top_words_per_topic)
    if n_topics < 2:
        return 1.0
    
    distances = []
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            intersection = len(top_words_per_topic[i] & top_words_per_topic[j])
            union = len(top_words_per_topic[i] | top_words_per_topic[j])
            jaccard_sim = intersection / union if union > 0 else 0
            distances.append(1 - jaccard_sim)
    
    return np.mean(distances)


def reconstruction_error(model, dtm):
    """
    Calculate reconstruction error.
    Lower = better fit (but can overfit with too many topics).
    """
    W = model.transform(dtm)  # Document-topic matrix
    H = model.components_     # Topic-term matrix
    reconstructed = W @ H
    error = np.linalg.norm(dtm.toarray() - reconstructed, 'fro')
    return error


def topic_separation(model):
    """
    Calculate average cosine distance between topics.
    Higher = better (topics are more distinct).
    """
    similarity = cosine_similarity(model.components_)
    n = similarity.shape[0]
    
    # Get off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    off_diag = similarity[mask]
    
    return 1 - np.mean(off_diag)  # Convert to distance


def topic_balance(doc_topic_matrix):
    """
    Calculate entropy-based balance of topic distribution.
    Higher = better (more evenly distributed).
    """
    avg_weights = doc_topic_matrix.mean(axis=0)
    avg_weights = avg_weights / avg_weights.sum()  # Normalize
    
    # Shannon entropy (normalized)
    n = len(avg_weights)
    entropy = -np.sum(avg_weights * np.log(avg_weights + 1e-10))
    max_entropy = np.log(n)
    
    return entropy / max_entropy


# ============================================================
# COMBINED SCORE
# ============================================================

def combined_score(coherence, diversity, separation, balance, 
                   w_coherence=0.35, w_diversity=0.30, 
                   w_separation=0.20, w_balance=0.15):
    """
    Weighted combined score. Higher = better overall model.
    """
    return (w_coherence * coherence + 
            w_diversity * diversity + 
            w_separation * separation + 
            w_balance * balance)


# ============================================================
# GRID SEARCH
# ============================================================

def find_optimal_topics(documents, min_topics=2, max_topics=10):
    """
    Test different topic numbers and find the best one.
    """
    print("="*60)
    print("FINDING OPTIMAL NUMBER OF TOPICS")
    print("="*60)
    print(f"Testing k = {min_topics} to {max_topics}...")
    print()
    
    results = []
    
    for k in range(min_topics, max_topics + 1):
        print(f"\n{'─'*40}")
        print(f"Testing k = {k} topics")
        print(f"{'─'*40}")
        
        # Fit NMF
        vectorizer = TfidfVectorizer(
            max_features=5000, min_df=3, max_df=0.95,
            ngram_range=(1, 2), stop_words='english',
            sublinear_tf=True
        )
        
        dtm = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        model = NMF(
            n_components=k, random_state=42,
            alpha_W=0.0, alpha_H=0.0, max_iter=400,
            solver='mu', init='nndsvda'
        )
        
        doc_topic_matrix = model.fit_transform(dtm)
        
        # Calculate metrics
        coherence = topic_coherence(model, feature_names, documents)
        diversity = topic_diversity(model, feature_names)
        separation = topic_separation(model)
        balance = topic_balance(doc_topic_matrix)
        combined = combined_score(coherence, diversity, separation, balance)
        error = reconstruction_error(model, dtm)
        
        print(f"  Coherence:   {coherence:.4f}  (↑ better)")
        print(f"  Diversity:   {diversity:.4f}  (↑ better)")
        print(f"  Separation:  {separation:.4f}  (↑ better)")
        print(f"  Balance:     {balance:.4f}  (↑ better)")
        print(f"  ⭐ Combined:   {combined:.4f}  (↑ better)")
        print(f"  Recon Error: {error:.2f}    (↓ better, but too low = overfit)")
        
        # Show top words
        print(f"\n  Top words per topic:")
        for i, topic in enumerate(model.components_):
            top_words = [feature_names[j] for j in topic.argsort()[-5:][::-1]]
            print(f"    T{i+1}: {', '.join(top_words)}")
        
        results.append({
            'k': k,
            'coherence': coherence,
            'diversity': diversity,
            'separation': separation,
            'balance': balance,
            'combined': combined,
            'reconstruction_error': error,
            'model': model,
            'vectorizer': vectorizer,
            'doc_topic_matrix': doc_topic_matrix
        })
    
    return results


# ============================================================
# VISUALIZE RESULTS
# ============================================================

def plot_optimization_results(results):
    """Plot all metrics to find the best k"""
    ks = [r['k'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = [
        ('coherence', 'Topic Coherence', '↑ Higher is better', axes[0, 0]),
        ('diversity', 'Topic Diversity', '↑ Higher is better', axes[0, 1]),
        ('separation', 'Topic Separation', '↑ Higher is better', axes[0, 2]),
        ('balance', 'Topic Balance', '↑ Higher is better', axes[1, 0]),
        ('combined', '⭐ Combined Score', '↑ Higher is better', axes[1, 1]),
        ('reconstruction_error', 'Reconstruction Error', '↓ Lower is better', axes[1, 2]),
    ]
    
    for metric, title, ylabel, ax in metrics:
        values = [r[metric] for r in results]
        ax.plot(ks, values, 'o-', linewidth=2, markersize=10, color='#2C3E50')
        
        # Highlight best k
        if 'error' in metric:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        ax.plot(ks[best_idx], values[best_idx], 'o', 
               markersize=15, color='#FF6B6B', markeredgewidth=2,
               markeredgecolor='darkred')
        ax.axvline(x=ks[best_idx], color='red', linestyle='--', alpha=0.3)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of Topics', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
    
    plt.suptitle('NMF Topic Number Optimization', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'topic_optimization.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Find and print best k
    best_k = max(results, key=lambda r: r['combined'])
    
    print(f"\n{'='*60}")
    print(f"⭐ OPTIMAL NUMBER OF TOPICS: k = {best_k['k']}")
    print(f"{'='*60}")
    print(f"Combined score: {best_k['combined']:.4f}")
    print(f"Coherence:      {best_k['coherence']:.4f}")
    print(f"Diversity:      {best_k['diversity']:.4f}")
    print(f"Separation:     {best_k['separation']:.4f}")
    print(f"Balance:        {best_k['balance']:.4f}")
    
    return best_k


# ============================================================
# ELBOW METHOD (Alternative)
# ============================================================

def plot_elbow(results):
    """Plot reconstruction error elbow curve"""
    ks = [r['k'] for r in results]
    errors = [r['reconstruction_error'] for r in results]
    
    # Calculate rate of change
    deltas = np.diff(errors)
    delta_ks = ks[1:]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    axes[0].plot(ks, errors, 'o-', linewidth=2, markersize=10, color='#2C3E50')
    axes[0].set_title('Reconstruction Error (Elbow Method)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Topics')
    axes[0].set_ylabel('Reconstruction Error')
    axes[0].set_xticks(ks)
    axes[0].grid(alpha=0.3)
    
    # Find elbow (point of diminishing returns)
    # Use the point where the rate of change slows significantly
    if len(deltas) > 1:
        delta_changes = np.diff(deltas)
        elbow_idx = np.argmax(delta_changes) + 1  # +1 for diff offset
        elbow_k = ks[elbow_idx]
        axes[0].axvline(x=elbow_k, color='red', linestyle='--', alpha=0.5,
                       label=f'Elbow at k={elbow_k}')
        axes[0].legend()
    
    # Rate of change
    axes[1].bar(delta_ks, np.abs(deltas), color='#4ECDC4', edgecolor='white')
    axes[1].set_title('Rate of Change in Error', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Topics')
    axes[1].set_ylabel('|Δ Error|')
    axes[1].set_xticks(delta_ks)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Elbow Method Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_method.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("NMF TOPIC NUMBER OPTIMIZATION")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   {len(df)} documents")
    
    # Preprocess
    print(f"\n🧹 Preprocessing...")
    preprocessor = LightweightPreprocessor()
    df['processed_text'] = df[TEXT_COLUMN].fillna('').apply(preprocessor.clean_text)
    
    valid = df['processed_text'].str.len() >= 50
    df = df[valid].reset_index(drop=True)
    documents = df['processed_text'].tolist()
    print(f"   {len(documents)} valid documents")
    
    # Run optimization
    results = find_optimal_topics(documents, MIN_TOPICS, MAX_TOPICS)
    
    # Plot results
    best_k = plot_optimization_results(results)
    plot_elbow(results)
    
    # Recommendations based on corpus size
    n_docs = len(documents)
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if n_docs < 100:
        recommended_range = "2-3"
    elif n_docs < 300:
        recommended_range = "3-5"
    else:
        recommended_range = "4-6"
    
    print(f"Corpus size: {n_docs} documents")
    print(f"General recommendation for your corpus size: {recommended_range} topics")
    print(f"Optimal based on metrics: k = {best_k['k']}")
    print(f"\nConsider these trade-offs:")
    print(f"  More topics = finer granularity but may fragment small themes")
    print(f"  Fewer topics = broader themes but may miss nuance")
    
    # Save results
    results_df = pd.DataFrame([{
        'k': r['k'],
        'coherence': r['coherence'],
        'diversity': r['diversity'],
        'separation': r['separation'],
        'balance': r['balance'],
        'combined_score': r['combined'],
        'reconstruction_error': r['reconstruction_error']
    } for r in results])
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'optimization_results.csv'), index=False)
    print(f"\n✓ Results saved to: {OUTPUT_DIR}/")
    
    return results, best_k


if __name__ == "__main__":
    results, best_k = main()
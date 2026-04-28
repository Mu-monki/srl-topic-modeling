"""
Optimal Topic Number Analysis
Tests different topic counts and evaluates coherence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import warnings
warnings.filterwarnings('ignore')

class TopicOptimizer:
    """
    Finds optimal number of topics using multiple metrics
    """
    
    def __init__(self, texts, vectorizer_type='tfidf', max_features=5000):
        """
        Parameters:
        -----------
        texts : list
            Preprocessed text documents
        vectorizer_type : str
            'count' or 'tfidf'
        max_features : int
            Maximum features for vectorizer
        """
        self.texts = texts
        self.max_features = max_features
        
        # Create document-term matrix
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=5,
                max_df=0.95,
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=5,
                max_df=0.95,
                stop_words='english'
            )
        
        self.dtm = self.vectorizer.fit_transform(texts)
        print(f"Document-term matrix shape: {self.dtm.shape}")
        
        # For Gensim coherence
        self.tokenized_texts = [text.split() for text in texts]
        self.dictionary = Dictionary(self.tokenized_texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.95)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenized_texts]
    
    def compute_coherence_lda(self, n_topics, random_state=42):
        """
        Compute coherence score for LDA with given number of topics
        
        Returns:
        --------
        dict : {'n_topics': int, 'coherence': float, 'perplexity': float}
        """
        model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(self.dtm)
        
        # Extract top words for each topic
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_features = topic.argsort()[:-11:-1]  # Top 10 words
            top_words = [feature_names[i] for i in top_features]
            topics.append(top_words)
        
        # Calculate coherence using Gensim
        coherence_model = CoherenceModel(
            topics=topics,
            texts=self.tokenized_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        
        # Calculate perplexity
        perplexity = model.perplexity(self.dtm)
        
        return {
            'n_topics': n_topics,
            'coherence': coherence,
            'perplexity': perplexity
        }
    
    def compute_coherence_nmf(self, n_topics, random_state=42):
        """
        Compute coherence score for NMF with given number of topics
        """
        model = NMF(
            n_components=n_topics,
            random_state=random_state,
            max_iter=200
        )
        model.fit(self.dtm)
        
        # Extract top words
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_features = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_features]
            topics.append(top_words)
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topics,
            texts=self.tokenized_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        
        return {
            'n_topics': n_topics,
            'coherence': coherence,
            'reconstruction_error': model.reconstruction_err_
        }
    
    def grid_search_topics(self, topic_range, model_type='lda', random_state=42):
        """
        Test multiple topic numbers
        
        Parameters:
        -----------
        topic_range : list
            List of topic numbers to test, e.g., range(3, 31)
        model_type : str
            'lda' or 'nmf'
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"TESTING {len(topic_range)} TOPIC COUNTS ({model_type.upper()})")
        print(f"{'='*60}\n")
        
        for n_topics in topic_range:
            print(f"Testing {n_topics} topics...", end=" ")
            
            if model_type == 'lda':
                result = self.compute_coherence_lda(n_topics, random_state)
            else:
                result = self.compute_coherence_nmf(n_topics, random_state)
            
            results.append(result)
            print(f"Coherence: {result.get('coherence', 'N/A'):.4f}")
        
        return pd.DataFrame(results)
    
    def plot_optimization_results(self, results_df, model_type='lda'):
        """
        Plot coherence and other metrics vs number of topics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Coherence Score
        axes[0, 0].plot(results_df['n_topics'], results_df['coherence'], 
                       marker='o', linewidth=2, markersize=8, color='steelblue')
        axes[0, 0].set_xlabel('Number of Topics', fontsize=12)
        axes[0, 0].set_ylabel('Coherence Score (c_v)', fontsize=12)
        axes[0, 0].set_title(f'{model_type.upper()} Coherence by Topic Count', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark best point
        best_idx = results_df['coherence'].idxmax()
        best_n = results_df.loc[best_idx, 'n_topics']
        best_c = results_df.loc[best_idx, 'coherence']
        axes[0, 0].plot(best_n, best_c, 'r*', markersize=20, 
                       label=f'Best: {best_n} topics (coherence={best_c:.4f})')
        axes[0, 0].legend()
        
        # 2. Perplexity (for LDA)
        if 'perplexity' in results_df.columns:
            axes[0, 1].plot(results_df['n_topics'], results_df['perplexity'], 
                           marker='s', linewidth=2, markersize=8, color='coral')
            axes[0, 1].set_xlabel('Number of Topics', fontsize=12)
            axes[0, 1].set_ylabel('Perplexity', fontsize=12)
            axes[0, 1].set_title(f'{model_type.upper()} Perplexity by Topic Count', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Lower perplexity is better
            best_perp_idx = results_df['perplexity'].idxmin()
            axes[0, 1].plot(results_df.loc[best_perp_idx, 'n_topics'], 
                           results_df.loc[best_perp_idx, 'perplexity'], 
                           'r*', markersize=20)
        
        # 3. Rate of Change (First Derivative)
        coherence_diff = np.diff(results_df['coherence'])
        n_range = results_df['n_topics'].values[1:]
        axes[1, 0].bar(n_range, coherence_diff, color='steelblue', alpha=0.7)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Number of Topics', fontsize=12)
        axes[1, 0].set_ylabel('Δ Coherence (Improvement)', fontsize=12)
        axes[1, 0].set_title('Coherence Improvement per Additional Topic', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Elbow Method Visualization
        axes[1, 1].plot(results_df['n_topics'], results_df['coherence'], 
                       marker='o', linewidth=2, markersize=8, color='steelblue')
        
        # Find elbow point (point of diminishing returns)
        if len(results_df) > 3:
            # Simple elbow detection: largest distance from line connecting endpoints
            x = results_df['n_topics'].values
            y = results_df['coherence'].values
            from numpy.linalg import norm
            
            # Line from first to last point
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[-1], y[-1]])
            
            max_dist = 0
            elbow_idx = 0
            for i in range(len(x)):
                p3 = np.array([x[i], y[i]])
                dist = norm(np.cross(p2-p1, p1-p3)) / norm(p2-p1)
                if dist > max_dist:
                    max_dist = dist
                    elbow_idx = i
            
            axes[1, 1].plot(x[elbow_idx], y[elbow_idx], 'r*', markersize=20,
                           label=f'Elbow: {x[elbow_idx]} topics')
            axes[1, 1].legend()
        
        axes[1, 1].set_xlabel('Number of Topics', fontsize=12)
        axes[1, 1].set_ylabel('Coherence Score (c_v)', fontsize=12)
        axes[1, 1].set_title('Elbow Method for Optimal Topics', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('topic_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def get_recommendations(self, results_df):
        """
        Get automated recommendations for optimal topic count
        """
        recommendations = {}
        
        # Best coherence
        best_coherence = results_df.loc[results_df['coherence'].idxmax()]
        recommendations['best_coherence'] = {
            'n_topics': int(best_coherence['n_topics']),
            'coherence': float(best_coherence['coherence']),
            'reason': 'Highest coherence score'
        }
        
        # Best perplexity (if available)
        if 'perplexity' in results_df.columns:
            best_perplexity = results_df.loc[results_df['perplexity'].idxmin()]
            recommendations['best_perplexity'] = {
                'n_topics': int(best_perplexity['n_topics']),
                'perplexity': float(best_perplexity['perplexity']),
                'reason': 'Lowest perplexity (best generalization)'
            }
        
        # Elbow point
        coherence_diff = np.diff(results_df['coherence'])
        # Find where improvement drops below threshold
        threshold = np.mean(coherence_diff) * 0.3
        for i, diff in enumerate(coherence_diff):
            if diff < threshold:
                recommendations['elbow'] = {
                    'n_topics': int(results_df['n_topics'].iloc[i+1]),
                    'coherence': float(results_df['coherence'].iloc[i+1]),
                    'reason': f'Diminishing returns (Δ < {threshold:.4f})'
                }
                break
        
        # Practical range recommendation
        top_25pct = results_df.nlargest(int(len(results_df)*0.25), 'coherence')
        min_n = int(top_25pct['n_topics'].min())
        max_n = int(top_25pct['n_topics'].max())
        recommendations['practical_range'] = {
            'range': (min_n, max_n),
            'reason': f'Top 25% coherence scores: {min_n}-{max_n} topics'
        }
        
        return recommendations


def main():
    """
    Example usage
    """
    # Load your preprocessed texts (from the topic modeling script)
    # Replace this with your actual data loading
    import pandas as pd
    
    # Option 1: Load from your existing processed data
    # df = pd.read_csv('topic_modeling_with_fulltext.csv')
    # texts = df['Full_Text_Cleaned'].fillna('').tolist()
    
    # Option 2: Load from saved text files
    texts = []
    clean_text_dir = 'extracted_text_clean'
    import os
    if os.path.exists(clean_text_dir):
        for filename in os.listdir(clean_text_dir):
            if filename.endswith('_clean.txt'):
                with open(os.path.join(clean_text_dir, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    
    if len(texts) == 0:
        print("No texts found. Please load your data first.")
        return
    
    print(f"Loaded {len(texts)} documents")
    
    # Initialize optimizer
    optimizer = TopicOptimizer(texts, vectorizer_type='tfidf', max_features=5000)
    
    # Test different topic counts
    # For your ~90 papers, test range 3-25
    topic_range = range(3, 26)
    
    # Run optimization
    results_df = optimizer.grid_search_topics(topic_range, model_type='lda')
    
    # Plot results
    optimizer.plot_optimization_results(results_df, model_type='lda')
    
    # Get recommendations
    recommendations = optimizer.get_recommendations(results_df)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    for key, value in recommendations.items():
        print(f"\n{key.upper()}:")
        if 'range' in value:
            print(f"  Range: {value['range'][0]}-{value['range'][1]} topics")
        else:
            print(f"  Topics: {value['n_topics']}")
        print(f"  Reason: {value['reason']}")


if __name__ == "__main__":
    main()
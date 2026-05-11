"""
Standalone Topic Wordcloud Generator
Creates wordclouds from a trained LDA/NMF model
Run this separately after your topic modeling is complete
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class TopicWordcloudGenerator:
    """
    Generates wordclouds from a trained topic model's topic-word distributions
    """
    
    def __init__(self, model_path, vectorizer_path):
        """
        Initialize with saved model files
        
        Parameters:
        -----------
        model_path : str
            Path to pickled sklearn model (.pkl)
        vectorizer_path : str
            Path to pickled sklearn vectorizer (.pkl)
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Get feature names and topic-term matrix
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.topic_term_matrix = self.model.components_
        self.n_topics = self.topic_term_matrix.shape[0]
        
        print(f"✓ Loaded model: {self.n_topics} topics, {len(self.feature_names)} terms")
        
        # Topic labels (customize these based on your analysis)
        self.topic_labels = {
            1: "Formal-Informal Volunteer Dynamics",
            2: "Community Resilience & Climate Health",
            3: "Volunteer Motivations & Trust",
            4: "Philippine Disaster Governance",
            5: "Operational & Computational Models",
            6: "Organizational Recovery & Coordination"
        }
        
        # Color schemes
        self.topic_colors = {
            1: 'YlOrRd',
            2: 'Blues',
            3: 'Greens',
            4: 'Purples',
            5: 'Oranges',
            6: 'RdPu'
        }
    
    def calculate_relevance(self, topic_idx, lambda_val=0.6, n_terms=50):
        """
        Calculate relevance scores for terms in a topic
        
        Relevance = λ * log(P(term|topic)) + (1-λ) * log(P(term|topic)/P(term))
        
        Parameters:
        -----------
        topic_idx : int (0-based)
        lambda_val : float (0-1)
            1.0 = pure frequency, 0.0 = pure exclusivity
        n_terms : int
            Number of top terms to return
        
        Returns:
        --------
        dict : {term: relevance_score}
        """
        # Get topic weights
        topic_weights = self.topic_term_matrix[topic_idx]
        
        # P(term | topic)
        p_term_topic = topic_weights / topic_weights.sum()
        
        # P(term) - marginal across all topics
        total_weights = self.topic_term_matrix.sum(axis=0)
        p_term = total_weights / total_weights.sum()
        
        # Calculate relevance
        scores = {}
        for i, term in enumerate(self.feature_names):
            p_tt = p_term_topic[i]
            p_t = p_term[i]
            
            if p_tt > 0 and p_t > 0:
                freq_score = np.log(p_tt)
                lift_score = np.log(p_tt / p_t)
                relevance = lambda_val * freq_score + (1 - lambda_val) * lift_score
                scores[term] = relevance
        
        # Sort and return top N
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_terms[:n_terms])
    
    def create_single_wordcloud(self, topic_num, lambda_val=0.6, save=True):
        """
        Create and save a wordcloud for a single topic
        
        Parameters:
        -----------
        topic_num : int (1-based)
        lambda_val : float
        save : bool
        """
        topic_idx = topic_num - 1
        
        # Get scores
        term_scores = self.calculate_relevance(topic_idx, lambda_val)
        
        # Scale scores for word sizing
        scores = list(term_scores.values())
        max_s, min_s = max(scores), min(scores)
        
        word_freq = {}
        for word, score in term_scores.items():
            scaled = 10 + (score - min_s) / (max_s - min_s) * 90 if max_s > min_s else 50
            word_freq[word] = scaled
        
        # Create wordcloud
        colormap = self.topic_colors.get(topic_num, 'viridis')
        wordcloud = WordCloud(
            width=800, height=500,
            background_color='white',
            colormap=colormap,
            max_words=50,
            min_font_size=10,
            max_font_size=150,
            relative_scaling=0.5,
            prefer_horizontal=0.7,
            random_state=42,
            collocations=False
        ).generate_from_frequencies(word_freq)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        label = self.topic_labels.get(topic_num, f"Topic {topic_num}")
        ax.set_title(f"Topic {topic_num}: {label}\n(λ = {lambda_val})", 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            os.makedirs('wordclouds', exist_ok=True)
            path = f'wordclouds/topic_{topic_num}_wordcloud.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {path}")
        
        plt.show()
        return fig
    
    def create_all_wordclouds(self, lambda_val=0.6):
        """
        Create wordclouds for all topics
        """
        print(f"\n{'='*50}")
        print(f"Generating {self.n_topics} wordclouds (λ = {lambda_val})")
        print(f"{'='*50}\n")
        
        for topic_num in range(1, self.n_topics + 1):
            label = self.topic_labels.get(topic_num, f"Topic {topic_num}")
            print(f"Topic {topic_num}: {label}")
            
            term_scores = self.calculate_relevance(topic_num - 1, lambda_val, n_terms=20)
            print(f"  Top terms: {', '.join(list(term_scores.keys())[:12])}")
            
            self.create_single_wordcloud(topic_num, lambda_val, save=True)
            print()
    
    def create_comparison_grid(self, lambda_val=0.6):
        """
        Create a single image with all topic wordclouds side by side
        """
        # Determine grid layout
        if self.n_topics <= 3:
            rows, cols = 1, self.n_topics
        elif self.n_topics <= 4:
            rows, cols = 2, 2
        elif self.n_topics <= 6:
            rows, cols = 2, 3
        else:
            rows = (self.n_topics + 2) // 3
            cols = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for topic_num in range(1, self.n_topics + 1):
            i = topic_num - 1
            term_scores = self.calculate_relevance(i, lambda_val, n_terms=40)
            
            if term_scores:
                scores = list(term_scores.values())
                max_s, min_s = max(scores), min(scores)
                
                word_freq = {}
                for word, score in term_scores.items():
                    scaled = 10 + (score - min_s) / (max_s - min_s) * 90 if max_s > min_s else 50
                    word_freq[word] = scaled
                
                colormap = self.topic_colors.get(topic_num, 'viridis')
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white', colormap=colormap,
                    max_words=40, min_font_size=8, max_font_size=80,
                    relative_scaling=0.5, random_state=42, collocations=False
                ).generate_from_frequencies(word_freq)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
            
            label = self.topic_labels.get(topic_num, f"Topic {topic_num}")
            axes[i].set_title(f"Topic {topic_num}: {label}", fontsize=11, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(self.n_topics, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f'Topic Wordclouds Comparison (λ = {lambda_val})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs('wordclouds', exist_ok=True)
        path = 'wordclouds/all_topics_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved comparison grid: {path}")
        plt.show()
    
    def export_term_table(self, lambda_val=0.6):
        """
        Export top terms per topic as CSV
        """
        data = []
        for topic_num in range(1, self.n_topics + 1):
            term_scores = self.calculate_relevance(topic_num - 1, lambda_val, n_terms=20)
            label = self.topic_labels.get(topic_num, f"Topic {topic_num}")
            
            for rank, (term, score) in enumerate(term_scores.items(), 1):
                data.append({
                    'Topic': topic_num,
                    'Topic_Label': label,
                    'Rank': rank,
                    'Term': term,
                    'Relevance_Score': round(score, 4)
                })
        
        df = pd.DataFrame(data)
        os.makedirs('wordclouds', exist_ok=True)
        path = 'wordclouds/topic_terms.csv'
        df.to_csv(path, index=False)
        print(f"✓ Saved term table: {path}")
        return df
    
    def compare_lambda(self, topic_num=1):
        """
        Show how lambda affects term rankings for a topic
        """
        print(f"\n{'='*60}")
        print(f"EFFECT OF λ ON TOPIC {topic_num} TERM RANKINGS")
        print(f"{'='*60}")
        print(f"{'Term':<25} {'λ=0.0':>10} {'λ=0.3':>10} {'λ=0.6':>10} {'λ=1.0':>10}")
        print(f"{'Rank (exclusive→frequent)':>25}")
        print("-" * 70)
        
        lambda_results = {}
        for lam in [0.0, 0.3, 0.6, 1.0]:
            lambda_results[lam] = self.calculate_relevance(topic_num - 1, lam, n_terms=30)
        
        # Get unique terms across all lambda values
        all_terms = []
        for lam in [0.0, 0.3, 0.6, 1.0]:
            all_terms.extend(lambda_results[lam].keys())
        all_terms = list(dict.fromkeys(all_terms))[:20]  # Unique, preserve order
        
        for term in all_terms:
            print(f"{term:<25}", end="")
            for lam in [0.0, 0.3, 0.6, 1.0]:
                score = lambda_results[lam].get(term, None)
                if score is not None:
                    print(f"{score:>10.4f}", end="")
                else:
                    print(f"{'--':>10}", end="")
            print()


# ==================== MAIN ====================

def main():
    # Step 1: First save your model and vectorizer from your topic modeling script
    # Add these lines to your topic modeling script:
    #
    # import pickle
    # pickle.dump(modeler.lda_model, open('lda_model.pkl', 'wb'))
    # pickle.dump(modeler.vectorizer, open('vectorizer.pkl', 'wb'))
    
    # ==================== CONFIGURATION ====================
    MODEL_PATH = 'lda_model.pkl'       # Path to saved model
    VECTORIZER_PATH = 'vectorizer.pkl'  # Path to saved vectorizer
    LAMBDA = 0.6                        # Relevance weighting
    # ======================================================
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        print("\nFirst save your model from your topic modeling script:")
        print("  import pickle")
        print("  pickle.dump(modeler.lda_model, open('lda_model.pkl', 'wb'))")
        print("  pickle.dump(modeler.vectorizer, open('vectorizer.pkl', 'wb'))")
        return
    
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Error: Vectorizer file not found: {VECTORIZER_PATH}")
        return
    
    # Load and generate
    print("Loading model...")
    generator = TopicWordcloudGenerator(MODEL_PATH, VECTORIZER_PATH)
    
    # Generate individual wordclouds
    generator.create_all_wordclouds(lambda_val=LAMBDA)
    
    # Create comparison grid
    generator.create_comparison_grid(lambda_val=LAMBDA)
    
    # Export term table
    generator.export_term_table(lambda_val=LAMBDA)
    
    # Show lambda comparison for Topic 1
    generator.compare_lambda(topic_num=1)
    
    print(f"\n{'='*50}")
    print("DONE! Check the 'wordclouds/' folder for outputs:")
    print("  - topic_X_wordcloud.png (individual wordclouds)")
    print("  - all_topics_comparison.png (comparison grid)")
    print("  - topic_terms.csv (term table)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
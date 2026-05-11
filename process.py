"""
Topic Modeling Script for Scopus CSV Data
Fixed version - handles pyLDAvis import issues
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Topic Modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Gensim for LDA
from gensim import corpora, models
from gensim.models import CoherenceModel

# Handle pyLDAvis import with fallback
try:
    import pyLDAvis
    import pyLDAvis.gensim_models
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False
    print("Warning: pyLDAvis not available. Interactive visualization will be skipped.")
    print("Install with: pip install pyLDAvis")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class TopicModeler:
    """
    A comprehensive class for topic modeling on academic publication data
    """
    
    def __init__(self, csv_path, text_column='Abstract', n_topics=10):
        """
        Initialize the TopicModeler
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        text_column : str
            Column containing text for topic modeling (default: 'Abstract')
        n_topics : int
            Number of topics to extract
        """
        self.csv_path = csv_path
        self.text_column = text_column
        self.n_topics = n_topics
        self.df = None
        self.processed_docs = None
        self.vectorizer = None
        self.document_term_matrix = None
        self.lda_model = None
        self.nmf_model = None
        self.gensim_lda = None
        self.gensim_dictionary = None
        self.gensim_corpus = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Custom stop words for academic papers
        self.custom_stop_words = set(stopwords.words('english')).union({
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
            'copyright', 'license', 'cc', 'creative', 'commons',  'cid'
        })
    
    def load_data(self):
        """Load the CSV file"""
        print("Loading data...")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} records")
            print(f"Columns available: {list(self.df.columns)}")
            
            # Check if text column exists
            if self.text_column not in self.df.columns:
                print(f"Warning: '{self.text_column}' column not found!")
                print("Available text columns:")
                text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
                for col in text_cols:
                    print(f"  - {col}")
                return False
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def preprocess_text(self, text):
        """
        Preprocess a single document
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and short words
        tokens = [token for token in tokens 
                 if token not in self.custom_stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_documents(self):
        """Preprocess all documents"""
        print("Preprocessing documents...")
        
        if self.text_column not in self.df.columns:
            print("Cannot preprocess: text column not found")
            return False
        
        # Handle missing values
        self.df[self.text_column] = self.df[self.text_column].fillna('')
        
        # Apply preprocessing
        self.processed_docs = self.df[self.text_column].apply(self.preprocess_text)
        
        # Remove empty documents
        valid_docs = self.processed_docs.str.len() > 0
        print(f"Valid documents after preprocessing: {valid_docs.sum()}/{len(valid_docs)}")
        
        # Reset index to maintain alignment
        self.processed_docs = self.processed_docs[valid_docs].reset_index(drop=True)
        self.df = self.df[valid_docs].reset_index(drop=True)
        
        return True
    
    def create_document_term_matrix(self, method='tfidf', max_features=5000, 
                                   min_df=5, max_df=0.95):
        """
        Create document-term matrix
        
        Parameters:
        -----------
        method : str
            'tf' for CountVectorizer or 'tfidf' for TfidfVectorizer
        max_features : int
            Maximum number of features
        min_df : int
            Minimum document frequency
        max_df : float
            Maximum document frequency
        """
        print(f"Creating document-term matrix using {method.upper()}...")
        
        if method == 'tf':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, 2),  # Unigrams and bigrams
                stop_words='english'
            )
        else:  # tfidf
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, 2),
                stop_words='english',
                sublinear_tf=True
            )
        
        self.document_term_matrix = self.vectorizer.fit_transform(self.processed_docs)
        print(f"Document-term matrix shape: {self.document_term_matrix.shape}")
        
        return True
    
    def run_lda(self, max_iter=10, learning_method='online', random_state=42):
        """
        Run LDA topic modeling using sklearn
        """
        print(f"Running sklearn LDA with {self.n_topics} topics...")
        
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=random_state,
            n_jobs=-1,
            learning_offset=50.,
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        
        self.lda_model.fit(self.document_term_matrix)
        
        # Calculate perplexity
        perplexity = self.lda_model.perplexity(self.document_term_matrix)
        print(f"Sklearn LDA Perplexity: {perplexity:.2f}")
        
        return self.lda_model
    
    def run_nmf(self, random_state=42):
        """
        Run NMF topic modeling
        """
        print(f"Running NMF with {self.n_topics} topics...")
        
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=random_state,
            alpha_W=0.1,
            alpha_H=0.1,
            l1_ratio=0.5,
            max_iter=200
        )
        
        self.nmf_model.fit(self.document_term_matrix)
        
        return self.nmf_model
    
    def run_gensim_lda(self, passes=10, random_state=42):
        """
        Run Gensim LDA with coherence optimization
        Returns the gensim model for better visualization
        """
        print("Running Gensim LDA model...")
        
        # Create dictionary and corpus for gensim
        texts = [doc.split() for doc in self.processed_docs]
        self.gensim_dictionary = corpora.Dictionary(texts)
        
        # Filter extremes
        self.gensim_dictionary.filter_extremes(no_below=5, no_above=0.95)
        
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in texts]
        
        # Train the model
        self.gensim_lda = models.LdaMulticore(
            corpus=self.gensim_corpus,
            id2word=self.gensim_dictionary,
            num_topics=self.n_topics,
            random_state=random_state,
            passes=passes,
            workers=3,
            # alpha='auto',
            eta='auto'
        )
        
        # Calculate coherence using multiple measures
        coherence_models = {
            'c_v': CoherenceModel(model=self.gensim_lda, texts=texts, 
                                 dictionary=self.gensim_dictionary, coherence='c_v'),
            'u_mass': CoherenceModel(model=self.gensim_lda, corpus=self.gensim_corpus, 
                                    dictionary=self.gensim_dictionary, coherence='u_mass')
        }
        
        for measure, cm in coherence_models.items():
            coherence = cm.get_coherence()
            print(f"Gensim LDA Coherence ({measure}): {coherence:.3f}")
        
        return self.gensim_lda, self.gensim_dictionary, self.gensim_corpus
    
    def display_topics(self, model_type='lda', n_top_words=15):
        """
        Display topics with their top words
        """
        if model_type == 'lda' and self.lda_model is not None:
            model = self.lda_model
            feature_names = self.vectorizer.get_feature_names_out()
        elif model_type == 'nmf' and self.nmf_model is not None:
            model = self.nmf_model
            feature_names = self.vectorizer.get_feature_names_out()
        elif model_type == 'gensim' and self.gensim_lda is not None:
            print("\nGensim LDA Topics:")
            for idx, topic in self.gensim_lda.print_topics(-1, num_words=n_top_words):
                print(f"\nTopic {idx + 1}:")
                print(topic)
            return self.gensim_lda.print_topics(-1, num_words=n_top_words)
        else:
            print(f"No {model_type.upper()} model found")
            return None
        
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            topics[f"Topic {topic_idx + 1}"] = dict(zip(top_features, weights))
            
            print(f"\nTopic {topic_idx + 1}:")
            print("Top words:", ", ".join([f"{word} ({weight:.3f})" 
                                         for word, weight in zip(top_features[:10], weights[:10])]))
        
        return topics
    
    def get_document_topics(self, model_type='lda'):
        """
        Get topic distribution for each document
        """
        if model_type == 'lda' and self.lda_model is not None:
            model = self.lda_model
            doc_topics = model.transform(self.document_term_matrix)
        elif model_type == 'nmf' and self.nmf_model is not None:
            model = self.nmf_model
            doc_topics = model.transform(self.document_term_matrix)
        elif model_type == 'gensim' and self.gensim_lda is not None:
            # Get topic distribution from gensim
            doc_topics = []
            for doc in self.gensim_corpus:
                topic_dist = self.gensim_lda.get_document_topics(doc, minimum_probability=0)
                topic_array = np.zeros(self.n_topics)
                for topic_id, prob in topic_dist:
                    topic_array[topic_id] = prob
                doc_topics.append(topic_array)
            doc_topics = np.array(doc_topics)
        else:
            print(f"No {model_type.upper()} model found")
            return None
        
        # Add topic columns to dataframe
        for i in range(self.n_topics):
            self.df[f'Topic_{i+1}_Weight'] = doc_topics[:, i]
        
        # Get dominant topic for each document
        self.df['Dominant_Topic'] = doc_topics.argmax(axis=1) + 1
        self.df['Dominant_Topic_Weight'] = doc_topics.max(axis=1)
        
        return doc_topics
    
    def visualize_topics_interactive(self):
        """
        Create interactive visualization using pyLDAvis (if available)
        Uses gensim model for better compatibility
        """
        if not PYLDAVIS_AVAILABLE:
            print("pyLDAvis is not available. Skipping interactive visualization.")
            print("To install: pip install pyLDAvis")
            return None
        
        if self.gensim_lda is None:
            print("Gensim LDA model not found. Running it now...")
            self.run_gensim_lda()
        
        try:
            print("Preparing interactive visualization...")
            vis_data = pyLDAvis.gensim_models.prepare(
                self.gensim_lda, 
                self.gensim_corpus, 
                self.gensim_dictionary,
                mds='tsne'
            )
            return vis_data
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Falling back to static visualizations only.")
            return None
    
    def plot_topic_distribution(self):
        """
        Plot the distribution of topics across documents
        """
        if 'Dominant_Topic' not in self.df.columns:
            print("Please run get_document_topics() first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Topic distribution
        topic_counts = self.df['Dominant_Topic'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_counts)))
        axes[0].bar(topic_counts.index, topic_counts.values, color=colors, edgecolor='black')
        axes[0].set_xlabel('Topic Number', fontsize=12)
        axes[0].set_ylabel('Number of Documents', fontsize=12)
        axes[0].set_title('Distribution of Dominant Topics', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(1, self.n_topics + 1))
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(topic_counts.values):
            axes[0].text(topic_counts.index[i], v + 1, str(v), ha='center')
        
        # Average topic weight
        avg_weights = []
        for i in range(1, self.n_topics + 1):
            col_name = f'Topic_{i}_Weight'
            if col_name in self.df.columns:
                avg_weights.append(self.df[col_name].mean())
            else:
                avg_weights.append(0)
        
        axes[1].bar(range(1, self.n_topics + 1), avg_weights, 
                   color=colors, edgecolor='black')
        axes[1].set_xlabel('Topic Number', fontsize=12)
        axes[1].set_ylabel('Average Weight', fontsize=12)
        axes[1].set_title('Average Topic Weights Across Documents', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(1, self.n_topics + 1))
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_token_chart(self):
        labels = [
            'Formal-informal responder dynamics', 
            'Resilience to climate hazards', 
            'Psychological drivers of volunteerism', 
            'Philippine disaster governance',
            'Computational optimization and modeling',
            'Recovery and coordination'
        ]
        sizes = [22.8, 20.0, 18.7, 13.6, 13.3, 11.6]
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red', 'orange']

        plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 

        # plt.title("Thematic token share per topic")
        plt.show()

        return 0
    
    def plot_topic_heatmap(self, n_top_words=10):
        """
        Create a heatmap of top words per topic
        """
        if self.lda_model is None and self.nmf_model is None:
            print("No sklearn model found. Please run LDA or NMF first.")
            return
        
        model = self.lda_model if self.lda_model is not None else self.nmf_model
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get unique top words across all topics
        all_top_words = set()
        for topic in model.components_:
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            all_top_words.update(top_features)
        
        all_top_words = sorted(list(all_top_words))
        
        # Create heatmap data
        heatmap_data = np.zeros((self.n_topics, len(all_top_words)))
        for i, topic in enumerate(model.components_):
            for j, word in enumerate(all_top_words):
                word_idx = self.vectorizer.vocabulary_.get(word)
                if word_idx is not None:
                    heatmap_data[i, j] = topic[word_idx]
        
        # Plot with better formatting
        fig, ax = plt.subplots(figsize=(max(15, len(all_top_words) * 0.5), 
                                      max(8, self.n_topics * 0.8)))
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   xticklabels=all_top_words, 
                   yticklabels=[f'Topic {i+1}' for i in range(self.n_topics)],
                   cbar_kws={'label': 'Weight'},
                   ax=ax,
                   annot_kws={'size': 8})
        
        ax.set_title('Topic-Word Distribution Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Words', fontsize=12)
        ax.set_ylabel('Topics', fontsize=12)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def save_results(self, output_path, model_type='lda'):
        """
        Save results to CSV
        """
        output_df = self.df.copy()
        
        # Add topic keywords if available
        if 'Dominant_Topic' in output_df.columns:
            if model_type == 'lda' and self.lda_model is not None:
                model = self.lda_model
                feature_names = self.vectorizer.get_feature_names_out()
            elif model_type == 'nmf' and self.nmf_model is not None:
                model = self.nmf_model
                feature_names = self.vectorizer.get_feature_names_out()
            elif model_type == 'gensim' and self.gensim_lda is not None:
                # For gensim, extract keywords differently
                topic_keywords = {}
                for topic_id in range(self.n_topics):
                    words = self.gensim_lda.show_topic(topic_id, topn=5)
                    topic_keywords[topic_id + 1] = ', '.join([word for word, _ in words])
                output_df['Topic_Keywords'] = output_df['Dominant_Topic'].map(topic_keywords)
                output_df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
                return output_path
            else:
                output_df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
                return output_path
            
            # Create topic keyword mapping
            topic_keywords = {}
            for topic_idx, topic in enumerate(model.components_):
                top_features_ind = topic.argsort()[:-6:-1]  # Top 5 words
                top_features = [feature_names[i] for i in top_features_ind]
                topic_keywords[topic_idx + 1] = ', '.join(top_features)
            
            output_df['Topic_Keywords'] = output_df['Dominant_Topic'].map(topic_keywords)
        
        output_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return output_path
    
    def analyze_topic_trends(self, year_column='Year'):
        """
        Analyze topic trends over years
        """
        if year_column not in self.df.columns:
            print(f"Column '{year_column}' not found")
            return
        
        if 'Dominant_Topic' not in self.df.columns:
            print("Please run get_document_topics() first")
            return
        
        # Group by year and topic
        yearly_topics = self.df.groupby([year_column, 'Dominant_Topic']).size().unstack(fill_value=0)
        
        # Calculate percentage
        yearly_topics_pct = yearly_topics.div(yearly_topics.sum(axis=1), axis=0) * 100
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Absolute counts
        yearly_topics.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab20')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Number of Documents', fontsize=12)
        axes[0].set_title('Topic Trends Over Years (Absolute Counts)', fontsize=14, fontweight='bold')
        axes[0].legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Percentage
        yearly_topics_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab20')
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Percentage of Documents (%)', fontsize=12)
        axes[1].set_title('Topic Trends Over Years (Percentage)', fontsize=14, fontweight='bold')
        axes[1].legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def analyze_author_topics(self, author_column='Authors', top_n_authors=10):
        """
        Analyze topic preferences by top authors
        """
        if author_column not in self.df.columns:
            print(f"Column '{author_column}' not found")
            return
        
        if 'Dominant_Topic' not in self.df.columns:
            print("Please run get_document_topics() first")
            return
        
        # Get top authors
        author_counts = self.df[author_column].value_counts().head(top_n_authors)
        top_authors = author_counts.index
        
        # Create author-topic matrix
        author_topics = pd.crosstab(self.df[author_column], self.df['Dominant_Topic'])
        author_topics = author_topics.loc[top_authors]
        
        # Normalize by row (percentage)
        author_topics_pct = author_topics.div(author_topics.sum(axis=1), axis=0) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, max(8, top_n_authors * 0.6)))
        sns.heatmap(author_topics_pct, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='YlOrRd',
                   xticklabels=[f'Topic {i}' for i in author_topics_pct.columns],
                   ax=ax,
                   cbar_kws={'label': 'Percentage (%)'})
        
        ax.set_title(f'Topic Distribution by Top {top_n_authors} Authors (%)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Topic', fontsize=12)
        ax.set_ylabel('Author', fontsize=12)
        plt.tight_layout()
        return fig


def main():
    """
    Main function to run topic modeling
    """
    # ==================== CONFIGURATION ====================
    # Modify these parameters based on your needs
    CSV_PATH = 'topic_modeling_with_fulltext.csv'  # Replace with your CSV file path
    TEXT_COLUMN = 'Full_Text_Cleaned'     # Column containing text for analysis
    # N_TOPICS = 10                # Number of topics to extract
    N_TOPICS = 6
    # ======================================================
    
    # Initialize topic modeler
    modeler = TopicModeler(
        csv_path=CSV_PATH,
        text_column=TEXT_COLUMN,
        n_topics=N_TOPICS
    )
    
    # Step 1: Load and process data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    if not modeler.load_data():
        print("Failed to load data. Exiting.")
        return
    
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING TEXT")
    print("="*60)
    if not modeler.preprocess_documents():
        print("Failed to preprocess documents. Exiting.")
        return
    
    # Step 3: Create document-term matrix
    print("\n" + "="*60)
    print("STEP 3: CREATING DOCUMENT-TERM MATRIX")
    print("="*60)
    modeler.create_document_term_matrix(
        method='tfidf',
        max_features=5000,
        min_df=5,
        max_df=0.95
    )
    
    # Step 4: Run multiple topic modeling approaches
    print("\n" + "="*60)
    print("STEP 4: RUNNING TOPIC MODELS")
    print("="*60)
    
    # Sklearn LDA
    print("\n--- Sklearn LDA ---")
    modeler.run_lda(max_iter=10)
    modeler.display_topics(model_type='lda', n_top_words=15)
    
    # NMF
    print("\n--- NMF ---")
    modeler.run_nmf()
    modeler.display_topics(model_type='nmf', n_top_words=15)
    
    # Gensim LDA
    print("\n--- Gensim LDA (with coherence scores) ---")
    modeler.run_gensim_lda(passes=10)
    modeler.display_topics(model_type='gensim', n_top_words=15)
    
    # Get document topics (using sklearn LDA)
    print("\n" + "="*60)
    print("STEP 5: ASSIGNING TOPICS TO DOCUMENTS")
    print("="*60)
    doc_topics = modeler.get_document_topics(model_type='lda')
    print(f"Topic assignments complete for {len(modeler.df)} documents")
    
    # Step 6: Create visualizations
    print("\n" + "="*60)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*60)
    
    # Topic distribution
    print("\nCreating topic distribution plots...")
    fig_dist = modeler.plot_topic_distribution()
    fig_dist.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: topic_distribution.png")
    
    # Topic heatmap
    print("Creating topic-word heatmap...")
    fig_heatmap = modeler.plot_topic_heatmap(n_top_words=15)
    fig_heatmap.savefig('topic_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: topic_heatmap.png")

    # Topic Token Share
    modeler.plot_token_chart()
    
    # Interactive visualization (if pyLDAvis is available)
    if PYLDAVIS_AVAILABLE:
        print("Creating interactive visualization...")
        vis_data = modeler.visualize_topics_interactive()
        if vis_data is not None:
            pyLDAvis.save_html(vis_data, 'topic_modeling_interactive.html')
            print("✓ Saved: topic_modeling_interactive.html")
    
    # Trend analysis (if Year column exists)
    if 'Year' in modeler.df.columns:
        print("Creating topic trend plots...")
        fig_trends = modeler.analyze_topic_trends(year_column='Year')
        if fig_trends:
            fig_trends.savefig('topic_trends_over_years.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: topic_trends_over_years.png")
    
    # Author-topic analysis
    print("Creating author-topic analysis...")
    fig_authors = modeler.analyze_author_topics(author_column='Authors', top_n_authors=10)
    if fig_authors:
        fig_authors.savefig('author_topic_preferences.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: author_topic_preferences.png")
    
    # Step 7: Save results
    print("\n" + "="*60)
    print("STEP 7: SAVING RESULTS")
    print("="*60)
    output_path = 'topic_modeling_results.csv'
    modeler.save_results(output_path, model_type='lda')
    print(f"✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TOPIC MODELING SUMMARY")
    print("="*60)
    print(f"\nTotal documents analyzed: {len(modeler.df)}")
    print(f"Number of topics: {N_TOPICS}")
    
    if 'Dominant_Topic' in modeler.df.columns:
        print(f"\nTopic Distribution:")
        topic_dist = modeler.df['Dominant_Topic'].value_counts().sort_index()
        for topic, count in topic_dist.items():
            pct = count / len(modeler.df) * 100
            bar = '█' * int(pct / 2)
            print(f"  Topic {topic:2d}: {count:4d} documents ({pct:5.1f}%) {bar}")
    
    # Show top documents for each topic
    print("\n" + "="*60)
    print("TOP DOCUMENTS PER TOPIC")
    print("="*60)
    for topic_num in range(1, N_TOPICS + 1):
        topic_docs = modeler.df[modeler.df['Dominant_Topic'] == topic_num].head(3)
        if len(topic_docs) > 0:
            print(f"\nTopic {topic_num} - Top Documents:")
            for idx, row in topic_docs.iterrows():
                title = str(row.get('Title', 'N/A'))[:100]
                print(f"  • {title}...")
    
    print("\n" + "="*60)
    print("SCRIPT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  📄 topic_modeling_results.csv - Full results with topic assignments")
    print("  📊 topic_distribution.png - Topic distribution charts")
    print("  📊 topic_heatmap.png - Topic-word heatmap")
    if PYLDAVIS_AVAILABLE:
        print("  🌐 topic_modeling_interactive.html - Interactive visualization")
    if 'Year' in modeler.df.columns:
        print("  📊 topic_trends_over_years.png - Topic trends over years")
    print("  📊 author_topic_preferences.png - Author topic preferences")


if __name__ == "__main__":
    main()
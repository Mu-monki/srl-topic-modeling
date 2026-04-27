"""
PDF Text Extraction and Preprocessing Script
Extracts text from OCR PDFs, saves as cleaned text files, and removes academic stop words
"""

import os
import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# PDF text extraction
try:
    import PyPDF2
    PYRPDF2_AVAILABLE = True
except ImportError:
    PYRPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not available. Install with: pip install pdfplumber")

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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


class PDFTextExtractor:
    """
    Extracts and preprocesses text from OCR PDFs for topic modeling
    """
    
    def __init__(self, csv_path, pdf_folder, index_column='EID', pdf_pattern='{index}.pdf'):
        """
        Initialize the PDF text extractor
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file with index column
        pdf_folder : str
            Path to folder containing PDF files
        index_column : str
            Column name in CSV that matches PDF filenames
        pdf_pattern : str
            Pattern for PDF filenames (use {index} as placeholder)
        """
        self.csv_path = csv_path
        self.pdf_folder = pdf_folder
        self.index_column = index_column
        self.pdf_pattern = pdf_pattern
        self.df = None
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Common English stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Academic and research paper specific stop words
        self.academic_stop_words = {
            # Paper structure words
            'abstract', 'introduction', 'background', 'methodology', 'methods',
            'materials', 'results', 'discussion', 'conclusion', 'conclusions',
            'acknowledgements', 'acknowledgments', 'references', 'appendix',
            'appendices', 'supplementary', 'supplement', 'figure', 'figures',
            'table', 'tables', 'equation', 'equations', 'section', 'sections',
            
            # Research terminology (context-dependent, remove if too aggressive)
            'study', 'research', 'paper', 'article', 'work', 'data', 'analysis',
            'findings', 'result', 'finding', 'literature', 'review', 'survey',
            'interview', 'questionnaire', 'sample', 'participant', 'participants',
            'respondent', 'respondents', 'variable', 'variables', 'factor',
            'factors', 'model', 'models', 'framework', 'approach', 'approaches',
            
            # Academic writing phrases
            'et', 'al', 'e.g', 'i.e', 'etc', 'via', 'per', 'versus', 'vs',
            'however', 'therefore', 'thus', 'hence', 'furthermore', 'moreover',
            'nevertheless', 'nonetheless', 'although', 'though', 'despite',
            'regarding', 'concerning', 'according', 'accordingly', 'consequently',
            'subsequently', 'finally', 'lastly', 'firstly', 'secondly', 'thirdly',
            
            # Verbs common in academic writing
            'used', 'using', 'based', 'shown', 'found', 'observed', 'reported',
            'identified', 'examined', 'investigated', 'analysed', 'analyzed',
            'explored', 'considered', 'suggested', 'indicated', 'demonstrated',
            'revealed', 'proposed', 'presented', 'discussed', 'described',
            'performed', 'conducted', 'carried', 'following', 'followed',
            'included', 'including', 'related', 'associated', 'compared',
            
            # Quantitative terms
            'significant', 'significantly', 'positive', 'negative', 'higher',
            'lower', 'greater', 'lesser', 'increased', 'decreased', 'related',
            'associated', 'correlated', 'predicted', 'explained', 'contributed',
            
            # Publishing/copyright terms
            'copyright', 'license', 'licensed', 'creative', 'commons',
            'published', 'publisher', 'publishing', 'elsevier', 'springer',
            'wiley', 'taylor', 'francis', 'routledge', 'sage', 'ieee',
            'acm', 'doi', 'issn', 'isbn', 'http', 'https', 'www',
            'crossref', 'crossmark', 'pubmed', 'medline', 'scopus',
            
            # PDF artifacts
            'page', 'pages', 'pdf', 'download', 'downloaded', 'file',
            'author', 'authors', 'manuscript', 'version', 'preprint',
            'postprint', 'pubdate', 'publication', 'journal', 'volume',
            'issue', 'number', 'citation', 'cited', 'citing',
            
            # Common in volunteer/disaster research
            'disaster', 'volunteer', 'volunteers', 'volunteering', 
            'emergency', 'response', 'community', 'communities',
            'management', 'risk', 'crisis', 'hazard', 'hazards',
        }
        
        # Combine all stop words
        self.all_stop_words = self.stop_words.union(self.academic_stop_words)
        
        # Create output directories
        self.raw_text_dir = 'extracted_text_raw'
        self.clean_text_dir = 'extracted_text_clean'
        os.makedirs(self.raw_text_dir, exist_ok=True)
        os.makedirs(self.clean_text_dir, exist_ok=True)
    
    def load_csv(self):
        """Load the CSV file"""
        print("Loading CSV file...")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} records from CSV")
            
            if self.index_column not in self.df.columns:
                print(f"Error: Column '{self.index_column}' not found in CSV")
                print(f"Available columns: {list(self.df.columns)}")
                return False
            
            print(f"Found {self.df[self.index_column].nunique()} unique indices")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_path, method='auto'):
        """
        Extract text from a PDF file
        
        Parameters:
        -----------
        pdf_path : str
            Path to the PDF file
        method : str
            'auto', 'pdfplumber', or 'pypdf2'
        
        Returns:
        --------
        str : Extracted text or empty string if failed
        """
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        if method in ['auto', 'pdfplumber'] and PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except Exception as e:
                if method == 'pdfplumber':
                    print(f"  pdfplumber failed: {e}")
        
        # Fall back to PyPDF2
        if method in ['auto', 'pypdf2'] and PYRPDF2_AVAILABLE:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except Exception as e:
                if method == 'pypdf2':
                    print(f"  PyPDF2 failed: {e}")
        
        return text
    
    def clean_text(self, text, remove_stopwords=True, lemmatize=True):
        """
        Clean and preprocess extracted text
        
        Parameters:
        -----------
        text : str
            Raw text to clean
        remove_stopwords : bool
            Whether to remove stop words
        lemmatize : bool
            Whether to lemmatize words
        
        Returns:
        --------
        str : Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove DOIs
        text = re.sub(r'doi\s*:\s*\S+', '', text)
        
        # Remove citation patterns like [1], [2,3], [1-5]
        text = re.sub(r'\[\d+(?:[,-]\d+)*\]', '', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*\n', '\n', text, flags=re.MULTILINE)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove short lines (likely headers/footers)
        lines = text.split('.')
        lines = [l.strip() for l in lines if len(l.strip()) > 20]
        text = '. '.join(lines)
        
        if remove_stopwords or lemmatize:
            # Tokenize
            tokens = word_tokenize(text)
            
            if remove_stopwords:
                # Remove stop words and short words
                tokens = [token for token in tokens 
                         if token not in self.all_stop_words and len(token) > 2]
            
            if lemmatize:
                # Lemmatize
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            text = ' '.join(tokens)
        
        return text
    
    def process_single_pdf(self, index_value, save_raw=True, save_clean=True):
        """
        Process a single PDF file
        
        Parameters:
        -----------
        index_value : str
            The index value matching the PDF filename
        save_raw : bool
            Whether to save raw extracted text
        save_clean : bool
            Whether to save cleaned text
        
        Returns:
        --------
        dict : {'index': str, 'raw_text': str, 'clean_text': str, 'success': bool}
        """
        # Construct PDF filename
        pdf_filename = self.pdf_pattern.format(index=index_value)
        pdf_path = os.path.join(self.pdf_folder, pdf_filename)
        
        result = {
            'index': index_value,
            'raw_text': '',
            'clean_text': '',
            'success': False,
            'error': None
        }
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            result['error'] = f"PDF not found: {pdf_path}"
            return result
        
        print(f"Processing: {pdf_filename}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            result['error'] = "No text extracted from PDF"
            return result
        
        result['raw_text'] = raw_text
        
        # Save raw text
        if save_raw:
            raw_filename = f"{index_value}_raw.txt"
            raw_path = os.path.join(self.raw_text_dir, raw_filename)
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print(f"  ✓ Saved raw text ({len(raw_text):,} chars)")
        
        # Clean text
        clean_text = self.clean_text(raw_text)
        
        if clean_text.strip():
            result['clean_text'] = clean_text
            result['success'] = True
            
            # Save clean text
            if save_clean:
                clean_filename = f"{index_value}_clean.txt"
                clean_path = os.path.join(self.clean_text_dir, clean_filename)
                with open(clean_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                print(f"  ✓ Saved clean text ({len(clean_text):,} chars)")
        else:
            result['error'] = "Text became empty after cleaning"
        
        return result
    
    def process_all_pdfs(self, limit=None, save_raw=True, save_clean=True):
        """
        Process all PDFs listed in the CSV
        
        Parameters:
        -----------
        limit : int or None
            Limit number of PDFs to process (for testing)
        save_raw : bool
            Whether to save raw extracted text
        save_clean : bool
            Whether to save cleaned text
        
        Returns:
        --------
        pd.DataFrame : Results dataframe
        """
        if self.df is None:
            print("No CSV loaded. Run load_csv() first.")
            return None
        
        # Get unique indices
        indices = self.df[self.index_column].dropna().unique()
        
        if limit:
            indices = indices[:limit]
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(indices)} PDFs")
        print(f"{'='*60}\n")
        
        results = []
        successful = 0
        failed = 0
        
        for i, idx in enumerate(indices, 1):
            print(f"[{i}/{len(indices)}] Index: {idx}")
            
            result = self.process_single_pdf(
                str(idx),
                save_raw=save_raw,
                save_clean=save_clean
            )
            
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
                print(f"  ⚠ {result['error']}")
            
            print()  # Blank line between PDFs
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total PDFs:           {len(indices)}")
        print(f"Successful:           {successful}")
        print(f"Failed:              {failed}")
        print(f"Raw text directory:   {os.path.abspath(self.raw_text_dir)}")
        print(f"Clean text directory: {os.path.abspath(self.clean_text_dir)}")
        
        if failed > 0:
            print(f"\nFailed indices:")
            for r in results:
                if not r['success']:
                    print(f"  - {r['index']}: {r['error']}")
        
        return results_df
    
    def merge_with_csv(self, results_df):
        """
        Merge extracted text with original CSV data
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from process_all_pdfs()
        
        Returns:
        --------
        pd.DataFrame : Merged dataframe
        """
        if self.df is None:
            print("No CSV loaded. Run load_csv() first.")
            return None
        
        # Ensure index column is string type in both dataframes
        self.df[self.index_column] = self.df[self.index_column].astype(str)
        results_df['index'] = results_df['index'].astype(str)
        
        # Merge
        merged = self.df.merge(
            results_df[['index', 'clean_text', 'success']],
            left_on=self.index_column,
            right_on='index',
            how='left'
        )
        
        # Rename columns for clarity
        merged = merged.rename(columns={
            'clean_text': 'Full_Text_Cleaned',
            'success': 'Full_Text_Available'
        })
        
        # Drop duplicate index column
        if 'index' in merged.columns:
            merged = merged.drop('index', axis=1)
        
        return merged
    
    def get_customizable_stop_words(self):
        """
        Return the academic stop words as a list for customization
        """
        return sorted(list(self.academic_stop_words))
    
    def add_custom_stop_words(self, words):
        """
        Add additional stop words
        
        Parameters:
        -----------
        words : list or set
            Words to add to stop words
        """
        if isinstance(words, list):
            words = set(words)
        self.academic_stop_words.update(words)
        self.all_stop_words = self.stop_words.union(self.academic_stop_words)
        print(f"Added {len(words)} custom stop words")
    
    def remove_stop_words(self, words):
        """
        Remove words from stop words list
        
        Parameters:
        -----------
        words : list or set
            Words to remove from stop words
        """
        if isinstance(words, list):
            words = set(words)
        self.academic_stop_words.difference_update(words)
        self.all_stop_words = self.stop_words.union(self.academic_stop_words)
        print(f"Removed {len(words)} stop words")


def main():
    """
    Main function to run PDF text extraction
    """
    # ==================== CONFIGURATION ====================
    # Modify these parameters based on your setup
    CSV_PATH = 'srl-full-txt-review.csv'  # Your CSV file
    PDF_FOLDER = 'pdfs'                    # Folder containing PDFs
    INDEX_COLUMN = 'EID'                  # Column matching PDF filenames
    PDF_PATTERN = '{index}.pdf'           # Pattern for PDF filenames
    
    # Processing options
    PROCESS_LIMIT = None                  # Set to number for testing, None for all
    SAVE_RAW = True                       # Save raw extracted text
    SAVE_CLEAN = True                     # Save cleaned text
    MERGE_WITH_CSV = True                 # Merge results with original CSV
    # ======================================================
    
    # Initialize extractor
    extractor = PDFTextExtractor(
        csv_path=CSV_PATH,
        pdf_folder=PDF_FOLDER,
        index_column=INDEX_COLUMN,
        pdf_pattern=PDF_PATTERN
    )
    
    # Step 1: Load CSV
    print("\n" + "="*60)
    print("STEP 1: LOADING CSV")
    print("="*60)
    if not extractor.load_csv():
        print("Failed to load CSV. Exiting.")
        return
    
    # Step 2: Display current stop words
    print("\n" + "="*60)
    print("STEP 2: STOP WORDS CONFIGURATION")
    print("="*60)
    stop_words = extractor.get_customizable_stop_words()
    print(f"Total academic stop words: {len(stop_words)}")
    print("Sample stop words:", ", ".join(list(stop_words)[:20]), "...")
    
    # Optional: Add or remove stop words here
    # extractor.add_custom_stop_words(['additional', 'words'])
    # extractor.remove_stop_words(['study', 'research'])  # If too aggressive
    
    # Step 3: Process PDFs
    print("\n" + "="*60)
    print("STEP 3: EXTRACTING TEXT FROM PDFs")
    print("="*60)
    
    # Check if PDF folder exists
    if not os.path.exists(PDF_FOLDER):
        print(f"Error: PDF folder '{PDF_FOLDER}' not found!")
        print("Please create the folder and add your PDF files.")
        return
    
    # List PDFs in folder
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in '{PDF_FOLDER}'")
    
    if len(pdf_files) == 0:
        print("No PDF files found. Please add PDFs to the folder.")
        return
    
    # Process all PDFs
    results_df = extractor.process_all_pdfs(
        limit=PROCESS_LIMIT,
        save_raw=SAVE_RAW,
        save_clean=SAVE_CLEAN
    )
    
    if results_df is None:
        print("Processing failed.")
        return
    
    # Step 4: Merge with CSV
    if MERGE_WITH_CSV:
        print("\n" + "="*60)
        print("STEP 4: MERGING WITH CSV")
        print("="*60)
        
        merged_df = extractor.merge_with_csv(results_df)
        
        if merged_df is not None:
            # Save merged data
            output_path = 'topic_modeling_with_fulltext.csv'
            merged_df.to_csv(output_path, index=False)
            print(f"✓ Saved merged data to: {output_path}")
            
            # Statistics
            available = merged_df['Full_Text_Available'].sum()
            print(f"\nPapers with full text available: {available}/{len(merged_df)}")
            
            # Sample of available texts
            if available > 0:
                print("\nSample cleaned text (first 200 characters):")
                sample = merged_df[merged_df['Full_Text_Available']]['Full_Text_Cleaned'].iloc[0]
                print(f"  {sample[:200]}...")
    
    # Step 5: Final summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"\nOutput directories:")
    print(f"  Raw text:  {os.path.abspath(extractor.raw_text_dir)}")
    print(f"  Clean text: {os.path.abspath(extractor.clean_text_dir)}")
    print(f"  Merged CSV: topic_modeling_with_fulltext.csv")
    print(f"\nYou can now use the clean text files or the merged CSV")
    print(f"with your topic modeling script by changing TEXT_COLUMN to 'Full_Text_Cleaned'")


if __name__ == "__main__":
    main()
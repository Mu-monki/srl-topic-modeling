"""
PDF Text Extraction Script - Handles corrupted text layers
Uses multiple methods to extract readable text from problematic PDFs
"""

import os
import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# PDF text extraction - multiple methods
try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    print("Info: pdfminer.six not installed. Install with: pip install pdfminer.six")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYRPDF2_AVAILABLE = True
except ImportError:
    PYRPDF2_AVAILABLE = False

# OCR as last resort
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


class PDFTextExtractor:
    """
    Extracts text from PDFs using multiple methods to handle encoding issues
    """
    
    def __init__(self, csv_path, pdf_folder, index_column='DocumentID'):
        self.csv_path = csv_path
        self.pdf_folder = pdf_folder
        self.index_column = index_column
        self.df = None
        
        self.lemmatizer = WordNetLemmatizer()
        
        # Base stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Academic stop words
        self.academic_stop_words = {
            'abstract', 'introduction', 'background', 'methodology', 'methods',
            'materials', 'results', 'discussion', 'conclusion', 'conclusions',
            'acknowledgements', 'acknowledgments', 'references', 'appendix',
            'appendices', 'supplementary', 'supplement', 'figure', 'figures',
            'table', 'tables', 'section', 'sections',
            'study', 'research', 'paper', 'article', 'data', 'analysis',
            'findings', 'literature', 'review', 'et', 'al',
            'copyright', 'license', 'elsevier', 'springer', 'wiley', 'taylor',
            'francis', 'routledge', 'sage', 'ieee', 'doi', 'http', 'https',
            'www', 'pdf', 'page', 'pages', 'journal', 'volume', 'issue', 'cid'
        }
        
        self.all_stop_words = self.stop_words.union(self.academic_stop_words)
        
        # Create output directories
        self.raw_text_dir = 'extracted_text_raw'
        self.clean_text_dir = 'extracted_text_clean'
        os.makedirs(self.raw_text_dir, exist_ok=True)
        os.makedirs(self.clean_text_dir, exist_ok=True)
    
    def load_csv(self):
        """Load the CSV file"""
        print("Loading CSV...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(self.df)} records")
        
        if self.index_column not in self.df.columns:
            print(f"Error: Column '{self.index_column}' not found!")
            return False
        return True
    
    def is_text_garbled(self, text, threshold=0.3):
        """
        Check if extracted text appears to be garbled
        Returns True if text seems garbled
        """
        if not text or len(text) < 100:
            return True
        
        # Count English words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(words) == 0:
            return True
        
        # Check against common English words
        common_words = {'the', 'of', 'and', 'in', 'to', 'a', 'is', 'that', 'for',
                       'it', 'as', 'with', 'was', 'on', 'are', 'by', 'this', 'or',
                       'an', 'be', 'which', 'not', 'also', 'from', 'at', 'has', 'can',
                       'its', 'these', 'such', 'they', 'each', 'were', 'between', 'other'}
        
        # Count how many words look like real English
        english_count = 0
        for word in words:
            word_lower = word.lower()
            if word_lower in common_words or len(word_lower) <= 10:
                english_count += 1
        
        ratio = english_count / len(words) if words else 0
        
        # Also check for PDF structural garbage
        garbage_patterns = ['obj', 'endobj', 'stream', 'endstream', 'xref', 
                          'trailer', 'startxref', 'mediabox', 'cropbox']
        garbage_count = sum(1 for p in garbage_patterns if p in text.lower())
        
        return ratio < threshold or garbage_count > 3
    
    def extract_text_pdfminer(self, pdf_path):
        """Extract text using pdfminer - best for complex PDFs"""
        if not PDFMINER_AVAILABLE:
            return ""
        try:
            text = pdfminer_extract(pdf_path)
            if text and len(text.strip()) > 100:
                return text
        except:
            pass
        return ""
    
    def extract_text_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber"""
        if not PDFPLUMBER_AVAILABLE:
            return ""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip() and len(text.strip()) > 100:
                return text
        except:
            pass
        return ""
    
    def extract_text_pypdf2(self, pdf_path):
        """Extract text using PyPDF2"""
        if not PYRPDF2_AVAILABLE:
            return ""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip() and len(text.strip()) > 100:
                return text
        except:
            pass
        return ""
    
    def extract_text_ocr(self, pdf_path, max_pages=10):
        """Extract text using OCR (last resort, slower)"""
        if not OCR_AVAILABLE:
            return ""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
            
            text = ""
            for i, image in enumerate(images):
                # OCR the image
                page_text = pytesseract.image_to_string(image)
                if page_text:
                    text += page_text + "\n"
            
            if text.strip() and len(text.strip()) > 100:
                return text
        except:
            pass
        return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text using all available methods, returning the best result
        """
        print(f"  Extracting text...", end=" ")
        
        # Try methods in order of preference
        methods = [
            ("pdfminer", self.extract_text_pdfminer),
            ("pdfplumber", self.extract_text_pdfplumber),
            ("pypdf2", self.extract_text_pypdf2),
        ]
        
        for method_name, method_func in methods:
            text = method_func(pdf_path)
            if text and not self.is_text_garbled(text):
                print(f"✓ ({method_name})")
                return text
            elif text:
                print(f"⚠ {method_name} produced garbled text, trying next...", end=" ")
        
        # If all text methods fail, try OCR
        if OCR_AVAILABLE:
            print(f"\n  Trying OCR...", end=" ")
            text = self.extract_text_ocr(pdf_path)
            if text and not self.is_text_garbled(text):
                print(f"✓ (OCR)")
                return text
        
        print(f"✗ Failed")
        return ""
    
    def clean_text(self, text):
        """Clean and preprocess extracted text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove headers/footers (lines with page numbers)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove DOIs
        text = re.sub(r'doi\s*:\s*\S+', '', text)
        
        # Remove citation patterns
        text = re.sub(r'\[\d+(?:[,-]\d+)*\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)  # Year citations like (2020)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?\;\:\-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove short fragments
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
        text = '. '.join(sentences)
        
        if not text.strip():
            return ""
        
        # Tokenize and remove stop words
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(t) 
                     for t in tokens 
                     if t not in self.all_stop_words and len(t) > 2 and t.isalpha()]
            return ' '.join(tokens)
        except:
            # Fallback: simple word filtering
            words = text.split()
            words = [w for w in words if len(w) > 2 and w.isalpha() 
                    and w not in self.all_stop_words]
            return ' '.join(words)
    
    def process_single_pdf(self, index_value):
        """Process a single PDF"""
        safe_index = str(index_value).replace('/', '_').replace('\\', '_').replace(':', '_')
        pdf_filename = f"{safe_index}.pdf"
        pdf_path = os.path.join(self.pdf_folder, pdf_filename)
        
        result = {
            'index': index_value,
            'raw_text': '',
            'clean_text': '',
            'success': False,
            'chars_raw': 0,
            'chars_clean': 0
        }
        
        if not os.path.exists(pdf_path):
            result['error'] = f"PDF not found"
            return result
        
        file_size = os.path.getsize(pdf_path)
        if file_size < 1000:
            result['error'] = f"PDF too small ({file_size} bytes)"
            return result
        
        # Extract text using best available method
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            result['error'] = "No text extracted"
            return result
        
        result['raw_text'] = raw_text
        result['chars_raw'] = len(raw_text)
        
        # Save raw
        raw_path = os.path.join(self.raw_text_dir, f"{safe_index}_raw.txt")
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
        # Clean
        clean_text = self.clean_text(raw_text)
        
        if clean_text.strip() and len(clean_text) > 100:
            result['clean_text'] = clean_text
            result['chars_clean'] = len(clean_text)
            result['success'] = True
            
            clean_path = os.path.join(self.clean_text_dir, f"{safe_index}_clean.txt")
            with open(clean_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)
        
        return result
    
    def process_all_pdfs(self, limit=None):
        """Process all PDFs"""
        if self.df is None:
            return None
        
        indices = self.df[self.index_column].dropna().unique()
        if limit:
            indices = indices[:limit]
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(indices)} PDFs")
        print(f"{'='*60}\n")
        
        results = []
        successful = 0
        
        for i, idx in enumerate(indices, 1):
            safe_idx = str(idx).replace('/', '_')
            print(f"[{i}/{len(indices)}] {safe_idx}")
            
            result = self.process_single_pdf(str(idx))
            results.append(result)
            
            if result['success']:
                successful += 1
                print(f"  ✓ {result['chars_raw']:,} chars → {result['chars_clean']:,} chars (cleaned)")
            else:
                print(f"  ⚠ {result.get('error', 'Failed')}")
        
        results_df = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{len(indices)}")
        print(f"Raw text dir:  {os.path.abspath(self.raw_text_dir)}")
        print(f"Clean text dir: {os.path.abspath(self.clean_text_dir)}")
        
        return results_df
    
    def merge_with_csv(self, results_df):
        """Merge with original CSV"""
        self.df[self.index_column] = self.df[self.index_column].astype(str)
        results_df['index'] = results_df['index'].astype(str)
        
        merged = self.df.merge(
            results_df[['index', 'clean_text', 'success']],
            left_on=self.index_column,
            right_on='index',
            how='left'
        )
        
        merged = merged.rename(columns={
            'clean_text': 'Full_Text_Cleaned',
            'success': 'Full_Text_Available'
        })
        
        if 'index' in merged.columns:
            merged = merged.drop('index', axis=1)
        
        return merged


def main():
    # ==================== CONFIGURATION ====================
    CSV_PATH = 'srl-full-txt-review.csv'
    PDF_FOLDER = 'pdf-files'
    INDEX_COLUMN = 'DocumentID'
    
    # Set to 3 for quick test, None for all
    PROCESS_LIMIT = 0
    # ======================================================
    
    extractor = PDFTextExtractor(
        csv_path=CSV_PATH,
        pdf_folder=PDF_FOLDER,
        index_column=INDEX_COLUMN
    )
    
    # Load CSV
    print("="*60)
    print("PDF TEXT EXTRACTION WITH ENCODING FIX")
    print("="*60)
    
    if not extractor.load_csv():
        return
    
    # Show available methods
    print(f"\nAvailable extraction methods:")
    print(f"  pdfminer.six: {'✓' if PDFMINER_AVAILABLE else '✗ (install: pip install pdfminer.six)'}")
    print(f"  pdfplumber:  {'✓' if PDFPLUMBER_AVAILABLE else '✗'}")
    print(f"  PyPDF2:      {'✓' if PYRPDF2_AVAILABLE else '✗'}")
    print(f"  OCR (ocr):   {'✓' if OCR_AVAILABLE else '✗'}")
    
    # Process PDFs
    results_df = extractor.process_all_pdfs(limit=PROCESS_LIMIT)
    
    if results_df is None:
        return
    
    # Merge and save
    merged_df = extractor.merge_with_csv(results_df)
    if merged_df is not None:
        output_path = 'topic_modeling_with_fulltext.csv'
        merged_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved: {output_path}")
        
        # Show sample of successful extraction
        success_df = merged_df[merged_df['Full_Text_Available'] == True]
        if len(success_df) > 0:
            print(f"\nSample cleaned text preview:")
            sample = success_df['Full_Text_Cleaned'].iloc[0]
            print(f"  {sample[:300]}...")


if __name__ == "__main__":
    main()
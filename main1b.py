import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core libraries
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFSectionExtractor:
    """
    Main class for extracting and ranking PDF sections based on user persona and job requirements.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the PDF extractor with English sentence embedding model.
        
        Args:
            model_name: Sentence transformer model for English embeddings (~90MB)
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the English sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_sections_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract sections and paragraphs from PDF with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of section dictionaries with text, metadata, and structure info
        """
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")
            
            for page_num, page in enumerate(doc, 1):
                # Get text blocks with font information
                blocks = page.get_text("dict")
                
                current_section = {
                    'text': '',
                    'page': page_num,
                    'document': Path(pdf_path).name,
                    'font_size': 0,
                    'is_heading': False,
                    'position': 0
                }
                
                for block in blocks.get('blocks', []):
                    if 'lines' not in block:
                        continue
                        
                    for line in block['lines']:
                        line_text = ''
                        max_font_size = 0
                        
                        for span in line.get('spans', []):
                            text = span.get('text', '').strip()
                            font_size = span.get('size', 12)
                            
                            if text:
                                line_text += text + ' '
                                max_font_size = max(max_font_size, font_size)
                        
                        line_text = line_text.strip()
                        if not line_text:
                            continue
                        
                        # Determine if this is likely a heading based on font size and length
                        is_potential_heading = (
                            max_font_size > 14 or 
                            (len(line_text) < 100 and max_font_size > 12)
                        )
                        
                        # If we detect a new heading, save the previous section
                        if is_potential_heading and current_section['text'].strip():
                            current_section['text'] = current_section['text'].strip()
                            if len(current_section['text']) > 20:  # Minimum content threshold
                                sections.append(current_section.copy())
                            
                            # Start new section
                            current_section = {
                                'text': line_text,
                                'page': page_num,
                                'document': Path(pdf_path).name,
                                'font_size': max_font_size,
                                'is_heading': True,
                                'position': len(sections)
                            }
                        else:
                            # Add to current section
                            if current_section['text']:
                                current_section['text'] += ' ' + line_text
                            else:
                                current_section['text'] = line_text
                            
                            current_section['font_size'] = max(current_section['font_size'], max_font_size)
                
                # Don't forget the last section of the page
                if current_section['text'].strip() and len(current_section['text']) > 20:
                    sections.append(current_section.copy())
            
            doc.close()
            logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            
        return sections
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of the given text (simplified for English-only model).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Always returns 'en' for English since we're using English-only model
        """
        # Since we're using English-only model, always return 'en'
        # This simplifies the code and removes dependency on langdetect
        return 'en'
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate English embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array of embeddings
        """
        try:
            # Clean texts
            clean_texts = [text.replace('\n', ' ').strip() for text in texts]
            embeddings = self.model.encode(clean_texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def calculate_relevance_scores(self, sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """
        Calculate relevance scores for sections based on persona and job requirements.
        
        Args:
            sections: List of extracted sections
            persona: User persona description
            job_to_be_done: Job requirement description
            
        Returns:
            Sections with relevance scores added
        """
        if not sections:
            return []
        
        # Combine persona and job for query embedding
        query_text = f"{persona} {job_to_be_done}"
        
        # Extract texts for embedding
        section_texts = [section['text'] for section in sections]
        all_texts = [query_text] + section_texts
        
        # Generate embeddings
        embeddings = self.generate_embeddings(all_texts)
        if embeddings.size == 0:
            logger.error("Failed to generate embeddings")
            return sections
        
        query_embedding = embeddings[0:1]  # First embedding is the query
        section_embeddings = embeddings[1:]  # Rest are sections
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]
        
        # Add scores and additional metadata to sections
        scored_sections = []
        for i, section in enumerate(sections):
            section_copy = section.copy()
            
            # Detect language
            section_copy['language'] = self.detect_language(section['text'])
            
            # Base similarity score
            base_score = similarities[i]
            
            # Apply weighting factors
            # 1. Heading bonus
            heading_bonus = 0.1 if section.get('is_heading', False) else 0
            
            # 2. Length factor (prefer substantial content, but not too long)
            text_length = len(section['text'])
            if 100 <= text_length <= 2000:
                length_factor = 1.0
            elif text_length < 100:
                length_factor = 0.7
            else:
                length_factor = 0.9
            
            # 3. Position factor (slight preference for earlier sections)
            position_factor = max(0.8, 1.0 - (section.get('position', 0) * 0.02))
            
            # Calculate final score
            final_score = (base_score + heading_bonus) * length_factor * position_factor
            section_copy['relevance_score'] = float(final_score)
            section_copy['base_similarity'] = float(base_score)
            
            scored_sections.append(section_copy)
        
        return scored_sections
    
    def extract_subsections(self, top_sections: List[Dict], max_subsections: int = 3) -> List[Dict]:
        """
        Extract and refine subsections from top-ranked sections.
        
        Args:
            top_sections: Top-ranked sections to analyze
            max_subsections: Maximum number of subsections per section
            
        Returns:
            List of refined subsections
        """
        subsections = []
        
        for section in top_sections[:5]:  # Process top 5 sections
            text = section['text']
            sentences = text.split('. ')
            
            if len(sentences) <= 2:
                # Short section, use as-is
                subsections.append({
                    'document': section['document'],
                    'page': section['page'],
                    'refined_text': text[:500] + ('...' if len(text) > 500 else ''),
                    'language': section['language'],
                    'parent_score': section['relevance_score']
                })
            else:
                # Split into meaningful chunks
                chunk_size = max(2, len(sentences) // max_subsections)
                for i in range(0, len(sentences), chunk_size):
                    chunk = '. '.join(sentences[i:i + chunk_size])
                    if len(chunk.strip()) > 50:  # Minimum subsection length
                        subsections.append({
                            'document': section['document'],
                            'page': section['page'],
                            'refined_text': chunk[:500] + ('...' if len(chunk) > 500 else ''),
                            'language': section['language'],
                            'parent_score': section['relevance_score']
                        })
        
        return subsections
    
    def process_documents(self, pdf_paths: List[str], persona: str, job_to_be_done: str, 
                         max_sections: int = 10) -> Dict:
        """
        Main processing function to extract and rank sections from multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            persona: User persona description
            job_to_be_done: Job requirement description
            max_sections: Maximum number of sections to return
            
        Returns:
            Structured JSON output with ranked sections and subsections
        """
        start_time = time.time()
        logger.info(f"Starting processing of {len(pdf_paths)} documents")
        
        # Extract sections from all PDFs
        all_sections = []
        for pdf_path in pdf_paths:
            sections = self.extract_sections_from_pdf(pdf_path)
            all_sections.extend(sections)
        
        logger.info(f"Total sections extracted: {len(all_sections)}")
        
        # Calculate relevance scores
        scored_sections = self.calculate_relevance_scores(all_sections, persona, job_to_be_done)
        
        # Sort by relevance score
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Get top sections
        top_sections = scored_sections[:max_sections]
        
        # Extract subsections
        subsections = self.extract_subsections(top_sections)
        
        # Prepare output
        output = {
            'metadata': {
                'documents': [Path(path).name for path in pdf_paths],
                'persona': persona,
                'job_to_be_done': job_to_be_done,
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(time.time() - start_time, 2),
                'total_sections_found': len(all_sections),
                'model_used': self.model_name
            },
            'extracted_sections': [
                {
                    'document': section['document'],
                    'page': section['page'],
                    'section_title': section['text'][:100] + ('...' if len(section['text']) > 100 else ''),
                    'importance_rank': i + 1,
                    'language': section['language'],
                    'relevance_score': round(section['relevance_score'], 4),
                    'full_text': section['text'][:1000] + ('...' if len(section['text']) > 1000 else '')
                }
                for i, section in enumerate(top_sections)
            ],
            'subsection_analysis': subsections[:15]  # Limit subsections
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return output


def main():
    """
    Example usage and testing function.
    """
    # Initialize the extractor
    extractor = PDFSectionExtractor()
    
    # Example usage
    pdf_paths = [
        # Add your PDF paths here
        # "document1.pdf",
        # "document2.pdf"
    ]
    
    persona = "PhD Researcher in Computational Biology"
    job_to_be_done = "Create a literature review on Graph Neural Networks"
    
    # For demonstration, we'll show how to use the system
    print("=" * 60)
    print("INTELLIGENT PDF SECTION EXTRACTOR")
    print("=" * 60)
    print(f"Model: {extractor.model_name}")
    print(f"Persona: {persona}")
    print(f"Job to be done: {job_to_be_done}")
    print("=" * 60)
    
    # If no PDF paths provided, prompt user for input
    if not pdf_paths:
        print("\nNo PDF paths found in the code.")
        print("Please provide PDF file paths to process:")
        
        while True:
            pdf_path = input("\nEnter PDF file path (or 'done' to finish, 'quit' to exit): ").strip()
            
            if pdf_path.lower() in ['quit', 'exit']:
                print("Exiting...")
                return
            
            if pdf_path.lower() == 'done':
                if pdf_paths:
                    break
                else:
                    print("No PDF files added. Please add at least one file.")
                    continue
            
            # Remove quotes if user wrapped the path in quotes
            if pdf_path.startswith('"') and pdf_path.endswith('"'):
                pdf_path = pdf_path[1:-1]
            elif pdf_path.startswith("'") and pdf_path.endswith("'"):
                pdf_path = pdf_path[1:-1]
            
            if not pdf_path:
                print("Please enter a valid path.")
                continue
            
            # Check if file exists
            if not Path(pdf_path).exists():
                print(f"Error: File not found: {pdf_path}")
                print("Please check the path and try again.")
                continue
            
            # Check if it's a PDF file
            if not pdf_path.lower().endswith('.pdf'):
                print("Warning: File doesn't have .pdf extension. Continue anyway? (y/n): ", end="")
                if input().lower() not in ['y', 'yes']:
                    continue
            
            pdf_paths.append(pdf_path)
            print(f"Added: {Path(pdf_path).name}")
            print(f"Total files: {len(pdf_paths)}")
    
    if pdf_paths:
        print(f"\nProcessing {len(pdf_paths)} PDF file(s)...")
        
        # Process documents
        result = extractor.process_documents(pdf_paths, persona, job_to_be_done)
        
        # Save result
        output_file = f"extraction_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Processing time: {result['metadata']['processing_time_seconds']} seconds")
        print(f"Sections found: {result['metadata']['total_sections_found']}")
        print(f"Top sections returned: {len(result['extracted_sections'])}")
        
        # Show top results summary
        if result['extracted_sections']:
            print(f"\nTop 3 Most Relevant Sections:")
            print("-" * 50)
            for i, section in enumerate(result['extracted_sections'][:3], 1):
                print(f"{i}. {section['section_title']}")
                print(f"   Document: {section['document']} | Page: {section['page']}")
                print(f"   Relevance Score: {section['relevance_score']:.4f}")
                print()
        
    else:
        print("No PDF files to process. Exiting.")
        print("\nExample code structure ready for use!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for the new dynamic tokenization chunking system
"""

import re
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass


class TokenCounter:
    """Accurate token counting for different models"""
    
    # Token estimation ratios for different content types
    TOKEN_RATIOS = {
        'code': 2.5,      # Code has more tokens per character
        'text': 4.0,      # Regular text
        'markdown': 3.5,  # Markdown with formatting
        'html': 2.8,      # HTML with tags
    }
    
    @staticmethod
    def estimate_tokens(text: str, content_type: str = 'text') -> int:
        """
        Estimate token count for text based on content type.
        More accurate than simple character/4 ratio.
        """
        if not text:
            return 0
            
        # Clean text for better estimation
        cleaned_text = TokenCounter._clean_text_for_counting(text)
        
        # Base estimation
        char_count = len(cleaned_text)
        ratio = TokenCounter.TOKEN_RATIOS.get(content_type, 4.0)
        base_tokens = char_count / ratio
        
        # Adjust for content complexity
        complexity_factor = TokenCounter._calculate_complexity_factor(cleaned_text)
        
        return int(base_tokens * complexity_factor)
    
    @staticmethod
    def _clean_text_for_counting(text: str) -> str:
        """Clean text for more accurate token counting"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags (but keep content)
        text = re.sub(r'<[^>]+>', '', text)
        # Remove markdown formatting
        text = re.sub(r'[*_`#]+', '', text)
        return text.strip()
    
    @staticmethod
    def _calculate_complexity_factor(text: str) -> float:
        """Calculate complexity factor based on text characteristics"""
        factor = 1.0
        
        # More punctuation = more tokens
        punct_ratio = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
        if punct_ratio > 0.1:
            factor += 0.2
        
        # More numbers = more tokens
        num_ratio = len(re.findall(r'\d', text)) / max(len(text), 1)
        if num_ratio > 0.1:
            factor += 0.1
        
        # More special characters = more tokens
        special_ratio = len(re.findall(r'[^\w\s\.,!?;:]', text)) / max(len(text), 1)
        if special_ratio > 0.05:
            factor += 0.1
        
        return min(factor, 1.5)  # Cap at 1.5x
    
    @staticmethod
    def is_within_token_limit(text: str, max_tokens: int, content_type: str = 'text') -> bool:
        """Check if text is within token limit"""
        return TokenCounter.estimate_tokens(text, content_type) <= max_tokens


@dataclass
class Document:
    """Document representation"""
    doc_id: str
    url: str
    title: str
    content: str
    source_url: str
    scraped_at: str


class DocumentChunker:
    """Advanced document chunker with dynamic tokenization and smart boundaries"""
    
    def __init__(self, 
                 chunk_size: int = 100, 
                 overlap: int = 10, 
                 max_tokens: int = 15000,
                 min_chunk_size: int = 20,
                 content_type: str = 'text'):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_tokens = max_tokens
        self.min_chunk_size = min_chunk_size
        self.content_type = content_type
    
    def chunk_document(self, doc: Document) -> List[str]:
        """Chunk document with smart boundaries and token limits"""
        # Detect content type for better token estimation
        content_type = self._detect_content_type(doc.content)
        
        # Split into semantic units (paragraphs, sections, etc.)
        semantic_units = self._split_into_semantic_units(doc.content)
        
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_word_count = 0
        
        for unit in semantic_units:
            unit_words = len(unit.split())
            potential_chunk = ' '.join(current_chunk + [unit])
            
            # Check both word count and token limits
            estimated_tokens = TokenCounter.estimate_tokens(potential_chunk, content_type)
            
            # If adding this unit would exceed limits, finalize current chunk
            if (current_word_count + unit_words > self.chunk_size or 
                estimated_tokens > self.max_tokens) and current_chunk:
                
                # Validate and save current chunk
                chunk_text = ' '.join(current_chunk)
                if self._validate_chunk(chunk_text, content_type):
                    chunks.append(chunk_text)
                
                # Create overlap for next chunk
                current_chunk = self._create_overlap(current_chunk)
                current_word_count = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(unit)
            current_word_count += unit_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self._validate_chunk(chunk_text, content_type):
                chunks.append(chunk_text)
        
        return chunks
    
    def _detect_content_type(self, content: str) -> str:
        """Detect content type for better token estimation"""
        # Check for code patterns
        if re.search(r'```|def |class |import |from ', content):
            return 'code'
        # Check for HTML
        elif re.search(r'<[^>]+>', content):
            return 'html'
        # Check for markdown
        elif re.search(r'#{1,6}\s|[*_`]|\[.*\]\(.*\)', content):
            return 'markdown'
        else:
            return 'text'
    
    def _split_into_semantic_units(self, content: str) -> List[str]:
        """Split content into semantic units (paragraphs, sections, etc.)"""
        # First, try to split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        
        units = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is too long, split by sentences
            if len(paragraph.split()) > self.chunk_size * 2:
                sentences = self._split_sentences(paragraph)
                units.extend(sentences)
            else:
                units.append(paragraph)
        
        return units
    
    def _split_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting that handles various formats"""
        # Handle different sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up and filter
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_overlap(self, current_chunk: List[str]) -> List[str]:
        """Create overlap from current chunk for next chunk"""
        overlap_units = []
        overlap_word_count = 0
        
        # Take last few units for overlap
        for unit in reversed(current_chunk):
            unit_words = len(unit.split())
            if overlap_word_count + unit_words <= self.overlap:
                overlap_units.insert(0, unit)
                overlap_word_count += unit_words
            else:
                break
        
        return overlap_units
    
    def _validate_chunk(self, chunk: str, content_type: str) -> bool:
        """Validate that chunk meets requirements"""
        word_count = len(chunk.split())
        token_count = TokenCounter.estimate_tokens(chunk, content_type)
        
        # Check minimum size
        if word_count < self.min_chunk_size:
            return False
        
        # Check token limit
        if token_count > self.max_tokens:
            print(f"WARNING: Chunk exceeds token limit: {token_count} > {self.max_tokens}")
            return False
        
        return True


def test_chunking_with_large_document():
    """Test the chunker with a large document to ensure token limits are respected"""
    # Create a test document that simulates the Python glossary
    large_content = """
    # Python Glossary Test Document
    
    This is a test document that simulates a large documentation page like the Python glossary.
    It contains multiple sections with detailed explanations that would normally exceed token limits.
    
    ## Section 1: Basic Concepts
    
    Python is a high-level programming language known for its simplicity and readability. 
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    
    ## Section 2: Data Types
    
    Python has several built-in data types including integers, floats, strings, lists, tuples, dictionaries, and sets.
    Each data type has specific characteristics and methods that make them suitable for different use cases.
    
    ## Section 3: Control Structures
    
    Python provides various control structures including if statements, for loops, while loops, and try-except blocks.
    These structures allow developers to control the flow of execution in their programs.
    
    ## Section 4: Functions and Classes
    
    Functions in Python are defined using the def keyword and can accept parameters and return values.
    Classes are defined using the class keyword and support inheritance, encapsulation, and polymorphism.
    
    ## Section 5: Advanced Features
    
    Python includes many advanced features such as decorators, generators, context managers, and metaclasses.
    These features provide powerful tools for creating sophisticated and maintainable code.
    
    ## Section 6: Standard Library
    
    Python comes with a comprehensive standard library that includes modules for file I/O, networking, 
    data processing, web development, and many other common programming tasks.
    
    ## Section 7: Third-Party Packages
    
    The Python ecosystem includes thousands of third-party packages available through PyPI.
    Popular packages include NumPy for numerical computing, Django for web development, and Pandas for data analysis.
    
    ## Section 8: Best Practices
    
    Python development follows several best practices including PEP 8 style guidelines, 
    proper documentation with docstrings, and comprehensive testing with frameworks like pytest.
    
    ## Section 9: Performance Considerations
    
    While Python is not the fastest language, there are many ways to optimize performance including
    using NumPy for numerical operations, Cython for compiled extensions, and profiling tools for optimization.
    
    ## Section 10: Conclusion
    
    Python is a versatile and powerful programming language that continues to grow in popularity.
    Its simplicity, readability, and extensive ecosystem make it an excellent choice for many programming tasks.
    """ * 10  # Repeat content to make it very large
    
    # Create test document
    test_doc = Document(
        doc_id="test_large_doc",
        url="https://example.com/large-doc",
        title="Large Test Document",
        content=large_content,
        source_url="https://example.com/large-doc",
        scraped_at=datetime.now().isoformat()
    )
    
    # Test chunker
    chunker = DocumentChunker(
        chunk_size=100,
        overlap=10,
        max_tokens=15000,
        min_chunk_size=20
    )
    
    chunks = chunker.chunk_document(test_doc)
    
    print(f"Created {len(chunks)} chunks from large document")
    
    # Validate all chunks
    all_valid = True
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        token_count = TokenCounter.estimate_tokens(chunk, 'text')
        
        print(f"Chunk {i+1}: {word_count} words, ~{token_count} tokens")
        
        if token_count > 15000:
            print(f"  WARNING: Chunk {i+1} exceeds token limit!")
            all_valid = False
    
    if all_valid:
        print("✅ All chunks are within token limits!")
    else:
        print("❌ Some chunks exceed token limits!")
    
    return all_valid


if __name__ == "__main__":
    # Test the chunking with large documents
    print("Testing chunking with large documents...")
    test_chunking_with_large_document()

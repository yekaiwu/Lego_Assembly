"""
Manual Input Handler: Processes LEGO instruction manuals (PDF/images).
Handles page extraction, step segmentation, and batch processing.
"""

# PyMuPDF import (required for PDF processing)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None  # Prevent NameError
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
from loguru import logger

class ManualInputHandler:
    """Handles LEGO instruction manual input processing."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./temp_manual_pages")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_manual(self, input_path: Union[str, Path]) -> List[str]:
        """
        Process a LEGO instruction manual (PDF or images).
        
        Args:
            input_path: Path to PDF file or directory containing images
        
        Returns:
            List of paths to extracted step images
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                return self._process_pdf(input_path)
            elif input_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Single image file
                return [str(input_path)]
        elif input_path.is_dir():
            return self._process_image_directory(input_path)
        
        raise ValueError(f"Invalid input path: {input_path}")
    
    def _process_pdf(self, pdf_path: Path) -> List[str]:
        """
        Extract pages from PDF and convert to images.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of paths to extracted page images
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Try PyMuPDF first if available (faster, better quality)
        if HAS_PYMUPDF:
            try:
                doc = fitz.open(pdf_path)
                page_paths = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for quality
                    
                    output_path = self.output_dir / f"page_{page_num + 1:03d}.png"
                    pix.save(str(output_path))
                    page_paths.append(str(output_path))
                    
                    logger.debug(f"Extracted page {page_num + 1}/{len(doc)}")
                
                doc.close()
                logger.info(f"Extracted {len(page_paths)} pages from PDF using PyMuPDF")
                return page_paths
            
            except Exception as e:
                logger.error(f"PyMuPDF failed: {e}")
                raise RuntimeError(f"PDF extraction failed: {e}. PyMuPDF is required for PDF processing.")
        else:
            raise RuntimeError("PyMuPDF is not installed. Please install it with: pip install PyMuPDF")
    
    def _process_image_directory(self, dir_path: Path) -> List[str]:
        """
        Process a directory containing instruction images.
        
        Args:
            dir_path: Path to directory with images
        
        Returns:
            Sorted list of image paths
        """
        logger.info(f"Processing image directory: {dir_path}")
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in sorted(dir_path.iterdir())
            if p.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_paths)} images")
        return image_paths
    
    def detect_step_boundaries(self, image_paths: List[str]) -> List[List[str]]:
        """
        Detect step boundaries in manual pages.
        Groups pages that belong to the same step.
        
        Args:
            image_paths: List of page image paths
        
        Returns:
            List of step groups, each containing one or more page paths
        """
        logger.info("Detecting step boundaries...")
        
        # Simple heuristic: one page = one step (can be improved)
        # TODO: Implement more sophisticated step detection using:
        # - Step number detection via OCR
        # - Visual similarity between pages
        # - Layout analysis to detect multi-page steps
        
        step_groups = [[img] for img in image_paths]
        
        logger.info(f"Detected {len(step_groups)} steps")
        return step_groups
    
    def segment_multi_step_page(self, image_path: str) -> List[str]:
        """
        Segment a page containing multiple steps into individual step images.
        
        Args:
            image_path: Path to page image
        
        Returns:
            List of paths to segmented step images
        """
        logger.info(f"Segmenting multi-step page: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return [image_path]
        
        # Simple segmentation: detect horizontal separators
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 2, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect line positions
        line_positions = []
        for i, row in enumerate(horizontal_lines):
            if np.sum(row) > img.shape[1] * 0.3 * 255:  # Threshold
                line_positions.append(i)
        
        # If no clear separators found, return original image
        if len(line_positions) < 2:
            logger.debug("No clear step separators detected, returning original page")
            return [image_path]
        
        # Split image at detected lines
        segmented_paths = []
        prev_y = 0
        
        for i, y in enumerate(line_positions + [img.shape[0]]):
            if y - prev_y > img.shape[0] * 0.1:  # Minimum segment height
                segment = img[prev_y:y, :]
                output_path = self.output_dir / f"{Path(image_path).stem}_segment_{i + 1}.png"
                cv2.imwrite(str(output_path), segment)
                segmented_paths.append(str(output_path))
                prev_y = y
        
        logger.info(f"Segmented into {len(segmented_paths)} sub-steps")
        return segmented_paths if segmented_paths else [image_path]
    
    def preprocess_image(self, image_path: str, enhance: bool = True) -> str:
        """
        Preprocess image for better VLM recognition.
        
        Args:
            image_path: Path to image
            enhance: Whether to apply enhancement (contrast, sharpness)
        
        Returns:
            Path to preprocessed image
        """
        if not enhance:
            return image_path
        
        logger.debug(f"Preprocessing image: {image_path}")
        
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)
        
        # Save preprocessed image
        output_path = self.output_dir / f"preprocessed_{Path(image_path).name}"
        img.save(output_path)
        
        return str(output_path)
    
    def batch_process_manuals(self, input_paths: List[Union[str, Path]]) -> List[List[str]]:
        """
        Process multiple manuals in batch.
        
        Args:
            input_paths: List of paths to PDF files or image directories
        
        Returns:
            List of page path lists for each manual
        """
        logger.info(f"Batch processing {len(input_paths)} manuals...")
        
        all_page_paths = []
        for input_path in input_paths:
            try:
                page_paths = self.process_manual(input_path)
                all_page_paths.append(page_paths)
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                all_page_paths.append([])
        
        logger.info(f"Batch processing complete. Processed {len(all_page_paths)} manuals")
        return all_page_paths


"""
Part Database: Manages LEGO part library with Rebrickable API integration.
Provides part matching, lookup, and local caching capabilities.
"""

import sqlite3
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from difflib import SequenceMatcher

from ..utils.config import get_config

class PartDatabase:
    """LEGO part database with Rebrickable API and local caching."""
    
    def __init__(self, db_path: Optional[Path] = None):
        config = get_config()
        self.db_path = db_path or config.paths.parts_db_path
        self.api_key = config.api.rebrickable_api_key
        self.api_base = "https://rebrickable.com/api/v3/lego"
        
        self._init_database()
        logger.info(f"Part database initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Parts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parts (
                part_num TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                part_cat_id INTEGER,
                part_material TEXT,
                year_from INTEGER,
                year_to INTEGER
            )
        ''')
        
        # Part categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS part_categories (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        ''')
        
        # Colors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS colors (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                rgb TEXT NOT NULL,
                is_trans BOOLEAN
            )
        ''')
        
        # Part-Color combinations (inventory)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS part_colors (
                part_num TEXT NOT NULL,
                color_id INTEGER NOT NULL,
                num_sets INTEGER,
                PRIMARY KEY (part_num, color_id),
                FOREIGN KEY (part_num) REFERENCES parts(part_num),
                FOREIGN KEY (color_id) REFERENCES colors(id)
            )
        ''')
        
        # Part dimensions cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS part_dimensions (
                part_num TEXT PRIMARY KEY,
                width_studs REAL,
                height_studs REAL,
                depth_studs REAL,
                FOREIGN KEY (part_num) REFERENCES parts(part_num)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.debug("Database schema initialized")
    
    def fetch_colors_from_api(self):
        """Fetch color data from Rebrickable API and cache locally."""
        if not self.api_key:
            logger.warning("No Rebrickable API key configured")
            return
        
        logger.info("Fetching colors from Rebrickable API...")
        
        headers = {"Authorization": f"key {self.api_key}"}
        url = f"{self.api_base}/colors/"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            colors = data.get("results", [])
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for color in colors:
                cursor.execute('''
                    INSERT OR REPLACE INTO colors (id, name, rgb, is_trans)
                    VALUES (?, ?, ?, ?)
                ''', (
                    color["id"],
                    color["name"],
                    color["rgb"],
                    color["is_trans"]
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Cached {len(colors)} colors")
        
        except Exception as e:
            logger.error(f"Failed to fetch colors: {e}")
    
    def fetch_part_from_api(self, part_num: str) -> Optional[Dict[str, Any]]:
        """Fetch part details from Rebrickable API."""
        if not self.api_key:
            return None
        
        headers = {"Authorization": f"key {self.api_key}"}
        url = f"{self.api_base}/parts/{part_num}/"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch part {part_num}: {e}")
            return None
    
    def search_parts_by_name(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search parts by name/description.
        
        Args:
            name: Part name or description
            limit: Maximum results to return
        
        Returns:
            List of matching parts
        """
        # First check local cache
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT part_num, name, part_cat_id
            FROM parts
            WHERE name LIKE ?
            LIMIT ?
        ''', (f"%{name}%", limit))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return [
                {"part_num": r[0], "name": r[1], "category_id": r[2]}
                for r in results
            ]
        
        # If not in cache, try API
        if self.api_key:
            headers = {"Authorization": f"key {self.api_key}"}
            url = f"{self.api_base}/parts/"
            params = {"search": name, "page_size": limit}
            
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Cache results
                self._cache_parts(data.get("results", []))
                
                return data.get("results", [])
            except Exception as e:
                logger.error(f"API search failed: {e}")
        
        return []
    
    def match_part_description(
        self, 
        description: str, 
        color: str,
        shape: str
    ) -> Optional[Dict[str, Any]]:
        """
        Match a VLM-extracted part description to a database part.
        
        Args:
            description: Full part description from VLM
            color: Color name
            shape: Shape/dimensions (e.g., "2x4 brick")
        
        Returns:
            Best matching part or None
        """
        logger.debug(f"Matching part: {description}, color: {color}, shape: {shape}")
        
        # Parse shape for dimensions
        dimensions = self._parse_dimensions(shape)
        
        # Search by shape/type
        search_terms = [shape, description]
        candidates = []
        
        for term in search_terms:
            results = self.search_parts_by_name(term, limit=20)
            candidates.extend(results)
        
        if not candidates:
            logger.warning(f"No candidates found for: {description}")
            return None
        
        # Score candidates
        scored = []
        for candidate in candidates:
            score = self._calculate_match_score(
                candidate, description, color, shape, dimensions
            )
            scored.append((score, candidate))
        
        # Return best match
        scored.sort(reverse=True, key=lambda x: x[0])
        best_match = scored[0][1] if scored[0][0] > 0.5 else None
        
        if best_match:
            logger.debug(f"Best match: {best_match['part_num']} - {best_match['name']}")
        else:
            logger.warning(f"No good match found for: {description}")
        
        return best_match
    
    def _parse_dimensions(self, shape: str) -> Optional[Tuple[int, int]]:
        """Parse dimensions from shape string (e.g., '2x4' -> (2, 4))."""
        import re
        match = re.search(r'(\d+)\s*x\s*(\d+)', shape.lower())
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None
    
    def _calculate_match_score(
        self,
        candidate: Dict[str, Any],
        description: str,
        color: str,
        shape: str,
        dimensions: Optional[Tuple[int, int]]
    ) -> float:
        """Calculate similarity score between candidate and description."""
        score = 0.0
        
        # Name similarity
        candidate_name = candidate.get("name", "").lower()
        name_sim = SequenceMatcher(None, candidate_name, description.lower()).ratio()
        score += name_sim * 0.5
        
        # Shape similarity
        shape_sim = SequenceMatcher(None, candidate_name, shape.lower()).ratio()
        score += shape_sim * 0.3
        
        # Dimension matching (if available)
        if dimensions:
            # TODO: Check actual dimensions from part_dimensions table
            pass
        
        # Color availability (bonus)
        # TODO: Check if color exists for this part in part_colors
        
        return score
    
    def _cache_parts(self, parts: List[Dict[str, Any]]):
        """Cache parts to local database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for part in parts:
            cursor.execute('''
                INSERT OR REPLACE INTO parts (part_num, name, part_cat_id)
                VALUES (?, ?, ?)
            ''', (
                part["part_num"],
                part["name"],
                part.get("part_cat_id")
            ))
        
        conn.commit()
        conn.close()
    
    def get_part_colors(self, part_num: str) -> List[Dict[str, Any]]:
        """Get available colors for a part."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.name, c.rgb, c.is_trans
            FROM colors c
            JOIN part_colors pc ON c.id = pc.color_id
            WHERE pc.part_num = ?
        ''', (part_num,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {"id": r[0], "name": r[1], "rgb": r[2], "is_trans": bool(r[3])}
            for r in results
        ]
    
    def get_part_dimensions(self, part_num: str) -> Optional[Dict[str, float]]:
        """Get cached dimensions for a part (in studs)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT width_studs, height_studs, depth_studs
            FROM part_dimensions
            WHERE part_num = ?
        ''', (part_num,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "width": result[0],
                "height": result[1],
                "depth": result[2]
            }
        return None
    
    def add_part_dimensions(self, part_num: str, width: float, height: float, depth: float):
        """Add/update part dimensions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO part_dimensions (part_num, width_studs, height_studs, depth_studs)
            VALUES (?, ?, ?, ?)
        ''', (part_num, width, height, depth))
        
        conn.commit()
        conn.close()


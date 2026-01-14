"""
Spatial Reasoning Engine: Determines 3D coordinates, orientations, and
connection points for LEGO parts in assembly.

Note: This module requires spatial_relationships data from VLM extraction.
If spatial relationships are disabled via --no-spatial-relationships flag,
methods in this module will receive empty/missing spatial data and return
default positions.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

class SpatialReasoning:
    """
    Handles 3D spatial calculations for LEGO assembly.

    This module processes spatial_relationships data extracted by the VLM.
    When spatial relationships are disabled, methods will gracefully handle
    missing data by returning default values.
    """
    
    # LEGO stud dimensions (1 stud = 8mm)
    STUD_SIZE_MM = 8.0
    PLATE_HEIGHT_MM = 3.2
    BRICK_HEIGHT_MM = 9.6
    
    def __init__(self, origin: Tuple[float, float, float] = (0, 0, 0)):
        """
        Initialize spatial reasoning engine.
        
        Args:
            origin: Origin point for coordinate system (x, y, z)
        """
        self.origin = np.array(origin)
        self.coordinate_system = "stud"  # Use stud-based coordinates
        logger.info(f"Spatial reasoning initialized with origin at {origin}")
    
    def calculate_position(
        self,
        target_info: Dict[str, Any],
        reference_parts: List[Dict[str, Any]],
        spatial_relationship: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate 3D position for a part based on spatial relationships.
        
        Args:
            target_info: Information about part to place
            reference_parts: Already-placed parts for reference
            spatial_relationship: Spatial relationship description from VLM
        
        Returns:
            Dictionary with x, y, z coordinates in studs
        """
        position_desc = (spatial_relationship.get("position") or "").lower()
        
        # If no reference parts, place at origin
        if not reference_parts:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Get reference part (usually the most recent)
        reference = reference_parts[-1]
        ref_pos = reference.get("position", {"x": 0, "y": 0, "z": 0})
        ref_dims = reference.get("dimensions", {"width": 1, "height": 1, "depth": 1})
        
        # Calculate offset based on position description
        offset = self._calculate_offset(position_desc, ref_dims)
        
        position = {
            "x": ref_pos["x"] + offset[0],
            "y": ref_pos["y"] + offset[1],
            "z": ref_pos["z"] + offset[2]
        }
        
        logger.debug(f"Calculated position: {position}")
        return position
    
    def _calculate_offset(
        self, 
        position_desc: str, 
        reference_dims: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """Calculate position offset based on description."""
        offset = [0.0, 0.0, 0.0]  # x, y, z
        
        # Vertical positions
        if "top" in position_desc or "above" in position_desc:
            offset[2] = reference_dims.get("height", 1)
        elif "bottom" in position_desc or "below" in position_desc:
            offset[2] = -reference_dims.get("height", 1)
        
        # Horizontal positions
        if "right" in position_desc:
            offset[0] = reference_dims.get("width", 1)
        elif "left" in position_desc:
            offset[0] = -reference_dims.get("width", 1)
        
        if "front" in position_desc:
            offset[1] = reference_dims.get("depth", 1)
        elif "back" in position_desc:
            offset[1] = -reference_dims.get("depth", 1)
        
        return tuple(offset)
    
    def calculate_orientation(
        self,
        target_info: Dict[str, Any],
        spatial_relationship: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate rotation angles for a part.
        
        Args:
            target_info: Part information
            spatial_relationship: Spatial relationship with rotation info
        
        Returns:
            Dictionary with x, y, z rotation angles in degrees
        """
        rotation_desc = (spatial_relationship.get("rotation") or "").lower()
        
        # Default: no rotation
        rotation = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Parse rotation description
        if "90" in rotation_desc or "quarter" in rotation_desc:
            if "clockwise" in rotation_desc or "right" in rotation_desc:
                rotation["z"] = 90.0
            elif "counterclockwise" in rotation_desc or "left" in rotation_desc:
                rotation["z"] = -90.0
        
        if "180" in rotation_desc or "half" in rotation_desc:
            rotation["z"] = 180.0
        
        if "upside" in rotation_desc or "flip" in rotation_desc:
            rotation["x"] = 180.0
        
        logger.debug(f"Calculated rotation: {rotation}")
        return rotation
    
    def determine_connection_points(
        self,
        part_info: Dict[str, Any],
        position: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Determine connection points (studs/tubes) for a part.
        
        Args:
            part_info: Part information including dimensions
            position: Part position
        
        Returns:
            List of connection point coordinates
        """
        dimensions = part_info.get("dimensions", {"width": 1, "depth": 1})
        width = dimensions.get("width", 1)
        depth = dimensions.get("depth", 1)
        
        connection_points = []
        
        # Top surface studs
        for x in range(int(width)):
            for y in range(int(depth)):
                point = {
                    "type": "stud",
                    "position": {
                        "x": position["x"] + x,
                        "y": position["y"] + y,
                        "z": position["z"] + dimensions.get("height", 1)
                    },
                    "available": True
                }
                connection_points.append(point)
        
        # Bottom surface tubes (inverse of studs)
        for x in range(int(width)):
            for y in range(int(depth)):
                point = {
                    "type": "tube",
                    "position": {
                        "x": position["x"] + x,
                        "y": position["y"] + y,
                        "z": position["z"]
                    },
                    "available": True
                }
                connection_points.append(point)
        
        return connection_points
    
    def validate_connection(
        self,
        part1_connections: List[Dict[str, Any]],
        part2_connections: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate that two parts can connect.
        
        Args:
            part1_connections: Connection points of first part
            part2_connections: Connection points of second part
        
        Returns:
            True if parts can connect
        """
        # Check for overlapping stud-tube pairs
        for conn1 in part1_connections:
            for conn2 in part2_connections:
                # Stud from part1 should align with tube from part2 (or vice versa)
                if self._connections_match(conn1, conn2):
                    return True
        
        return False
    
    def _connections_match(
        self, 
        conn1: Dict[str, Any], 
        conn2: Dict[str, Any]
    ) -> bool:
        """Check if two connection points match."""
        # One must be stud, other must be tube
        if conn1["type"] == conn2["type"]:
            return False
        
        # Positions must align
        pos1 = conn1["position"]
        pos2 = conn2["position"]
        
        tolerance = 0.1  # Small tolerance for floating point comparison
        
        return (
            abs(pos1["x"] - pos2["x"]) < tolerance and
            abs(pos1["y"] - pos2["y"]) < tolerance and
            abs(pos1["z"] - pos2["z"]) < tolerance
        )
    
    def calculate_bounding_box(
        self,
        parts: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate bounding box for a collection of parts.
        
        Args:
            parts: List of parts with positions and dimensions
        
        Returns:
            Bounding box with min and max coordinates
        """
        if not parts:
            return {"min": {"x": 0, "y": 0, "z": 0}, "max": {"x": 0, "y": 0, "z": 0}}
        
        min_coords = {"x": float('inf'), "y": float('inf'), "z": float('inf')}
        max_coords = {"x": float('-inf'), "y": float('-inf'), "z": float('-inf')}
        
        for part in parts:
            pos = part.get("position", {"x": 0, "y": 0, "z": 0})
            dims = part.get("dimensions", {"width": 1, "height": 1, "depth": 1})
            
            # Update min/max
            min_coords["x"] = min(min_coords["x"], pos["x"])
            min_coords["y"] = min(min_coords["y"], pos["y"])
            min_coords["z"] = min(min_coords["z"], pos["z"])
            
            max_coords["x"] = max(max_coords["x"], pos["x"] + dims.get("width", 1))
            max_coords["y"] = max(max_coords["y"], pos["y"] + dims.get("depth", 1))
            max_coords["z"] = max(max_coords["z"], pos["z"] + dims.get("height", 1))
        
        return {"min": min_coords, "max": max_coords}
    
    def check_collision(
        self,
        new_part: Dict[str, Any],
        existing_parts: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if a new part collides with existing parts.
        
        Args:
            new_part: Part to check
            existing_parts: Already-placed parts
        
        Returns:
            True if collision detected
        """
        new_pos = new_part.get("position", {"x": 0, "y": 0, "z": 0})
        new_dims = new_part.get("dimensions", {"width": 1, "height": 1, "depth": 1})
        
        for existing in existing_parts:
            if self._parts_overlap(new_part, existing):
                logger.warning("Collision detected!")
                return True
        
        return False
    
    def _parts_overlap(
        self,
        part1: Dict[str, Any],
        part2: Dict[str, Any]
    ) -> bool:
        """Check if two parts overlap in 3D space."""
        pos1 = part1.get("position", {"x": 0, "y": 0, "z": 0})
        dims1 = part1.get("dimensions", {"width": 1, "height": 1, "depth": 1})
        
        pos2 = part2.get("position", {"x": 0, "y": 0, "z": 0})
        dims2 = part2.get("dimensions", {"width": 1, "height": 1, "depth": 1})
        
        # Check overlap in each axis
        x_overlap = (pos1["x"] < pos2["x"] + dims2.get("width", 1) and
                     pos1["x"] + dims1.get("width", 1) > pos2["x"])
        
        y_overlap = (pos1["y"] < pos2["y"] + dims2.get("depth", 1) and
                     pos1["y"] + dims1.get("depth", 1) > pos2["y"])
        
        z_overlap = (pos1["z"] < pos2["z"] + dims2.get("height", 1) and
                     pos1["z"] + dims1.get("height", 1) > pos2["z"])
        
        return x_overlap and y_overlap and z_overlap


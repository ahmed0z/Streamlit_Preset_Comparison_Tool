"""
Data processing utilities for cleaning and normalizing input data
"""
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import config

class DataProcessor:
    """Handles data cleaning and normalization for comparison"""
    
    # Common unit mappings and variations
    UNIT_MAPPINGS = {
        # Length units
        'mm': ['mm', 'millimeter', 'millimeters', 'milimeter'],
        'cm': ['cm', 'centimeter', 'centimeters', 'centimetre'],
        'inches': ['inch', 'inches', '"', 'in', "'"],
        'feet': ['foot', 'feet', 'ft', "'"],
        'm': ['m', 'meter', 'meters', 'metre'],
        
        # Weight units
        'kg': ['kg', 'kilogram', 'kilograms', 'kilo'],
        'g': ['g', 'gram', 'grams'],
        'lb': ['lb', 'lbs', 'pound', 'pounds'],
        'oz': ['oz', 'ounce', 'ounces'],
        
        # Temperature units
        '°C': ['°C', 'celsius', 'centigrade', '°c', 'c'],
        '°F': ['°F', 'fahrenheit', '°f', 'f'],
        'K': ['K', 'kelvin'],
        
        # Pressure units
        'psi': ['psi', 'PSI', 'pounds per square inch'],
        'bar': ['bar', 'bars'],
        'Pa': ['Pa', 'pascal', 'pascals'],
        'kPa': ['kPa', 'kilopascal', 'kilopascals'],
        
        # Volume units
        'L': ['L', 'l', 'liter', 'liters', 'litre', 'litres'],
        'mL': ['mL', 'ml', 'milliliter', 'milliliters'],
        'gal': ['gal', 'gallon', 'gallons']
    }
    
    # Condition keywords
    CONDITION_KEYWORDS = ['@', 'at', 'under', 'over', 'above', 'below', 'during', 'when']
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison
        
        Args:
            text: Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize case for comparison (but preserve original formatting)
        return text
    
    @staticmethod
    def extract_value_components(value_str: str) -> Dict[str, Any]:
        """
        Extract components from a value string
        
        Args:
            value_str: String containing value, units, conditions, etc.
            
        Returns:
            Dict containing extracted components
        """
        if pd.isna(value_str) or not value_str:
            return {
                'original': '',
                'cleaned': '',
                'numbers': [],
                'units': [],
                'conditions': [],
                'has_multiple_values': False,
                'normalized_units': []
            }
        
        value_str = str(value_str).strip()
        
        # Extract numbers (including decimals)
        number_pattern = r'-?\d+\.?\d*'
        numbers = re.findall(number_pattern, value_str)
        numbers = [float(n) if '.' in n else int(n) for n in numbers]
        
        # Extract units
        units = DataProcessor._extract_units(value_str)
        normalized_units = [DataProcessor._normalize_unit(unit) for unit in units]
        
        # Extract conditions (text after @ symbol or condition keywords)
        conditions = DataProcessor._extract_conditions(value_str)
        
        # Check for multiple values (comma separated)
        has_multiple_values = ',' in value_str or ' and ' in value_str.lower()
        
        # Create cleaned version (alphanumeric only for fuzzy matching)
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', value_str)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        
        return {
            'original': value_str,
            'cleaned': cleaned,
            'numbers': numbers,
            'units': units,
            'conditions': conditions,
            'has_multiple_values': has_multiple_values,
            'normalized_units': normalized_units
        }
    
    @staticmethod
    def _extract_units(text: str) -> List[str]:
        """Extract unit strings from text"""
        units = []
        
        # Common unit patterns
        unit_patterns = [
            r'\b\d+\.?\d*\s*([a-zA-Z°"\']+)',  # Number followed by unit
            r'\(([a-zA-Z°"\']+)\)',            # Unit in parentheses
            r'\b(mm|cm|kg|lb|°C|°F|psi|bar|L|mL|gal|inch|inches|feet|ft)\b'  # Common units
        ]
        
        for pattern in unit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            units.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_units = []
        for unit in units:
            unit_lower = unit.lower()
            if unit_lower not in seen:
                seen.add(unit_lower)
                unique_units.append(unit)
        
        return unique_units
    
    @staticmethod
    def _normalize_unit(unit: str) -> str:
        """Normalize unit to standard form"""
        unit_lower = unit.lower().strip()
        
        for standard_unit, variations in DataProcessor.UNIT_MAPPINGS.items():
            if unit_lower in [v.lower() for v in variations]:
                return standard_unit
        
        return unit  # Return original if no mapping found
    
    @staticmethod
    def _extract_conditions(text: str) -> List[str]:
        """Extract condition information from text"""
        conditions = []
        
        # Split by @ symbol
        if '@' in text:
            parts = text.split('@')
            if len(parts) > 1:
                conditions.append(parts[1].strip())
        
        # Look for condition keywords
        for keyword in DataProcessor.CONDITION_KEYWORDS:
            pattern = rf'{keyword}\s+([^,\n]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend([match.strip() for match in matches])
        
        return list(set(conditions))  # Remove duplicates
    
    @staticmethod
    def prepare_for_matching(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
        """
        Prepare dataframe for matching by extracting value components
        
        Args:
            df: Input dataframe
            value_column: Name of column containing values to process
            
        Returns:
            pd.DataFrame: Enhanced dataframe with extracted components
        """
        df = df.copy()
        
        # Extract components for each value
        components = df[value_column].apply(DataProcessor.extract_value_components)
        
        # Add component columns
        df['original_value'] = components.apply(lambda x: x['original'])
        df['cleaned_value'] = components.apply(lambda x: x['cleaned'])
        df['numbers'] = components.apply(lambda x: x['numbers'])
        df['units'] = components.apply(lambda x: x['units'])
        df['conditions'] = components.apply(lambda x: x['conditions'])
        df['has_multiple_values'] = components.apply(lambda x: x['has_multiple_values'])
        df['normalized_units'] = components.apply(lambda x: x['normalized_units'])
        
        return df
    
    @staticmethod
    def create_search_variants(value: str) -> List[str]:
        """
        Create different variants of a value for searching
        
        Args:
            value: Original value string
            
        Returns:
            List[str]: List of value variants
        """
        variants = []
        
        if not value or pd.isna(value):
            return variants
        
        value = str(value).strip()
        variants.append(value)  # Original
        
        # Lowercase variant
        variants.append(value.lower())
        
        # Remove punctuation variant
        cleaned = re.sub(r'[^\w\s]', ' ', value)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned and cleaned not in variants:
            variants.append(cleaned)
            variants.append(cleaned.lower())
        
        # Remove numbers variant (for text matching)
        no_numbers = re.sub(r'\d+\.?\d*', '', value)
        no_numbers = re.sub(r'\s+', ' ', no_numbers).strip()
        if no_numbers and no_numbers not in variants:
            variants.append(no_numbers)
        
        # Extract just the text parts
        text_only = re.sub(r'[^\w\s]', ' ', value)
        text_only = re.sub(r'\d+', '', text_only)
        text_only = re.sub(r'\s+', ' ', text_only).strip()
        if text_only and text_only not in variants:
            variants.append(text_only)
        
        # Remove duplicates while preserving order
        unique_variants = []
        for variant in variants:
            if variant and variant not in unique_variants:
                unique_variants.append(variant)
        
        return unique_variants
    
    @staticmethod
    def detect_value_type(value: str) -> str:
        """
        Detect the type of value (numeric, text, mixed, etc.)
        
        Args:
            value: Value string to analyze
            
        Returns:
            str: Detected value type
        """
        if pd.isna(value) or not value:
            return 'empty'
        
        value = str(value).strip()
        
        # Check for numbers
        has_numbers = bool(re.search(r'\d', value))
        
        # Check for units
        has_units = any(unit.lower() in value.lower() 
                       for unit_list in DataProcessor.UNIT_MAPPINGS.values() 
                       for unit in unit_list)
        
        # Check for conditions
        has_conditions = any(keyword in value.lower() 
                           for keyword in DataProcessor.CONDITION_KEYWORDS)
        
        # Check for multiple values
        has_multiple = ',' in value or ' and ' in value.lower()
        
        # Determine type
        if has_numbers and has_units and has_conditions:
            return 'complex_measurement'
        elif has_numbers and has_units:
            return 'measurement'
        elif has_numbers and not has_units:
            return 'numeric'
        elif has_multiple:
            return 'multiple_values'
        else:
            return 'text'
    
    @staticmethod
    def calculate_value_similarity(val1_components: Dict, val2_components: Dict) -> float:
        """
        Calculate similarity between two value components
        
        Args:
            val1_components: Components of first value
            val2_components: Components of second value
            
        Returns:
            float: Similarity score (0-1)
        """
        similarity_scores = []
        
        # Number similarity
        if val1_components['numbers'] and val2_components['numbers']:
            num_sim = DataProcessor._calculate_numeric_similarity(
                val1_components['numbers'], val2_components['numbers']
            )
            similarity_scores.append(('numeric', num_sim, 0.4))
        
        # Unit similarity
        if val1_components['normalized_units'] and val2_components['normalized_units']:
            unit_sim = DataProcessor._calculate_unit_similarity(
                val1_components['normalized_units'], val2_components['normalized_units']
            )
            similarity_scores.append(('units', unit_sim, 0.3))
        
        # Condition similarity
        if val1_components['conditions'] and val2_components['conditions']:
            cond_sim = DataProcessor._calculate_condition_similarity(
                val1_components['conditions'], val2_components['conditions']
            )
            similarity_scores.append(('conditions', cond_sim, 0.3))
        
        # Calculate weighted average
        if similarity_scores:
            total_weight = sum(weight for _, _, weight in similarity_scores)
            weighted_sum = sum(score * weight for _, score, weight in similarity_scores)
            return weighted_sum / total_weight
        
        return 0.0
    
    @staticmethod
    def _calculate_numeric_similarity(nums1: List[float], nums2: List[float]) -> float:
        """Calculate similarity between numeric values"""
        if not nums1 or not nums2:
            return 0.0
        
        # For single numbers, calculate percentage difference
        if len(nums1) == 1 and len(nums2) == 1:
            diff = abs(nums1[0] - nums2[0])
            avg = (abs(nums1[0]) + abs(nums2[0])) / 2
            if avg == 0:
                return 1.0 if diff == 0 else 0.0
            return max(0, 1 - (diff / avg))
        
        # For multiple numbers, find best matches
        scores = []
        for n1 in nums1:
            for n2 in nums2:
                diff = abs(n1 - n2)
                avg = (abs(n1) + abs(n2)) / 2
                if avg == 0:
                    score = 1.0 if diff == 0 else 0.0
                else:
                    score = max(0, 1 - (diff / avg))
                scores.append(score)
        
        return max(scores) if scores else 0.0
    
    @staticmethod
    def _calculate_unit_similarity(units1: List[str], units2: List[str]) -> float:
        """Calculate similarity between unit lists"""
        if not units1 or not units2:
            return 0.0
        
        # Check for exact matches
        common_units = set(units1) & set(units2)
        if common_units:
            return 1.0
        
        # Check for compatible units (same type)
        unit_types = {
            'length': ['mm', 'cm', 'inches', 'feet', 'm'],
            'weight': ['kg', 'g', 'lb', 'oz'],
            'temperature': ['°C', '°F', 'K'],
            'pressure': ['psi', 'bar', 'Pa', 'kPa'],
            'volume': ['L', 'mL', 'gal']
        }
        
        for unit_type, type_units in unit_types.items():
            if (any(u in type_units for u in units1) and 
                any(u in type_units for u in units2)):
                return 0.8  # High similarity for same unit type
        
        return 0.0
    
    @staticmethod
    def _calculate_condition_similarity(conds1: List[str], conds2: List[str]) -> float:
        """Calculate similarity between condition lists"""
        if not conds1 or not conds2:
            return 0.0
        
        # Simple text similarity for conditions
        from rapidfuzz import fuzz
        
        max_similarity = 0.0
        for c1 in conds1:
            for c2 in conds2:
                similarity = fuzz.ratio(c1.lower(), c2.lower()) / 100.0
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
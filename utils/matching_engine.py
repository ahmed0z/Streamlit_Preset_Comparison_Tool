"""
Enhanced matching engine for comparing input values against preset database
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from rapidfuzz import fuzz, process
import streamlit as st
import config
from utils.data_processor import DataProcessor
from collections import Counter
import math

class UnitConverter:
    """Comprehensive unit conversion system for measurements
    
    Handles business-critical unit conversions to prevent duplicate entries
    and ensure format standardization across the database.
    """
    
    # Conversion factors to millimeters (base unit for length)
    LENGTH_CONVERSIONS = {
        # Metric
        'mm': 1.0,
        'millimeter': 1.0,
        'millimeters': 1.0,
        'cm': 10.0,
        'centimeter': 10.0,
        'centimeters': 10.0,
        'm': 1000.0,
        'meter': 1000.0,
        'meters': 1000.0,
        'km': 1000000.0,
        'kilometer': 1000000.0,
        'kilometers': 1000000.0,
        
        # Imperial
        'in': 25.4,
        'inch': 25.4,
        'inches': 25.4,
        '"': 25.4,  # inch symbol
        "'": 25.4,  # sometimes used for inches
        'ft': 304.8,
        'foot': 304.8,
        'feet': 304.8,
        'yd': 914.4,
        'yard': 914.4,
        'yards': 914.4,
    }
    
    # Conversion factors to grams (base unit for weight/mass)
    WEIGHT_CONVERSIONS = {
        'mg': 0.001,
        'milligram': 0.001,
        'g': 1.0,
        'gram': 1.0,
        'grams': 1.0,
        'kg': 1000.0,
        'kilogram': 1000.0,
        'kilograms': 1000.0,
        'lb': 453.592,
        'lbs': 453.592,
        'pound': 453.592,
        'pounds': 453.592,
        'oz': 28.3495,
        'ounce': 28.3495,
        'ounces': 28.3495,
    }
    
    @classmethod
    def normalize_unit(cls, unit: str) -> str:
        """Normalize unit string for conversion"""
        return unit.lower().strip().replace('"', 'in').replace("'", 'ft')
    
    @classmethod
    def convert_length(cls, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert length measurements between units"""
        from_norm = cls.normalize_unit(from_unit)
        to_norm = cls.normalize_unit(to_unit)
        
        if from_norm not in cls.LENGTH_CONVERSIONS or to_norm not in cls.LENGTH_CONVERSIONS:
            return None
            
        # Convert to base unit (mm) then to target unit
        base_value = value * cls.LENGTH_CONVERSIONS[from_norm]
        converted_value = base_value / cls.LENGTH_CONVERSIONS[to_norm]
        
        return converted_value
    
    @classmethod
    def convert_weight(cls, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert weight measurements between units"""
        from_norm = cls.normalize_unit(from_unit)
        to_norm = cls.normalize_unit(to_unit)
        
        if from_norm not in cls.WEIGHT_CONVERSIONS or to_norm not in cls.WEIGHT_CONVERSIONS:
            return None
            
        base_value = value * cls.WEIGHT_CONVERSIONS[from_norm]
        converted_value = base_value / cls.WEIGHT_CONVERSIONS[to_norm]
        
        return converted_value
    
    @classmethod
    def get_unit_type(cls, unit: str) -> Optional[str]:
        """Determine the type of unit (length, weight, volume)"""
        norm_unit = cls.normalize_unit(unit)
        
        if norm_unit in cls.LENGTH_CONVERSIONS:
            return 'length'
        elif norm_unit in cls.WEIGHT_CONVERSIONS:
            return 'weight'
        else:
            return None
    
    @classmethod
    def are_convertible_units(cls, unit1: str, unit2: str) -> bool:
        """Check if two units can be converted between each other"""
        type1 = cls.get_unit_type(unit1)
        type2 = cls.get_unit_type(unit2)
        
        return type1 is not None and type1 == type2
    
    @classmethod
    def convert_value(cls, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Auto-detect unit type and convert appropriately"""
        unit_type = cls.get_unit_type(from_unit)
        
        if unit_type == 'length':
            return cls.convert_length(value, from_unit, to_unit)
        elif unit_type == 'weight':
            return cls.convert_weight(value, from_unit, to_unit)
        else:
            return None
    
    @classmethod
    def mm_to_inches_with_mm(cls, mm_value: float) -> str:
        """Convert mm to standardized format: x.xx\" (xxx.xxmm)
        
        Business Rule: This format prevents duplicate entries by standardizing
        all length measurements to inches with mm conversion in parentheses.
        """
        inches = mm_value / 25.4
        return f'{inches:.2f}" ({mm_value:.2f}mm)'
    
    @classmethod
    def parse_inches_with_mm_format(cls, value_str: str) -> Optional[Dict[str, float]]:
        """Parse format like '4.61" (116.99mm)' to extract both values
        
        Returns:
            Dict with 'inches', 'mm', and 'mm_from_inches' values
        """
        # Pattern to match: number + " + space + (number + mm)
        pattern = r'([\d\.]+)\"\s*\(([\d\.]+)mm\)'
        match = re.search(pattern, value_str)
        
        if match:
            inches_value = float(match.group(1))
            mm_value = float(match.group(2))
            return {
                'inches': inches_value,
                'mm': mm_value,
                'mm_from_inches': inches_value * 25.4  # Convert inches back to mm for verification
            }
        return None
    
    @classmethod
    def values_are_equivalent(cls, value1: float, unit1: str, value2: float, unit2: str, 
                             tolerance_percent: float = 0.5) -> Tuple[bool, float, str]:
        """Check if two values are equivalent after unit conversion
        
        Args:
            value1, unit1: First value and unit
            value2, unit2: Second value and unit  
            tolerance_percent: Tolerance for rounding differences (0.5% default)
            
        Returns:
            (is_equivalent, similarity_score, explanation)
        """
        if not cls.are_convertible_units(unit1, unit2):
            return False, 0.0, "Different unit types - not convertible"
        
        # Convert both to same base unit for comparison
        unit_type = cls.get_unit_type(unit1)
        if unit_type == 'length':
            base1_mm = cls.convert_length(value1, unit1, 'mm')
            base2_mm = cls.convert_length(value2, unit2, 'mm')
            base_unit = 'mm'
        elif unit_type == 'weight':
            base1_mm = cls.convert_weight(value1, unit1, 'g')
            base2_mm = cls.convert_weight(value2, unit2, 'g')
            base_unit = 'g'
        else:
            return False, 0.0, "Unknown unit type"
        
        if base1_mm is None or base2_mm is None:
            return False, 0.0, "Unit conversion failed"
        
        # Calculate percentage difference
        if base2_mm == 0:
            if base1_mm == 0:
                return True, 1.0, "Both values are zero"
            else:
                return False, 0.0, "Cannot compare zero with non-zero value"
        
        diff_percent = abs(base1_mm - base2_mm) / base2_mm * 100
        
        if diff_percent <= tolerance_percent:
            similarity = 1.0 - (diff_percent / tolerance_percent) * 0.1  # Small penalty for rounding
            return True, similarity, f"Values are equivalent within {tolerance_percent}% tolerance (diff: {diff_percent:.2f}%)"
        else:
            # Calculate similarity based on difference
            similarity = max(0.0, 1.0 - (diff_percent / 100.0))  # Decrease similarity with difference
            return False, similarity, f"Values differ by {diff_percent:.2f}% after conversion ({base1_mm:.2f}{base_unit} vs {base2_mm:.2f}{base_unit})"


class EnhancedMatchingEngine:
    """Enhanced matching engine with advanced rules for exact/partial matching"""
    
    def __init__(self, preset_df: pd.DataFrame):
        """Initialize with preset database"""
        self.preset_df = preset_df
        self.preprocessed_presets = None
        self.format_patterns = {}  # Store common format patterns per context
        self._prepare_preset_data()
        self._analyze_format_patterns()
    
    def _prepare_preset_data(self):
        """Prepare preset data for efficient matching"""
        if self.preset_df is not None and not self.preset_df.empty:
            preset_col = self._get_preset_values_column()
            if preset_col:
                self.preprocessed_presets = DataProcessor.prepare_for_matching(
                    self.preset_df, preset_col
                )
    
    def _get_preset_values_column(self) -> Optional[str]:
        """Get the preset values column name"""
        possible_names = ['Preset values', 'Preset Values', 'Values', 'Preset Value', 'Reference Values']
        for col in self.preset_df.columns:
            if col in possible_names:
                return col
        return self.preset_df.columns[-1] if len(self.preset_df.columns) > 0 else None
    
    def _analyze_format_patterns(self):
        """Analyze common formatting patterns in preset data for normalization
        
        Format Normalization (Composite Key Rule):
        - Detect the most common formatting style in the Preset DB
        - Example: if most entries are "1V" (no space), normalize to "1V"
        - If majority are "1 V" (with space), suggest "1 V"
        - Apply consistently across all comparisons
        """
        if self.preprocessed_presets is None:
            return
            
        preset_col = self._get_preset_values_column()
        if not preset_col:
            return
        
        # Use larger sample for better pattern detection
        sample_size = min(5000, len(self.preset_df))
        sample_df = self.preset_df.head(sample_size)
        
        # Analyze patterns by composite key context
        composite_keys = self._get_unique_composite_keys(sample_df)
        
        for context_key in composite_keys[:50]:  # Limit for performance
            category, sub_cat, attr_name = context_key.split('|')
            
            # Get values for this specific context
            context_mask = (
                (sample_df['Category'] == category) &
                (sample_df['Sub-Category'] == sub_cat) &
                (sample_df['Attribute Name'] == attr_name)
            )
            context_values = sample_df[context_mask][preset_col].dropna().astype(str).tolist()
            
            if len(context_values) >= 5:  # Need sufficient samples
                self.format_patterns[context_key] = self._detect_common_format_enhanced(context_values)
    
    def _get_unique_composite_keys(self, df: pd.DataFrame) -> List[str]:
        """Get unique composite keys from dataframe"""
        composite_keys = []
        
        for _, row in df.iterrows():
            category = str(row.get('Category', '')).strip()
            sub_cat = str(row.get('Sub-Category', '')).strip()
            attr_name = str(row.get('Attribute Name', '')).strip()
            
            if category and sub_cat and attr_name:
                key = f"{category}|{sub_cat}|{attr_name}"
                if key not in composite_keys:
                    composite_keys.append(key)
        
        return composite_keys
    
    def _detect_common_format_enhanced(self, values: List[str]) -> Dict[str, Any]:
        """Enhanced format pattern detection for normalization
        
        Analyzes:
        - Unit spacing patterns ("1V" vs "1 V")
        - Case patterns ("kg" vs "KG")
        - Decimal formatting ("1.0" vs "1")
        - Special characters and punctuation
        """
        patterns = {
            'spacing_with_units': [],
            'case_patterns': [],
            'decimal_patterns': [],
            'punctuation_patterns': []
        }
        
        for value in values[:100]:  # Analyze up to 100 values per context
            # Analyze unit spacing patterns
            if re.search(r'\d\s+[a-zA-Z]', value):
                patterns['spacing_with_units'].append('with_space')
            elif re.search(r'\d[a-zA-Z]', value):
                patterns['spacing_with_units'].append('no_space')
            
            # Analyze case patterns for units
            units = re.findall(r'[a-zA-Z]+', value)
            for unit in units:
                if unit.islower():
                    patterns['case_patterns'].append('lowercase')
                elif unit.isupper():
                    patterns['case_patterns'].append('uppercase')
                else:
                    patterns['case_patterns'].append('mixed')
            
            # Analyze decimal formatting
            numbers = re.findall(r'\d+\.\d+', value)
            for num in numbers:
                if num.endswith('.0'):
                    patterns['decimal_patterns'].append('show_zero')
                else:
                    patterns['decimal_patterns'].append('hide_zero')
            
            # Analyze punctuation usage
            if ',' in value:
                patterns['punctuation_patterns'].append('use_comma')
            if ';' in value:
                patterns['punctuation_patterns'].append('use_semicolon')
        
        # Determine most common patterns
        common_format = {}
        for pattern_type, pattern_list in patterns.items():
            if pattern_list:
                counter = Counter(pattern_list)
                most_common = counter.most_common(1)[0]
                # Only consider pattern if it appears in >50% of values
                if most_common[1] > len(pattern_list) * 0.5:
                    common_format[pattern_type] = most_common[0]
                    common_format[f'{pattern_type}_confidence'] = most_common[1] / len(pattern_list)
        
        return common_format
    

    
    def find_matches(self, input_value: str, category: str = None, 
                    sub_category: str = None, attribute_name: str = None) -> List[Dict[str, Any]]:
        """Find matches using enhanced rules according to specifications"""
        if not input_value or pd.isna(input_value):
            return []
        
        input_value = str(input_value).strip()
        
        if self.preprocessed_presets is None or self.preprocessed_presets.empty:
            return []
        
        # Filter by composite key
        search_df = self._filter_by_composite_key(category, sub_category, attribute_name)
        
        if search_df.empty:
            return []
        
        # Get context for format normalization
        context_key = f"{category}|{sub_category}|{attribute_name}"
        common_format = self.format_patterns.get(context_key, {})
        
        # RULE 1: EXACT MATCH - Highest Priority
        # Only when the entire cell value matches exactly
        # Status = Exact Match
        # If exact, do not generate duplicate suggestion rows
        exact_matches = self._find_exact_matches(input_value, search_df, common_format)
        if exact_matches:
            return exact_matches  # Return immediately, no duplicates
        
        # RULE 2: PARTIAL MATCHES
        # When part of input string matches a preset value
        # Status = Partial Match
        # Must return similarity score + explanation
        matches = []
        
        # VALUES WITH UNITS - Special handling
        if self._has_units(input_value):
            unit_matches = self._find_unit_matches(input_value, search_df, common_format)
            matches.extend(unit_matches)
        
        # Other partial matches
        partial_matches = self._find_partial_matches(input_value, search_df, common_format)
        matches.extend(partial_matches)
        
        # RULE 3: Similarity Threshold
        # Only consider matches with ≥75% similarity
        matches = [m for m in matches if m['similarity'] >= config.MATCHING_THRESHOLD]
        
        # Remove duplicates and sort by similarity
        unique_matches = self._remove_duplicate_matches(matches)
        unique_matches = sorted(unique_matches, key=lambda x: x['similarity'], reverse=True)
        unique_matches = unique_matches[:config.MAX_RESULTS_PER_INPUT]
        
        return unique_matches
    
    def _filter_by_composite_key(self, category: str, sub_category: str, attribute_name: str) -> pd.DataFrame:
        """Filter dataset by composite key"""
        search_df = self.preprocessed_presets.copy()
        
        if category:
            category_cols = [col for col in search_df.columns if col in ['Category', 'category']]
            if category_cols:
                search_df = search_df[search_df[category_cols[0]].str.contains(
                    str(category), case=False, na=False, regex=False
                )]
        
        if sub_category:
            sub_category_cols = [col for col in search_df.columns if col in ['Sub-Category', 'sub-category', 'Sub Category']]
            if sub_category_cols:
                search_df = search_df[search_df[sub_category_cols[0]].str.contains(
                    str(sub_category), case=False, na=False, regex=False
                )]
        
        if attribute_name:
            attr_cols = [col for col in search_df.columns if col in ['Attribute Name', 'attribute_name', 'Attribute']]
            if attr_cols:
                search_df = search_df[search_df[attr_cols[0]].str.contains(
                    str(attribute_name), case=False, na=False, regex=False
                )]
        
        return search_df
    
    def _remove_duplicate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate matches based on preset value"""
        seen_values = set()
        unique_matches = []
        
        for match in matches:
            preset_value = match['preset_value']
            if preset_value not in seen_values:
                seen_values.add(preset_value)
                unique_matches.append(match)
        
        return unique_matches
    
    def _find_exact_matches(self, input_value: str, search_df: pd.DataFrame, 
                           common_format: Dict) -> List[Dict[str, Any]]:
        """Find exact matches - entire cell value matches exactly
        
        RULE: Only when the entire cell value matches exactly with a preset value.
        Status = Exact Match.
        If exact, do not generate duplicate suggestion rows.
        """
        preset_col = self._get_preset_values_column()
        if not preset_col:
            return []
        
        # Search for exact matches
        for idx, row in search_df.iterrows():
            preset_value = str(row[preset_col]) if pd.notna(row[preset_col]) else ""
            if not preset_value:
                continue
            
            # Strict exact match check
            if self._is_exact_match(input_value, preset_value):
                return [{
                    'preset_value': preset_value,
                    'similarity': 1.0,
                    'match_type': 'exact',
                    'comment': 'Perfect exact match found',
                    'suggested_value': preset_value,  # Keep original preset format
                    'status': config.MatchStatus.EXACT_MATCH,
                    'row_data': row.to_dict()
                }]  # Return immediately with single match, no duplicates
        
        return []  # No exact matches found
    
    def _is_exact_match(self, input_value: str, preset_value: str) -> bool:
        """Check if two values are exact matches based on ORIGINAL values
        
        IMPORTANT: Exact match is based on original values, not normalized values
        This ensures case sensitivity and exact formatting preservation
        
        Rules:
        - Must be character-by-character identical
        - Case sensitive comparison
        - Whitespace sensitive comparison
        - No normalization applied
        """
        # Direct comparison of original values - no normalization
        return input_value.strip() == preset_value.strip()
    
    def _normalize_for_exact_match(self, value: str) -> str:
        """Normalize value for exact matching"""
        # Remove extra whitespace, normalize case
        normalized = re.sub(r'\s+', ' ', value.strip().lower())
        return normalized
    
    def _has_units(self, value: str) -> bool:
        """Check if value contains units"""
        # Check for common unit patterns
        unit_pattern = r'\d+\.?\d*\s*[a-zA-Z\Ω°]+'
        return bool(re.search(unit_pattern, value))
    
    def _find_unit_matches(self, input_value: str, search_df: pd.DataFrame, 
                          common_format: Dict) -> List[Dict[str, Any]]:
        """Handle values with units according to specifications
        
        ENHANCED: Now includes unit conversion detection for business-critical 
        format standardization (e.g., '117mm' → '4.61" (116.99mm)')
        
        RULE: Values with Units
        Always split into two sections:
        1. Numeric value (parsed as float if possible)
        2. Unit (standardized: e.g., kg, V, cm)
        
        Compare both independently:
        - Value comparison (numeric equivalence, tolerance if needed)
        - Unit comparison (case-insensitive, synonyms normalized)
        - Unit conversion detection for equivalent measurements
        """
        preset_col = self._get_preset_values_column()
        if not preset_col:
            return []
        
        # Parse input value into numeric and unit components
        input_parsed = self._parse_value_with_unit_enhanced(input_value)
        if not input_parsed:
            return []
        
        matches = []
        
        for idx, row in search_df.iterrows():
            preset_value = str(row[preset_col]) if pd.notna(row[preset_col]) else ""
            if not preset_value:
                continue
            
            # Check for standard inches+mm format first (business rule)
            inches_mm_parsed = UnitConverter.parse_inches_with_mm_format(preset_value)
            if inches_mm_parsed:
                # Handle conversion from simple unit (e.g., 'mm') to inches+mm format
                match_result = self._compare_with_inches_mm_format(
                    input_parsed, inches_mm_parsed, preset_value, row
                )
                if match_result:
                    matches.append(match_result)
                continue
            
            # Parse preset value for standard unit comparison
            preset_parsed = self._parse_value_with_unit_enhanced(preset_value)
            if not preset_parsed:
                continue
            
            # Check for unit conversion equivalence
            input_unit = input_parsed['unit']
            preset_unit = preset_parsed['unit']
            
            if UnitConverter.are_convertible_units(input_unit, preset_unit):
                # Values are convertible - check equivalence
                is_equivalent, similarity, explanation = UnitConverter.values_are_equivalent(
                    input_parsed['numeric'], input_unit,
                    preset_parsed['numeric'], preset_unit
                )
                
                if is_equivalent or similarity >= config.MATCHING_THRESHOLD:
                    # Generate conversion comment
                    comment = self._generate_conversion_comment(
                        input_parsed, preset_parsed, similarity, explanation, is_equivalent
                    )
                    
                    matches.append({
                        'preset_value': preset_value,
                        'similarity': similarity,
                        'match_type': 'unit_conversion',
                        'comment': comment,
                        'suggested_value': preset_value,  # Use exact preset formatting
                        'status': config.MatchStatus.EXACT_MATCH if is_equivalent else config.MatchStatus.PARTIAL_MATCH,
                        'row_data': row.to_dict(),
                        'conversion_info': {
                            'input_value': input_parsed['numeric'],
                            'input_unit': input_unit,
                            'preset_value': preset_parsed['numeric'],
                            'preset_unit': preset_unit,
                            'explanation': explanation
                        }
                    })
            else:
                # Standard unit comparison (existing logic)
                numeric_similarity = self._compare_numeric_values_enhanced(
                    input_parsed['numeric'], preset_parsed['numeric']
                )
                unit_similarity = self._compare_units_enhanced(
                    input_unit, preset_unit
                )
                
                # Combined similarity with proper weighting
                overall_similarity = (numeric_similarity * 0.7) + (unit_similarity * 0.3)
                
                if overall_similarity >= config.MATCHING_THRESHOLD:
                    # Generate specific comment for unit matches
                    comment = self._generate_unit_comment_enhanced(
                        input_parsed, preset_parsed, numeric_similarity, unit_similarity
                    )
                    
                    matches.append({
                        'preset_value': preset_value,
                        'similarity': overall_similarity,
                        'match_type': 'unit_based',
                        'comment': comment,
                        'suggested_value': preset_value,  # Use exact preset formatting
                        'status': config.MatchStatus.PARTIAL_MATCH,
                        'row_data': row.to_dict()
                    })
        
        return matches
    
    def _compare_with_inches_mm_format(self, input_parsed: Dict, inches_mm_parsed: Dict, 
                                      preset_value: str, preset_row) -> Optional[Dict[str, Any]]:
        """Compare input value with preset in inches+mm format
        
        Business Rule: Handle cases like '117mm' vs '4.61" (116.99mm)'
        """
        input_unit = input_parsed['unit']
        input_value = input_parsed['numeric']
        
        # Check if input is in mm and can be compared with the mm value in parentheses
        if UnitConverter.normalize_unit(input_unit) == 'mm':
            preset_mm = inches_mm_parsed['mm']
            
            # Compare mm values directly
            is_equivalent, similarity, explanation = UnitConverter.values_are_equivalent(
                input_value, 'mm', preset_mm, 'mm', tolerance_percent=1.0  # Allow 1% tolerance for rounding
            )
            
            if is_equivalent or similarity >= config.MATCHING_THRESHOLD:
                # Generate standardization comment
                comment = f"Input {input_value}mm matches preset {preset_mm}mm (standardized format: {inches_mm_parsed['inches']:.2f}\" with mm conversion)"
                if not is_equivalent:
                    comment += f" - {explanation}"
                
                return {
                    'preset_value': preset_value,
                    'similarity': similarity,
                    'match_type': 'format_standardization',
                    'comment': comment,
                    'suggested_value': preset_value,  # Use standardized format
                    'status': config.MatchStatus.EXACT_MATCH if is_equivalent else config.MatchStatus.PARTIAL_MATCH,
                    'row_data': preset_row.to_dict(),
                    'standardization_info': {
                        'input_mm': input_value,
                        'preset_mm': preset_mm,
                        'preset_inches': inches_mm_parsed['inches'],
                        'suggested_format': 'inches_with_mm_parentheses'
                    }
                }
        
        # Check if input is in inches and can be compared with the inches value
        elif UnitConverter.normalize_unit(input_unit) in ['in', 'inch', 'inches', '"']:
            preset_inches = inches_mm_parsed['inches']
            
            is_equivalent, similarity, explanation = UnitConverter.values_are_equivalent(
                input_value, 'in', preset_inches, 'in', tolerance_percent=1.0
            )
            
            if is_equivalent or similarity >= config.MATCHING_THRESHOLD:
                comment = f"Input {input_value}\" matches preset {preset_inches:.2f}\" (standardized format includes mm conversion)"
                if not is_equivalent:
                    comment += f" - {explanation}"
                
                return {
                    'preset_value': preset_value,
                    'similarity': similarity,
                    'match_type': 'format_standardization',
                    'comment': comment,
                    'suggested_value': preset_value,
                    'status': config.MatchStatus.EXACT_MATCH if is_equivalent else config.MatchStatus.PARTIAL_MATCH,
                    'row_data': preset_row.to_dict(),
                    'standardization_info': {
                        'input_inches': input_value,
                        'preset_inches': preset_inches,
                        'preset_mm': inches_mm_parsed['mm'],
                        'suggested_format': 'inches_with_mm_parentheses'
                    }
                }
        
        return None
    
    def _generate_conversion_comment(self, input_parsed: Dict, preset_parsed: Dict, 
                                    similarity: float, explanation: str, is_equivalent: bool) -> str:
        """Generate comment for unit conversion matches"""
        input_str = f"{input_parsed['numeric']}{input_parsed['unit']}"
        preset_str = f"{preset_parsed['numeric']}{preset_parsed['unit']}"
        
        if is_equivalent:
            return f"Equivalent measurement: {input_str} = {preset_str} (unit conversion detected)"
        else:
            return f"Similar measurement: {input_str} vs {preset_str} ({similarity:.1%} match after conversion) - {explanation}"
    
    def _parse_value_with_unit_enhanced(self, value: str) -> Optional[Dict[str, Any]]:
        """Enhanced parsing of value into numeric and unit components
        
        Handles various formats:
        - "20 kg", "20kg", "20.5 kg"
        - "5.2 V", "5.2V", "5V"
        - "10 cm", "10.0cm"
        - Complex units: "20 kg @ 30°C", "1.5 A/V"
        - Complex dimensions: '2.8" L x 0.70" W (71.1 mm x 17.8m)'
        
        Returns parsed info with original formatting preserved for comparison
        """
        # Store original value for format analysis
        original_value = value
        
        # Remove extra whitespace but preserve structure
        value = re.sub(r'\s+', ' ', value.strip())
        
        # Enhanced pattern to capture various unit formats including complex dimensions
        patterns = [
            # Complex dimensions with parentheses: '2.8" L x 0.70" W (71.1 mm x 17.8m)'
            r'([\d\.]+)\s*(["\']?)\s*([LWHlwhDd]?)\s*x\s*([\d\.]+)\s*(["\']?)\s*([LWHlwhDd]?)\s*(?:\(([^)]+)\))?',
            # Standard: number + space + unit (e.g., "20 kg", "5.2 V")
            r'([+-]?\d+\.?\d*)\s+([a-zA-ZΩ°/"\'\_\-]+)',
            # No space: number + unit (e.g., "20kg", "5V")
            r'([+-]?\d+\.?\d*)([a-zA-ZΩ°/"\'\_\-]+)',
            # With conditions: "20 kg @ 30°C" - extract main value and unit
            r'([+-]?\d+\.?\d*)\s+([a-zA-ZΩ°]+)(?:\s*@.*)?',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, value)
            if match:
                try:
                    if i == 0:  # Complex dimensions pattern
                        # Extract dimension information
                        length = float(match.group(1))
                        length_unit = (match.group(2) or '') + (match.group(3) or '')
                        width = float(match.group(4))
                        width_unit = (match.group(5) or '') + (match.group(6) or '')
                        parentheses_content = match.group(7) or ''
                        
                        return {
                            'numeric': length,  # Primary dimension
                            'unit': length_unit or 'dimension',
                            'original': original_value,
                            'has_conditions': '@' in value or '°' in value,
                            'is_complex_dimension': True,
                            'dimension_info': {
                                'length': length,
                                'length_unit': length_unit,
                                'width': width, 
                                'width_unit': width_unit,
                                'parentheses_content': parentheses_content
                            },
                            'format_details': self._extract_format_details(original_value)
                        }
                    else:
                        # Standard patterns
                        numeric = float(match.group(1))
                        unit = match.group(2).strip()
                        return {
                            'numeric': numeric,
                            'unit': unit,
                            'original': original_value,
                            'has_conditions': '@' in value or '°' in value,
                            'is_complex_dimension': False,
                            'format_details': self._extract_format_details(original_value)
                        }
                except ValueError:
                    continue
        
        return None
    
    def _extract_format_details(self, value: str) -> Dict[str, Any]:
        """Extract detailed formatting information from a value
        
        Analyzes:
        - Spacing patterns
        - Case patterns
        - Punctuation usage
        - Quote marks
        - Parentheses usage
        - Decimal formatting
        """
        details = {
            'has_quotes': '"' in value or "'" in value,
            'quote_types': [],
            'has_parentheses': '(' in value and ')' in value,
            'spacing_around_x': None,
            'decimal_places': {},
            'case_pattern': None,
            'units_found': [],
            'has_mm': 'mm' in value,
            'has_inches': '"' in value or "'" in value or 'inch' in value.lower(),
        }
        
        # Analyze quote types
        if '"' in value:
            details['quote_types'].append('double')
        if "'" in value:
            details['quote_types'].append('single')
            
        # Analyze spacing around 'x'
        x_matches = re.findall(r'(\S?)\s*x\s*(\S?)', value, re.IGNORECASE)
        if x_matches:
            before, after = x_matches[0]
            if before and after:
                details['spacing_around_x'] = 'none'
            else:
                details['spacing_around_x'] = 'spaces'
                
        # Analyze decimal places for numbers
        numbers = re.findall(r'\d+\.\d+', value)
        for i, num in enumerate(numbers):
            decimal_part = num.split('.')[1]
            details['decimal_places'][f'number_{i}'] = len(decimal_part)
            
        # Analyze units
        units = re.findall(r'[a-zA-Z]+', value)
        details['units_found'] = list(set(units))
        
        # Analyze case patterns for units
        for unit in units:
            if unit.islower():
                details['case_pattern'] = 'lowercase'
            elif unit.isupper():
                details['case_pattern'] = 'uppercase'
            else:
                details['case_pattern'] = 'mixed'
                break
                
        return details
    
    def _compare_numeric_values_enhanced(self, val1: float, val2: float, tolerance: float = 0.001) -> float:
        """Enhanced numeric comparison with configurable tolerance
        
        Returns:
        - 1.0 for exact match
        - High similarity for values within tolerance
        - Decreasing similarity based on percentage difference
        """
        if val1 == val2:
            return 1.0
        
        # Handle zero values
        if val2 == 0:
            return 1.0 if val1 == 0 else 0.0
        
        # Calculate percentage difference
        diff_percent = abs(val1 - val2) / abs(val2)
        
        # Within tolerance is considered exact
        if diff_percent <= tolerance:
            return 1.0
        
        # Gradual decrease in similarity based on difference
        if diff_percent <= 0.05:  # Within 5%
            return 0.95
        elif diff_percent <= 0.10:  # Within 10%
            return 0.85
        elif diff_percent <= 0.20:  # Within 20%
            return 0.75
        else:
            # Exponential decay for larger differences
            return max(0.0, 1.0 - diff_percent)
    
    def _compare_units_enhanced(self, unit1: str, unit2: str) -> float:
        """Enhanced unit comparison with standardization and synonyms
        
        Handles:
        - Case-insensitive comparison
        - Unit synonyms (kg/kilogram, V/volt, etc.)
        - Abbreviations vs full names
        - Common variations
        """
        # Normalize both units
        unit1_norm = self._standardize_unit(unit1)
        unit2_norm = self._standardize_unit(unit2)
        
        # Exact match after normalization
        if unit1_norm == unit2_norm:
            return 1.0
        
        # Check for synonymous units
        if self._are_synonymous_units_enhanced(unit1_norm, unit2_norm):
            return 0.95
        
        # Fuzzy matching for similar units
        fuzzy_similarity = fuzz.ratio(unit1_norm, unit2_norm) / 100.0
        
        # Only consider reasonable similarities
        return fuzzy_similarity if fuzzy_similarity >= 0.7 else 0.0
    
    def _standardize_unit(self, unit: str) -> str:
        """Standardize unit to common format (kg, V, cm, etc.)"""
        unit = unit.lower().strip()
        
        # Comprehensive unit mapping to standardized forms
        unit_standards = {
            # Weight/Mass
            'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg', 'kilo': 'kg',
            'g': 'g', 'gram': 'g', 'grams': 'g', 'gr': 'g',
            'lb': 'lb', 'lbs': 'lb', 'pound': 'lb', 'pounds': 'lb',
            
            # Electrical
            'v': 'v', 'volt': 'v', 'volts': 'v', 'voltage': 'v',
            'a': 'a', 'amp': 'a', 'amps': 'a', 'ampere': 'a', 'amperes': 'a',
            'w': 'w', 'watt': 'w', 'watts': 'w',
            'ohm': 'ohm', 'ohms': 'ohm', 'ω': 'ohm', 'Ω': 'ohm',
            
            # Length/Distance
            'mm': 'mm', 'millimeter': 'mm', 'millimeters': 'mm', 'millimetre': 'mm',
            'cm': 'cm', 'centimeter': 'cm', 'centimeters': 'cm', 'centimetre': 'cm',
            'm': 'm', 'meter': 'm', 'meters': 'm', 'metre': 'm', 'metres': 'm',
            'km': 'km', 'kilometer': 'km', 'kilometers': 'km', 'kilometre': 'km',
            'in': 'in', 'inch': 'in', 'inches': 'in',
            'ft': 'ft', 'foot': 'ft', 'feet': 'ft',
            
            # Volume
            'l': 'l', 'liter': 'l', 'liters': 'l', 'litre': 'l', 'litres': 'l',
            'ml': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
            
            # Temperature
            '°c': '°c', 'celsius': '°c', 'c': '°c',
            '°f': '°f', 'fahrenheit': '°f', 'f': '°f',
            'k': 'k', 'kelvin': 'k',
        }
        
        return unit_standards.get(unit, unit)
    
    def _are_synonymous_units_enhanced(self, unit1: str, unit2: str) -> bool:
        """Check if units are synonymous using enhanced grouping"""
        synonym_groups = [
            # Electrical
            ['v', 'volt'],
            ['a', 'amp', 'ampere'],
            ['w', 'watt'],
            ['ohm', 'ω'],
            
            # Weight/Mass
            ['kg', 'kilogram'],
            ['g', 'gram'],
            ['lb', 'pound'],
            
            # Length
            ['mm', 'millimeter'],
            ['cm', 'centimeter'],
            ['m', 'meter'],
            ['km', 'kilometer'],
            ['in', 'inch'],
            ['ft', 'foot'],
            
            # Volume
            ['l', 'liter'],
            ['ml', 'milliliter'],
            
            # Temperature
            ['°c', 'celsius'],
            ['°f', 'fahrenheit'],
            ['k', 'kelvin']
        ]
        
        for group in synonym_groups:
            if unit1 in group and unit2 in group:
                return True
        
        return False
    
    def _generate_unit_comment_enhanced(self, input_parsed: Dict, preset_parsed: Dict, 
                              numeric_sim: float, unit_sim: float) -> str:
        """Generate detailed comment for unit-based matches with specific formatting differences
        
        Provides specific explanations like:
        - 'unit spacing mismatch'
        - 'different format'
        - 'same value different unit'
        - Detailed dimension formatting differences
        """
        input_val = input_parsed['numeric']
        preset_val = preset_parsed['numeric']
        input_unit = input_parsed['unit']
        preset_unit = preset_parsed['unit']
        input_original = input_parsed['original']
        preset_original = preset_parsed['original']
        
        # Get format details for both values
        input_format = input_parsed.get('format_details', {})
        preset_format = preset_parsed.get('format_details', {})
        
        # Handle complex dimensions
        if input_parsed.get('is_complex_dimension') or preset_parsed.get('is_complex_dimension'):
            return self._generate_dimension_comment(input_parsed, preset_parsed, numeric_sim, unit_sim)
        
        # Exact numeric and unit match
        if numeric_sim == 1.0 and unit_sim == 1.0:
            # Check if there are formatting differences despite same values
            format_differences = self._detect_format_differences(input_format, preset_format, input_original, preset_original)
            if format_differences:
                return f"Same value with different format: {format_differences}"
            else:
                return "Exact numeric value and unit match"
        
        # Same numeric value, different units or unit formatting
        elif numeric_sim == 1.0:
            if unit_sim > 0.9:
                # Check specific formatting differences
                format_differences = self._detect_format_differences(input_format, preset_format, input_original, preset_original)
                if format_differences:
                    return f"Same numeric value, unit format difference: {format_differences}"
                else:
                    return "Same numeric value, equivalent unit format"
            elif unit_sim > 0.7:
                return f"Same numeric value ({input_val}), similar unit ({input_unit} vs {preset_unit})"
            else:
                return f"Same numeric value ({input_val}), different unit ({input_unit} vs {preset_unit})"
        
        # Same unit, different numeric values
        elif unit_sim == 1.0:
            diff_pct = abs(input_val - preset_val) / preset_val * 100 if preset_val != 0 else 0
            return f"Same unit ({input_unit}), different numeric value ({diff_pct:.1f}% difference)"
        
        # Both values and units are similar but not exact
        elif numeric_sim > 0.9 and unit_sim > 0.9:
            format_differences = self._detect_format_differences(input_format, preset_format, input_original, preset_original)
            if format_differences:
                return f"Very similar value and unit with format differences: {format_differences}"
            else:
                return "Very similar numeric value and unit format"
        
        # Spacing or formatting differences
        elif numeric_sim > 0.95 and unit_sim > 0.7:
            format_differences = self._detect_format_differences(input_format, preset_format, input_original, preset_original)
            if format_differences:
                return f"Same value, format differences: {format_differences}"
            elif ' ' in input_original and ' ' not in preset_original:
                return "Same value, unit spacing mismatch (input has space, preset doesn't)"
            elif ' ' not in input_original and ' ' in preset_original:
                return "Same value, unit spacing mismatch (preset has space, input doesn't)"
            else:
                return "Same value, different format"
        
        # General partial match
        else:
            return f"Partial match: {numeric_sim:.0%} numeric similarity, {unit_sim:.0%} unit similarity"
    
    def _generate_dimension_comment(self, input_parsed: Dict, preset_parsed: Dict, 
                                   numeric_sim: float, unit_sim: float) -> str:
        """Generate specific comments for complex dimension values"""
        input_original = input_parsed['original']
        preset_original = preset_parsed['original']
        
        # Extract dimension info
        input_dim = input_parsed.get('dimension_info', {})
        preset_dim = preset_parsed.get('dimension_info', {})
        
        differences = []
        
        # Compare dimensions
        if input_dim.get('length') == preset_dim.get('length'):
            if input_dim.get('width') == preset_dim.get('width'):
                # Same dimensions, check formatting
                format_diffs = self._detect_dimension_format_differences(input_original, preset_original)
                if format_diffs:
                    return f"Same dimensions, different format: {', '.join(format_diffs)}"
                else:
                    return "Same dimensions and format"
            else:
                differences.append(f"width differs ({input_dim.get('width')} vs {preset_dim.get('width')})")
        else:
            differences.append(f"length differs ({input_dim.get('length')} vs {preset_dim.get('length')})")
        
        # Check parentheses content
        input_paren = input_dim.get('parentheses_content', '')
        preset_paren = preset_dim.get('parentheses_content', '')
        if input_paren != preset_paren:
            differences.append("parentheses content differs")
        
        if differences:
            return f"Similar dimensions with differences: {', '.join(differences)}"
        else:
            return f"Partial dimension match: {numeric_sim:.0%} similarity"
    
    def _detect_format_differences(self, input_format: Dict, preset_format: Dict, 
                                  input_original: str, preset_original: str) -> str:
        """Detect specific formatting differences between input and preset values"""
        differences = []
        
        # Check spacing differences
        if ' ' in input_original and ' ' not in preset_original:
            differences.append("input has spaces, preset doesn't")
        elif ' ' not in input_original and ' ' in preset_original:
            differences.append("preset has spaces, input doesn't")
        
        # Check quote differences
        input_quotes = input_format.get('quote_types', [])
        preset_quotes = preset_format.get('quote_types', [])
        if set(input_quotes) != set(preset_quotes):
            if input_quotes and not preset_quotes:
                differences.append("input has quotes, preset doesn't")
            elif preset_quotes and not input_quotes:
                differences.append("preset has quotes, input doesn't")
            elif input_quotes != preset_quotes:
                differences.append(f"different quote types ({input_quotes} vs {preset_quotes})")
        
        # Check parentheses differences
        if input_format.get('has_parentheses') != preset_format.get('has_parentheses'):
            if input_format.get('has_parentheses'):
                differences.append("input has parentheses, preset doesn't")
            else:
                differences.append("preset has parentheses, input doesn't")
        
        # Check case differences
        input_case = input_format.get('case_pattern')
        preset_case = preset_format.get('case_pattern')
        if input_case != preset_case and input_case and preset_case:
            differences.append(f"case differs ({input_case} vs {preset_case})")
        
        # Check decimal places
        input_decimals = input_format.get('decimal_places', {})
        preset_decimals = preset_format.get('decimal_places', {})
        if len(input_decimals) != len(preset_decimals):
            differences.append("different decimal precision")
        
        return ", ".join(differences) if differences else ""
    
    def _detect_dimension_format_differences(self, input_original: str, preset_original: str) -> List[str]:
        """Detect specific format differences in dimension strings"""
        differences = []
        
        # Check for spacing around 'x'
        input_x_spacing = bool(re.search(r'\s+x\s+', input_original))
        preset_x_spacing = bool(re.search(r'\s+x\s+', preset_original))
        if input_x_spacing != preset_x_spacing:
            if input_x_spacing:
                differences.append("input has spaces around 'x', preset doesn't")
            else:
                differences.append("preset has spaces around 'x', input doesn't")
        
        # Check parentheses content formatting
        input_paren_match = re.search(r'\(([^)]+)\)', input_original)
        preset_paren_match = re.search(r'\(([^)]+)\)', preset_original)
        
        if input_paren_match and preset_paren_match:
            input_paren = input_paren_match.group(1)
            preset_paren = preset_paren_match.group(1)
            
            # Check for unit spacing differences in parentheses
            if 'mm' in input_paren and 'mm' in preset_paren:
                input_mm_spacing = bool(re.search(r'\d\s+mm', input_paren))
                preset_mm_spacing = bool(re.search(r'\d\s+mm', preset_paren))
                if input_mm_spacing != preset_mm_spacing:
                    if input_mm_spacing:
                        differences.append("input has spaces before 'mm', preset doesn't")
                    else:
                        differences.append("preset has spaces before 'mm', input doesn't")
            
            # Check for different units in parentheses (m vs mm)
            input_units = set(re.findall(r'[a-zA-Z]+', input_paren))
            preset_units = set(re.findall(r'[a-zA-Z]+', preset_paren))
            if input_units != preset_units:
                differences.append(f"different units in parentheses ({input_units} vs {preset_units})")
        
        # Check quote mark formatting
        input_quotes = input_original.count('"')
        preset_quotes = preset_original.count('"')
        if input_quotes != preset_quotes:
            differences.append(f"different number of quote marks ({input_quotes} vs {preset_quotes})")
        
        return differences
    
    def _format_unit_value_enhanced(self, numeric: float, unit: str, common_format: Dict, 
                                    preset_original: str = None) -> str:
        """Format unit value to match preset formatting exactly
        
        IMPORTANT: Suggested values must match preset formatting exactly
        - Case sensitivity preserved
        - Spacing patterns preserved  
        - Punctuation preserved
        - Unit format preserved
        
        Args:
            numeric: The numeric value
            unit: The unit
            common_format: Common formatting patterns (optional)
            preset_original: The original preset value to copy formatting from
        
        Returns:
            Formatted value that exactly matches preset formatting
        """
        # If we have a preset original value, use its exact formatting
        if preset_original:
            return preset_original
        
        # Fallback: try to apply common formatting patterns
        spacing_pattern = common_format.get('spacing_with_units', 'with_space')
        case_pattern = common_format.get('case_patterns', 'preserve')
        
        # Preserve original unit case if no clear pattern
        if case_pattern == 'preserve' or not case_pattern:
            formatted_unit = unit
        elif case_pattern == 'uppercase':
            formatted_unit = unit.upper()
        elif case_pattern == 'lowercase':
            formatted_unit = unit.lower()
        else:
            formatted_unit = unit  # Keep original
        
        # Format numeric value (preserve precision if possible)
        if numeric == int(numeric):
            formatted_numeric = str(int(numeric))
        else:
            # Try to match decimal precision from common format
            formatted_numeric = f"{numeric:.3g}"  # Remove trailing zeros
        
        # Apply spacing pattern
        if spacing_pattern == 'with_space':
            return f"{formatted_numeric} {formatted_unit}"
        else:
            return f"{formatted_numeric}{formatted_unit}"
    

    
    def _find_partial_matches(self, input_value: str, search_df: pd.DataFrame, 
                             common_format: Dict) -> List[Dict[str, Any]]:
        """Find partial matches according to specifications
        
        RULE: When part of the input string matches a preset value
        Examples: substring, rearranged words, overlapping tokens
        Status = Partial Match
        Must return similarity score + explanation
        """
        matches = []
        preset_values = self._get_preset_values_list(search_df)
        
        if not preset_values:
            return matches
        
        # Limit for performance - process first 1000 values
        preset_values = preset_values[:1000]
        
        # 1. SUBSTRING MATCHES
        # "contains substring"
        substring_matches = self._find_substring_matches_enhanced(input_value, search_df)
        matches.extend(substring_matches)
        
        # 2. REARRANGED WORDS
        # "word rearranged"
        token_matches = self._find_token_matches_enhanced(input_value, search_df)
        matches.extend(token_matches)
        
        # 3. OVERLAPPING TOKENS
        # Similar words or concepts
        fuzzy_matches = self._find_fuzzy_matches_enhanced(input_value, preset_values, search_df)
        matches.extend(fuzzy_matches)
        
        return matches
    
    def _get_preset_values_list(self, search_df: pd.DataFrame) -> List[str]:
        """Extract preset values as list for processing"""
        preset_col = self._get_preset_values_column()
        if not preset_col:
            return []
        
        return search_df[preset_col].dropna().astype(str).tolist()
    
    def _find_substring_matches_enhanced(self, input_value: str, search_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find substring matches with detailed comments"""
        matches = []
        input_lower = input_value.lower().strip()
        preset_col = self._get_preset_values_column()
        
        for idx, row in search_df.iterrows():
            preset_value = str(row[preset_col]) if pd.notna(row[preset_col]) else ""
            if not preset_value:
                continue
                
            preset_lower = preset_value.lower().strip()
            
            # Check for substring relationships
            if input_lower in preset_lower and input_lower != preset_lower:
                # Input is substring of preset
                similarity = len(input_lower) / len(preset_lower)
                comment = "Input contains substring of preset value"
                
            elif preset_lower in input_lower and input_lower != preset_lower:
                # Preset is substring of input
                similarity = len(preset_lower) / len(input_lower)
                comment = "Preset value contains substring of input"
                
            else:
                continue
            
            if similarity >= config.MATCHING_THRESHOLD:
                matches.append({
                    'preset_value': preset_value,
                    'similarity': similarity,
                    'match_type': 'substring',
                    'comment': comment,
                    'suggested_value': preset_value,  # Use exact preset formatting
                    'status': config.MatchStatus.PARTIAL_MATCH,
                    'row_data': row.to_dict()
                })
        
        return matches
    
    def _find_token_matches_enhanced(self, input_value: str, search_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find matches with rearranged words/tokens"""
        matches = []
        input_tokens = set(re.findall(r'\w+', input_value.lower()))
        preset_col = self._get_preset_values_column()
        
        if not input_tokens:
            return matches
        
        for idx, row in search_df.iterrows():
            preset_value = str(row[preset_col]) if pd.notna(row[preset_col]) else ""
            if not preset_value:
                continue
                
            preset_tokens = set(re.findall(r'\w+', preset_value.lower()))
            
            if not preset_tokens:
                continue
            
            # Calculate token overlap (Jaccard similarity)
            common_tokens = input_tokens & preset_tokens
            union_tokens = input_tokens | preset_tokens
            
            if not union_tokens:
                continue
                
            similarity = len(common_tokens) / len(union_tokens)
            
            if similarity >= config.MATCHING_THRESHOLD:
                # Generate specific comment based on token relationship
                if input_tokens == preset_tokens:
                    comment = "Same words in different order"
                elif len(common_tokens) == len(input_tokens):
                    comment = "Input words are subset of preset value"
                elif len(common_tokens) == len(preset_tokens):
                    comment = "Preset words are subset of input value"
                else:
                    overlap_pct = (len(common_tokens) / max(len(input_tokens), len(preset_tokens))) * 100
                    comment = f"Overlapping tokens: {overlap_pct:.0f}% word similarity"
                
                matches.append({
                    'preset_value': preset_value,
                    'similarity': similarity,
                    'match_type': 'token_overlap',
                    'comment': comment,
                    'suggested_value': preset_value,  # Use exact preset formatting
                    'status': config.MatchStatus.PARTIAL_MATCH,
                    'row_data': row.to_dict()
                })
        
        return matches
    
    def _find_fuzzy_matches_enhanced(self, input_value: str, preset_values: List[str], 
                                    search_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Enhanced fuzzy matching with detailed comments"""
        matches = []
        preset_col = self._get_preset_values_column()
        
        # Use multiple fuzzy algorithms
        algorithms = [
            ('ratio', fuzz.ratio, 'character-level differences'),
            ('partial_ratio', fuzz.partial_ratio, 'partial string match'),
            ('token_sort_ratio', fuzz.token_sort_ratio, 'word order differences'),
            ('token_set_ratio', fuzz.token_set_ratio, 'different word sets')
        ]
        
        for algo_name, algo_func, description in algorithms:
            try:
                results = process.extract(
                    input_value, 
                    preset_values, 
                    scorer=algo_func,
                    limit=config.MAX_RESULTS_PER_INPUT
                )
                
                for preset_value, score, _ in results:
                    similarity = score / 100.0
                    
                    if similarity >= config.MATCHING_THRESHOLD:
                        # Find corresponding row
                        matching_rows = search_df[search_df[preset_col] == preset_value]
                        if not matching_rows.empty:
                            row = matching_rows.iloc[0]
                            
                            comment = f"Fuzzy match detected: {description} ({similarity:.0%} similarity)"
                            
                            matches.append({
                                'preset_value': preset_value,
                                'similarity': similarity,
                                'match_type': f'fuzzy_{algo_name}',
                                'comment': comment,
                                'suggested_value': preset_value,  # Use exact preset formatting
                                'status': config.MatchStatus.PARTIAL_MATCH,
                                'row_data': row.to_dict()
                            })
                            
            except Exception:
                continue
        
        return matches

# Alias for backward compatibility
MatchingEngine = EnhancedMatchingEngine

class MatchResult:
    """Container for match results with comparison metadata"""
    
    def __init__(self, input_value: str, matches: List[Dict[str, Any]]):
        self.input_value = input_value
        self.matches = matches
        self.best_match = matches[0] if matches else None
        self.status = self._determine_status()
    
    def _determine_status(self) -> str:
        """Determine overall status for this input"""
        if not self.matches:
            return config.MatchStatus.NOT_FOUND
        
        best_similarity = max(m['similarity'] for m in self.matches)
        
        if best_similarity >= config.EXACT_MATCH_THRESHOLD:
            return config.MatchStatus.EXACT_MATCH
        else:
            return config.MatchStatus.PARTIAL_MATCH
    
    def to_result_rows(self, category: str = None, sub_category: str = None, 
                      attribute_name: str = None) -> List[Dict[str, Any]]:
        """Convert to result rows for display/export with required structure
        
        Output Rules - Each result row must include:
        - Original Input
        - Matched Preset Value(s)
        - Similarity %
        - Comment (e.g., "different format", "unit spacing mismatch", "word rearranged", "contains substring")
        - Suggested Value (normalized to the most common preset format)
        - Status (Exact Match / Partial Match / Not Found)
        """
        if not self.matches:
            # No matches found case
            composite_key = f"{category or 'N/A'} | {sub_category or 'N/A'} | {attribute_name or 'N/A'}"
            return [{
                'Composite Key': composite_key,
                'Category': category or '',
                'Sub-Category': sub_category or '',
                'Attribute Name': attribute_name or '',
                'Original Input': self.input_value,
                'Matched Preset Value': '',
                'Similarity %': 0.0,
                'Comment': 'No matches found above 75% similarity threshold in specified context',
                'Suggested Value': '',
                'Status': config.MatchStatus.NOT_FOUND
            }]
        
        result_rows = []
        for match in self.matches:
            # Extract row data if available
            row_data = match.get('row_data', {})
            match_category = row_data.get('Category', category or '')
            match_sub_category = row_data.get('Sub-Category', sub_category or '')
            match_attribute = row_data.get('Attribute Name', attribute_name or '')
            
            composite_key = f"{match_category} | {match_sub_category} | {match_attribute}"
            
            result_rows.append({
                'Composite Key': composite_key,
                'Category': match_category,
                'Sub-Category': match_sub_category,
                'Attribute Name': match_attribute,
                'Original Input': self.input_value,
                'Matched Preset Value': match['preset_value'],
                'Similarity %': round(match['similarity'] * 100, 1),
                'Comment': match['comment'],
                'Suggested Value': match.get('suggested_value', match['preset_value']),
                'Status': match['status']
            })
        
        return result_rows
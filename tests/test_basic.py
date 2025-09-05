"""
Basic tests for the Preset Comparison Tool
"""
import pytest
import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DataProcessor
from utils.matching_engine import MatchingEngine
import config

class TestDataProcessor:
    """Test data processing functionality"""
    
    def test_normalize_text(self):
        """Test text normalization"""
        # Test basic normalization
        result = DataProcessor.normalize_text("  Hello   World  ")
        assert result == "Hello World"
        
        # Test empty input
        result = DataProcessor.normalize_text("")
        assert result == ""
        
        # Test None input
        result = DataProcessor.normalize_text(None)
        assert result == ""
    
    def test_extract_value_components(self):
        """Test value component extraction"""
        # Test simple value
        result = DataProcessor.extract_value_components("20 kg")
        assert len(result['numbers']) > 0
        assert result['numbers'][0] == 20
        assert 'kg' in result['units']
        
        # Test complex value with condition
        result = DataProcessor.extract_value_components("20 kg @ 30°C")
        assert len(result['numbers']) >= 2
        assert len(result['conditions']) > 0
        
        # Test empty value
        result = DataProcessor.extract_value_components("")
        assert result['original'] == ""
        assert len(result['numbers']) == 0

class TestMatchingEngine:
    """Test matching engine functionality"""
    
    @pytest.fixture
    def sample_preset_data(self):
        """Create sample preset data for testing"""
        data = {
            'Category': ['Electronics', 'Clothing', 'Tools'],
            'Sub-Category': ['Smartphones', 'Shirts', 'Screwdrivers'],
            'Attribute Name': ['Screen Size', 'Material', 'Length'],
            'Preset Values': ['6.1 inches', '100% Cotton', '10 cm']
        }
        return pd.DataFrame(data)
    
    def test_exact_matching(self, sample_preset_data):
        """Test exact matching functionality"""
        engine = MatchingEngine(sample_preset_data)
        
        # Test exact match
        matches = engine.find_matches("6.1 inches")
        assert len(matches) > 0
        
        if matches:
            assert matches[0]['similarity'] >= 0.99  # Should be very high for exact match
    
    def test_fuzzy_matching(self, sample_preset_data):
        """Test fuzzy matching functionality"""
        engine = MatchingEngine(sample_preset_data)
        
        # Test fuzzy match (case difference)
        matches = engine.find_matches("6.1 INCHES")
        assert len(matches) > 0
        
        # Test partial match
        matches = engine.find_matches("cotton")
        # Should find some similarity to "100% Cotton"

class TestConfiguration:
    """Test configuration settings"""
    
    def test_config_values(self):
        """Test that configuration values are reasonable"""
        assert 0 < config.MATCHING_THRESHOLD <= 1.0
        assert config.MAX_RESULTS_PER_INPUT > 0
        assert config.MAX_FILE_SIZE_MB > 0
        assert config.EXACT_MATCH_THRESHOLD > config.MATCHING_THRESHOLD

class TestIntegration:
    """Integration tests"""
    
    def test_sample_data_processing(self):
        """Test processing of sample data"""
        # Create sample input
        input_data = {
            'Structure': ['Electronics'],
            'Full Structure': ['Electronics - Smartphones'],
            'Attribute Name': ['Screen Size'],
            'Attribute value': ['6.1 inch']
        }
        input_df = pd.DataFrame(input_data)
        
        # Process the data
        processed_df = DataProcessor.prepare_for_matching(input_df, 'Attribute value')
        
        assert 'cleaned_value' in processed_df.columns
        assert 'numbers' in processed_df.columns
        assert 'units' in processed_df.columns

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
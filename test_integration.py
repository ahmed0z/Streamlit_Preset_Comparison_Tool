#!/usr/bin/env python3
"""
Integration test for the enhanced matching engine with Streamlit app
"""

import pandas as pd
import sys
import os
from pathlib import Path
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.matching_engine import EnhancedMatchingEngine, MatchResult
from utils.file_handler import FileHandler

def create_sample_input_data():
    """Create sample input data for testing"""
    input_data = {
        'Category': [
            'Electronics', 'Electronics', 'Mechanical', 'Electrical'
        ],
        'Sub-Category': [
            'Capacitors', 'Resistors', 'Fasteners', 'Connectors'
        ],
        'Attribute Name': [
            'Voltage', 'Resistance', 'Weight', 'Current Rating'
        ],
        'Input values': [
            '5V', '100 ohm', '20 kg', '2A'
        ]
    }
    
    return pd.DataFrame(input_data)

def create_sample_preset_data():
    """Create sample preset data for testing"""
    preset_data = {
        'Category': [
            'Electronics', 'Electronics', 'Electronics', 'Electronics',
            'Mechanical', 'Mechanical', 'Electrical', 'Electrical'
        ],
        'Sub-Category': [
            'Capacitors', 'Capacitors', 'Resistors', 'Resistors',
            'Fasteners', 'Fasteners', 'Connectors', 'Connectors'
        ],
        'Attribute Name': [
            'Voltage', 'Voltage', 'Resistance', 'Resistance',
            'Weight', 'Weight', 'Current Rating', 'Current Rating'
        ],
        'Preset values': [
            '5V', '5 V', '100 ohm', '100 Ohm',
            '20 kg', '20kg', '2A', '2 A'
        ]
    }
    
    return pd.DataFrame(preset_data)

def test_end_to_end_workflow():
    """Test the complete workflow that the Streamlit app would use"""
    print("🔄 Testing End-to-End Workflow...")
    
    # Create test data
    input_df = create_sample_input_data()
    preset_df = create_sample_preset_data()
    
    # Initialize enhanced matching engine
    matching_engine = EnhancedMatchingEngine(preset_df)
    
    print(f"✅ Created test data: {len(input_df)} input rows, {len(preset_df)} preset rows")
    print(f"✅ Initialized EnhancedMatchingEngine")
    
    # Process each input row (mimics app.py logic)
    all_results = []
    
    input_column = 'Input values'
    
    for i, (idx, row) in enumerate(input_df.iterrows()):
        # Extract context information
        category = str(row.get('Category', '')) if pd.notna(row.get('Category')) else None
        sub_category = str(row.get('Sub-Category', '')) if pd.notna(row.get('Sub-Category')) else None
        attribute_name = str(row.get('Attribute Name', '')) if pd.notna(row.get('Attribute Name')) else None
        input_value = str(row.get(input_column, '')) if pd.notna(row.get(input_column)) else None
        
        if not input_value or not category or not attribute_name:
            continue
        
        print(f"🔍 Processing: {input_value} in {category} > {sub_category} > {attribute_name}")
        
        # Find matches using composite key (same as app)
        matches = matching_engine.find_matches(
            input_value,
            category=category,
            sub_category=sub_category, 
            attribute_name=attribute_name
        )
        
        # Create match result
        match_result = MatchResult(input_value, matches)
        
        # Convert to result rows with context information (enhanced API)
        result_rows = match_result.to_result_rows(category, sub_category, attribute_name)
        
        all_results.extend(result_rows)
        
        # Show individual results
        for result_row in result_rows:
            status = result_row['Status']
            similarity = result_row['Similarity %']
            matched_value = result_row['Matched Preset Value']
            comment = result_row['Comment']
            
            print(f"  ✅ {status}: {matched_value} ({similarity}%) - {comment}")
    
    print(f"\n📊 Final Results Summary:")
    print(f"Total result rows: {len(all_results)}")
    
    # Create results DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        # Analyze results
        exact_matches = len(results_df[results_df['Status'] == config.MatchStatus.EXACT_MATCH])
        partial_matches = len(results_df[results_df['Status'] == config.MatchStatus.PARTIAL_MATCH])
        not_found = len(results_df[results_df['Status'] == config.MatchStatus.NOT_FOUND])
        
        print(f"  Exact Matches: {exact_matches}")
        print(f"  Partial Matches: {partial_matches}")
        print(f"  Not Found: {not_found}")
        
        # Check required columns
        required_columns = [
            'Composite Key', 'Category', 'Sub-Category', 'Attribute Name',
            'Original Input', 'Matched Preset Value', 'Similarity %',
            'Comment', 'Suggested Value', 'Status'
        ]
        
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        
        if not missing_columns:
            print("✅ All required output columns present")
        else:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        # Check data quality
        all_above_threshold = all(
            results_df['Similarity %'] >= 75.0 
            for idx, row in results_df.iterrows() 
            if row['Status'] != config.MatchStatus.NOT_FOUND
        )
        
        if all_above_threshold:
            print("✅ All matches above 75% similarity threshold")
        else:
            print("❌ Some matches below 75% threshold found")
        
        # Show sample results
        print(f"\n📋 Sample Results:")
        for idx, row in results_df.head(3).iterrows():
            print(f"  Input: '{row['Original Input']}' -> '{row['Matched Preset Value']}' ({row['Similarity %']}%)")
            print(f"    Status: {row['Status']}, Comment: {row['Comment']}")
        
        return True
    else:
        print("❌ No results generated")
        return False

def test_output_format_compliance():
    """Test that output format meets all requirements"""
    print("\n🔍 Testing Output Format Compliance...")
    
    preset_df = create_sample_preset_data()
    engine = EnhancedMatchingEngine(preset_df)
    
    # Test various input scenarios
    test_cases = [
        # Exact match
        ('5V', 'Electronics', 'Capacitors', 'Voltage'),
        # Unit difference 
        ('5 volts', 'Electronics', 'Capacitors', 'Voltage'),
        # No match
        ('xyz', 'Electronics', 'Capacitors', 'Voltage'),
    ]
    
    all_valid = True
    
    for input_val, category, sub_cat, attr_name in test_cases:
        matches = engine.find_matches(input_val, category, sub_cat, attr_name)
        match_result = MatchResult(input_val, matches)
        result_rows = match_result.to_result_rows(category, sub_cat, attr_name)
        
        for result_row in result_rows:
            # Check required fields
            required_fields = {
                'Original Input': str,
                'Matched Preset Value': str,
                'Similarity %': (int, float),
                'Comment': str,
                'Suggested Value': str,
                'Status': str
            }
            
            for field, expected_type in required_fields.items():
                if field not in result_row:
                    print(f"❌ Missing field: {field}")
                    all_valid = False
                elif not isinstance(result_row[field], expected_type):
                    print(f"❌ Wrong type for {field}: expected {expected_type}, got {type(result_row[field])}")
                    all_valid = False
            
            # Check status values
            valid_statuses = [config.MatchStatus.EXACT_MATCH, config.MatchStatus.PARTIAL_MATCH, config.MatchStatus.NOT_FOUND]
            if result_row['Status'] not in valid_statuses:
                print(f"❌ Invalid status: {result_row['Status']}")
                all_valid = False
            
            # Check similarity percentage range
            similarity = result_row['Similarity %']
            if not (0 <= similarity <= 100):
                print(f"❌ Similarity % out of range: {similarity}")
                all_valid = False
    
    if all_valid:
        print("✅ All output format requirements met")
    
    return all_valid

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Running Integration Tests for Enhanced Matching Engine")
    print("=" * 70)
    
    tests = [
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Output Format Compliance", test_output_format_compliance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {str(e)}")
    
    print("\n" + "=" * 70)
    print(f"📊 INTEGRATION TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The enhanced matching engine is ready for production.")
        return True
    else:
        print("⚠️  Some integration tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
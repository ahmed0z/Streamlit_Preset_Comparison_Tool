#!/usr/bin/env python3
"""
Test script for enhanced case-sensitive formatting detection
Tests the specific example provided by user
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.matching_engine import EnhancedMatchingEngine, MatchResult

def test_case_sensitive_formatting():
    """Test case-sensitive formatting with the user's specific example"""
    print("🔍 Testing Case-Sensitive Format Detection")
    print("=" * 60)
    
    # Create test data with the user's specific example
    preset_data = {
        'Category': ['Electronics', 'Electronics'],
        'Sub-Category': ['Components', 'Components'],
        'Attribute Name': ['Dimensions', 'Dimensions'],
        'Preset values': [
            '2.80" L x 0.70" W (71.1mm x 17.8mm)',
            '2.8" L x 0.70" W (71.1 mm x 17.8m)'  # Different format
        ]
    }
    
    input_data = {
        'Category': ['Electronics'],
        'Sub-Category': ['Components'], 
        'Attribute Name': ['Dimensions'],
        'Input values': ['2.8" L x 0.70" W (71.1 mm x 17.8m)']  # User's example
    }
    
    preset_df = pd.DataFrame(preset_data)
    input_df = pd.DataFrame(input_data)
    
    print("📋 Test Data:")
    print(f"Input: {input_df.iloc[0]['Input values']}")
    print(f"Preset 1: {preset_df.iloc[0]['Preset values']}")
    print(f"Preset 2: {preset_df.iloc[1]['Preset values']}")
    print()
    
    # Initialize enhanced matching engine
    engine = EnhancedMatchingEngine(preset_df)
    
    # Test matching
    input_value = input_df.iloc[0]['Input values']
    category = input_df.iloc[0]['Category']
    sub_category = input_df.iloc[0]['Sub-Category']
    attribute_name = input_df.iloc[0]['Attribute Name']
    
    print(f"🔍 Processing: {input_value}")
    print(f"Context: {category} > {sub_category} > {attribute_name}")
    print()
    
    # Find matches
    matches = engine.find_matches(
        input_value,
        category=category,
        sub_category=sub_category,
        attribute_name=attribute_name
    )
    
    print(f"📊 Found {len(matches)} matches:")
    print("-" * 40)
    
    for i, match in enumerate(matches, 1):
        print(f"Match {i}:")
        print(f"  Preset Value: '{match['preset_value']}'")
        print(f"  Similarity: {match['similarity']:.1%}")
        print(f"  Match Type: {match['match_type']}")
        print(f"  Status: {match['status']}")
        print(f"  Comment: {match['comment']}")
        print(f"  Suggested Value: '{match.get('suggested_value', match['preset_value'])}'")
        print()
    
    # Create MatchResult and test output format
    match_result = MatchResult(input_value, matches)
    result_rows = match_result.to_result_rows(category, sub_category, attribute_name)
    
    print("📋 Result Rows:")
    print("-" * 40)
    for row in result_rows:
        print(f"Original Input: '{row['Original Input']}'")
        print(f"Matched Preset Value: '{row['Matched Preset Value']}'")
        print(f"Similarity %: {row['Similarity %']}")
        print(f"Comment: {row['Comment']}")
        print(f"Suggested Value: '{row['Suggested Value']}'")
        print(f"Status: {row['Status']}")
        print(f"Composite Key: {row['Composite Key']}")
        print()
    
    return len(matches) > 0

def test_exact_match_case_sensitivity():
    """Test that exact matches are truly case-sensitive"""
    print("🔍 Testing Exact Match Case Sensitivity")
    print("=" * 60)
    
    # Test cases for case sensitivity
    test_cases = [
        # (input, preset, should_be_exact)
        ('5V', '5V', True),           # Exact match
        ('5V', '5v', False),          # Case difference
        ('5 V', '5 V', True),         # Exact with space
        ('5 V', '5V', False),         # Space difference
        ('100 Ohm', '100 Ohm', True), # Exact match
        ('100 ohm', '100 Ohm', False), # Case difference
        ('100ohm', '100 Ohm', False), # Space and case difference
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_val, preset_val, should_be_exact in test_cases:
        # Create minimal test data
        preset_df = pd.DataFrame({
            'Category': ['Test'],
            'Sub-Category': ['Test'],
            'Attribute Name': ['Test'],
            'Preset values': [preset_val]
        })
        
        engine = EnhancedMatchingEngine(preset_df)
        matches = engine.find_matches(input_val, 'Test', 'Test', 'Test')
        
        is_exact = len(matches) > 0 and matches[0]['status'] == config.MatchStatus.EXACT_MATCH
        
        if should_be_exact == is_exact:
            print(f"✅ '{input_val}' vs '{preset_val}': {matches[0]['status'] if matches else 'No match'}")
            passed += 1
        else:
            expected = "Exact Match" if should_be_exact else "Not Exact"
            actual = matches[0]['status'] if matches else 'No match'
            print(f"❌ '{input_val}' vs '{preset_val}': Expected {expected}, got {actual}")
    
    print(f"\nCase Sensitivity Tests: {passed}/{total} passed")
    return passed == total

def test_suggested_value_formatting():
    """Test that suggested values preserve exact preset formatting"""
    print("🔍 Testing Suggested Value Formatting")
    print("=" * 60)
    
    test_cases = [
        # (input, preset, description)
        ('5v', '5V', 'Case preservation'),
        ('5 v', '5V', 'Space and case'),
        ('100ohm', '100 Ohm', 'Space and case in unit'),
        ('2.8"L x 0.70"W', '2.80" L x 0.70" W', 'Dimension formatting'),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_val, preset_val, description in test_cases:
        preset_df = pd.DataFrame({
            'Category': ['Test'],
            'Sub-Category': ['Test'],
            'Attribute Name': ['Test'],
            'Preset values': [preset_val]
        })
        
        engine = EnhancedMatchingEngine(preset_df)
        matches = engine.find_matches(input_val, 'Test', 'Test', 'Test')
        
        if matches:
            suggested = matches[0].get('suggested_value', matches[0]['preset_value'])
            if suggested == preset_val:
                print(f"✅ {description}: '{input_val}' → '{suggested}'")
                passed += 1
            else:
                print(f"❌ {description}: '{input_val}' → '{suggested}' (expected '{preset_val}')")
        else:
            print(f"❌ {description}: No matches found for '{input_val}'")
    
    print(f"\nSuggested Value Tests: {passed}/{total} passed")
    return passed == total

def run_formatting_tests():
    """Run all formatting tests"""
    print("🚀 Running Enhanced Case-Sensitive Formatting Tests")
    print("=" * 70)
    
    tests = [
        ("Case-Sensitive Format Detection", test_case_sensitive_formatting),
        ("Exact Match Case Sensitivity", test_exact_match_case_sensitivity),
        ("Suggested Value Formatting", test_suggested_value_formatting),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 50)
        
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
    print(f"📊 FORMATTING TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All formatting tests passed! Case-sensitive matching is working correctly.")
        return True
    else:
        print("⚠️  Some formatting tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_formatting_tests()
    sys.exit(0 if success else 1)
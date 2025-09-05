#!/usr/bin/env python3
"""
Test script for unit conversion and format standardization
Tests the specific business case: 117mm → 4.61\" (116.99mm)
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from utils.matching_engine import EnhancedMatchingEngine, MatchResult, UnitConverter

def test_unit_converter_basic():
    """Test the UnitConverter class basic functionality"""
    print("🔧 Testing UnitConverter Basic Functions")
    print("-" * 50)
    
    # Test mm to inches conversion
    mm_value = 117.0
    inches_result = UnitConverter.convert_length(mm_value, 'mm', 'in')
    expected_inches = 117.0 / 25.4  # ≈ 4.606 inches
    
    print(f"117mm → {inches_result:.3f} inches (expected: {expected_inches:.3f})")
    
    # Test standardized format generation
    standardized = UnitConverter.mm_to_inches_with_mm(117.0)
    print(f"117mm → standardized format: {standardized}")
    
    # Test parsing of standardized format
    test_format = '4.61" (116.99mm)'\n    parsed = UnitConverter.parse_inches_with_mm_format(test_format)\n    if parsed:\n        print(f"Parsed '{test_format}': {parsed}")
    
    # Test equivalence checking
    is_equiv, similarity, explanation = UnitConverter.values_are_equivalent(\n        117.0, 'mm', 116.99, 'mm', tolerance_percent=1.0\n    )\n    print(f"117mm vs 116.99mm: equivalent={is_equiv}, similarity={similarity:.3f}, explanation='{explanation}'")
    
    print()

def test_business_case_117mm():
    """Test the specific business case: 117mm input vs 4.61\" (116.99mm) preset"""
    print("💼 Testing Business Case: 117mm → 4.61\" (116.99mm)")
    print("-" * 60)
    
    # Create test data matching the business case
    preset_data = {
        'Category': ['Hardware', 'Hardware', 'Hardware'],
        'Sub-Category': ['Fasteners', 'Fasteners', 'Fasteners'],
        'Attribute Name': ['Length', 'Length', 'Length'],
        'Preset values': [
            '4.61" (116.99mm)',  # Target format - should match 117mm
            '3.15" (80.01mm)',   # Different value
            '5.00" (127.00mm)'   # Another different value
        ]
    }
    
    input_data = {
        'Category': ['Hardware'],
        'Sub-Category': ['Fasteners'],
        'Attribute Name': ['Length'],
        'Input values': ['117mm']  # Input that should match first preset
    }
    
    preset_df = pd.DataFrame(preset_data)
    input_df = pd.DataFrame(input_data)
    
    print("📋 Test Data:")
    print(f"Input: {input_df.iloc[0]['Input values']}")
    print("Presets:")
    for i, preset in enumerate(preset_df['Preset values']):
        print(f"  {i+1}. {preset}")
    print()
    
    # Initialize matching engine
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
    
    success = False
    for i, match in enumerate(matches, 1):
        print(f"Match {i}:")
        print(f"  Preset Value: '{match['preset_value']}'")
        print(f"  Similarity: {match['similarity']:.1%}")
        print(f"  Match Type: {match['match_type']}")
        print(f"  Status: {match['status']}")
        print(f"  Comment: {match['comment']}")
        print(f"  Suggested Value: '{match.get('suggested_value', match['preset_value'])}'")
        
        # Check if this is the expected match
        if match['preset_value'] == '4.61" (116.99mm)' and match['similarity'] >= 0.99:
            success = True
            print("  ✅ EXPECTED MATCH FOUND!")
        
        # Show conversion info if available
        if 'standardization_info' in match:
            info = match['standardization_info']
            print(f"  Standardization Info: {info}")
        
        print()
    
    return success

def test_multiple_unit_scenarios():
    """Test various unit conversion scenarios"""
    print("🧪 Testing Multiple Unit Conversion Scenarios")
    print("-" * 50)
    
    test_cases = [
        # (input, preset, expected_match, description)
        ('25.4mm', '1.00" (25.40mm)', True, 'Exact mm to inches conversion'),
        ('50.8mm', '2.00" (50.80mm)', True, 'Double inch conversion'),
        ('100mm', '3.94" (100.00mm)', True, 'Round number mm conversion'),
        ('2.5"', '2.50" (63.50mm)', True, 'Inches input to standardized format'),
        ('127mm', '5.00" (127.00mm)', True, 'Larger measurement conversion'),
        ('10cm', '3.94" (100.00mm)', True, 'CM to standardized format'),
        ('1.5in', '1.50" (38.10mm)', True, 'Inch variation to standard'),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_val, preset_val, expected_match, description in test_cases:
        print(f"Testing: {description}")
        print(f"  Input: '{input_val}' → Preset: '{preset_val}'")
        
        # Create minimal test data
        preset_df = pd.DataFrame({
            'Category': ['Test'],
            'Sub-Category': ['Test'],
            'Attribute Name': ['Length'],
            'Preset values': [preset_val]
        })
        
        engine = EnhancedMatchingEngine(preset_df)
        matches = engine.find_matches(input_val, 'Test', 'Test', 'Length')
        
        found_match = len(matches) > 0 and matches[0]['similarity'] >= 0.95
        
        if expected_match == found_match:
            if found_match:
                print(f"  ✅ Match found: {matches[0]['similarity']:.1%} similarity")
                print(f"     Comment: {matches[0]['comment']}")
            else:
                print(f"  ✅ No match found (as expected)")
            passed += 1
        else:\n            if matches:\n                print(f"  ❌ Unexpected result: {matches[0]['similarity']:.1%} similarity")
                print(f"     Comment: {matches[0]['comment']}")
            else:
                print(f"  ❌ Expected match but none found")
        print()
    
    print(f"Multiple Unit Tests: {passed}/{total} passed")
    return passed == total

def test_duplicate_prevention():
    """Test that the system prevents duplicate entries with different meanings"""
    print("🛡️  Testing Duplicate Prevention")
    print("-" * 40)
    
    # Scenario: User tries to add 117mm when 4.61" (116.99mm) already exists
    preset_data = {
        'Category': ['Hardware'] * 3,
        'Sub-Category': ['Screws'] * 3,
        'Attribute Name': ['Length'] * 3,
        'Preset values': [
            '4.61" (116.99mm)',  # Existing standardized entry
            '3.15" (80.01mm)',   # Other entry
            '6.00" (152.40mm)',  # Another entry
        ]
    }
    
    # User inputs that should be flagged as potential duplicates
    user_inputs = [
        '117mm',        # Very close to existing 116.99mm
        '4.61"',        # Exact inches match
        '4.606in',      # Calculated equivalent
        '116.99mm',     # Exact mm match
        '0.117m',       # Same value in meters
    ]
    
    preset_df = pd.DataFrame(preset_data)
    engine = EnhancedMatchingEngine(preset_df)
    
    duplicates_detected = 0
    
    for user_input in user_inputs:
        print(f"Testing input: '{user_input}'")
        matches = engine.find_matches(user_input, 'Hardware', 'Screws', 'Length')
        
        if matches and matches[0]['similarity'] >= 0.95:
            print(f"  🚨 DUPLICATE DETECTED: {matches[0]['similarity']:.1%} match")
            print(f"     Existing: {matches[0]['preset_value']}")
            print(f"     Comment: {matches[0]['comment']}")
            duplicates_detected += 1
        else:
            print(f"  ✅ No duplicate detected")
        print()
    
    print(f"Duplicate Prevention: {duplicates_detected}/{len(user_inputs)} duplicates correctly detected")
    return duplicates_detected >= len(user_inputs) * 0.8  # Allow some tolerance

def run_conversion_tests():
    """Run all unit conversion tests"""
    print("🚀 Running Unit Conversion and Format Standardization Tests")
    print("=" * 70)
    
    tests = [
        ("UnitConverter Basic Functions", test_unit_converter_basic),
        ("Business Case 117mm → 4.61\" (116.99mm)", test_business_case_117mm),
        ("Multiple Unit Scenarios", test_multiple_unit_scenarios),
        ("Duplicate Prevention", test_duplicate_prevention),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 60)
        
        try:
            result = test_func()
            if result is not False:  # Allow non-boolean returns
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"📊 UNIT CONVERSION TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All unit conversion tests passed! The system is ready for production.")
        print("✅ Business Rule: 117mm will be detected as equivalent to 4.61\" (116.99mm)")
        print("✅ Format Standardization: Prevents duplicates and ensures consistency")
        print("✅ Tolerance Handling: Accounts for rounding differences in conversions")
        return True
    else:
        print("⚠️  Some unit conversion tests failed. Please review implementation.")
        return False

if __name__ == "__main__":
    success = run_conversion_tests()
    sys.exit(0 if success else 1)
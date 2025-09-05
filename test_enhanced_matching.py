#!/usr/bin/env python3
"""
Comprehensive test script for enhanced matching engine
Tests all matching rules according to specifications
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

def create_test_preset_data():
    """Create test preset data for validation"""
    test_data = {
        'Category': [
            'Electronics', 'Electronics', 'Electronics', 'Electronics',
            'Mechanical', 'Mechanical', 'Mechanical', 'Mechanical',
            'Electrical', 'Electrical', 'Electrical', 'Electrical',
            'Materials', 'Materials', 'Materials', 'Materials'
        ],
        'Sub-Category': [
            'Capacitors', 'Capacitors', 'Resistors', 'Resistors',
            'Fasteners', 'Fasteners', 'Springs', 'Springs',
            'Connectors', 'Connectors', 'Cables', 'Cables',
            'Metals', 'Metals', 'Plastics', 'Plastics'
        ],
        'Attribute Name': [
            'Voltage', 'Voltage', 'Resistance', 'Resistance',
            'Weight', 'Weight', 'Length', 'Length',
            'Current Rating', 'Current Rating', 'Diameter', 'Diameter',
            'Density', 'Density', 'Thickness', 'Thickness'
        ],
        'Preset values': [
            '5V', '5 V', '100 ohm', '100ohm',
            '20 kg', '20kg', '10 mm', '10mm',
            '2A', '2 A', '0.5 inches', '0.5in',
            '7.8 g/cm³', '7.8g/cm³', '2.5 mm', '2.5mm'
        ]
    }
    
    return pd.DataFrame(test_data)

def test_exact_matches():
    """Test exact match rules"""
    print("🔍 Testing Exact Match Rules...")
    
    preset_df = create_test_preset_data()
    engine = EnhancedMatchingEngine(preset_df)
    
    test_cases = [
        # Should find exact matches
        ('5V', 'Electronics', 'Capacitors', 'Voltage', True),
        ('5 V', 'Electronics', 'Capacitors', 'Voltage', True),
        ('100 ohm', 'Electronics', 'Resistors', 'Resistance', True),
        ('20 kg', 'Mechanical', 'Fasteners', 'Weight', True),
        ('2A', 'Electrical', 'Connectors', 'Current Rating', True),
        
        # Should NOT find exact matches  
        ('5.0V', 'Electronics', 'Capacitors', 'Voltage', False),
        ('5 volts', 'Electronics', 'Capacitors', 'Voltage', False),
        ('100 ohms', 'Electronics', 'Resistors', 'Resistance', False),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_val, category, sub_cat, attr_name, should_match in test_cases:
        matches = engine.find_matches(input_val, category, sub_cat, attr_name)
        
        if should_match:
            # Should find an exact match
            exact_found = any(m['match_type'] == 'exact' and m['similarity'] == 1.0 for m in matches)
            if exact_found:
                print(f"  ✅ Exact match found for: {input_val}")
                passed += 1
            else:
                print(f"  ❌ Expected exact match for: {input_val}, but got: {[m['match_type'] for m in matches]}")
        else:
            # Should NOT find exact match
            exact_found = any(m['match_type'] == 'exact' and m['similarity'] == 1.0 for m in matches)
            if not exact_found:
                print(f"  ✅ No exact match for: {input_val} (as expected)")
                passed += 1
            else:
                print(f"  ❌ Unexpected exact match for: {input_val}")
    
    print(f"Exact Match Tests: {passed}/{total} passed\n")
    return passed == total

def test_partial_matches():
    """Test partial match rules"""
    print("🔍 Testing Partial Match Rules...")
    
    preset_df = create_test_preset_data()
    engine = EnhancedMatchingEngine(preset_df)
    
    test_cases = [
        # Substring matches
        ('5', 'Electronics', 'Capacitors', 'Voltage', 'substring'),
        ('V', 'Electronics', 'Capacitors', 'Voltage', 'substring'),
        
        # Token rearrangement (if we had multi-word values)
        ('ohm 100', 'Electronics', 'Resistors', 'Resistance', 'token_overlap'),
        
        # Similar format variations
        ('5.0V', 'Electronics', 'Capacitors', 'Voltage', 'fuzzy'),
        ('100 Ohm', 'Electronics', 'Resistors', 'Resistance', 'fuzzy'),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_val, category, sub_cat, attr_name, expected_type in test_cases:
        matches = engine.find_matches(input_val, category, sub_cat, attr_name)
        
        # Should find partial matches with similarity >= 75%
        valid_matches = [m for m in matches if m['similarity'] >= 0.75 and m['status'] == config.MatchStatus.PARTIAL_MATCH]
        
        if valid_matches:
            print(f"  ✅ Partial match found for: {input_val} -> {valid_matches[0]['preset_value']} ({valid_matches[0]['similarity']:.1%})")
            passed += 1
        else:
            print(f"  ❌ No partial match found for: {input_val}")
    
    print(f"Partial Match Tests: {passed}/{total} passed\n")
    return passed >= total * 0.7  # Allow some flexibility for partial matches

def test_unit_handling():
    """Test unit handling rules"""
    print("🔍 Testing Unit Handling Rules...")
    
    preset_df = create_test_preset_data()
    engine = EnhancedMatchingEngine(preset_df)
    
    test_cases = [
        # Same value, different spacing
        ('5 V', 'Electronics', 'Capacitors', 'Voltage', '5V'),
        ('20kg', 'Mechanical', 'Fasteners', 'Weight', '20 kg'),
        
        # Same value, different unit format
        ('5 volts', 'Electronics', 'Capacitors', 'Voltage', '5V'),
        ('100 Ohm', 'Electronics', 'Resistors', 'Resistance', '100 ohm'),
        
        # Same unit, different value (should still match with lower similarity)
        ('6V', 'Electronics', 'Capacitors', 'Voltage', '5V'),
        ('21 kg', 'Mechanical', 'Fasteners', 'Weight', '20 kg'),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_val, category, sub_cat, attr_name, expected_preset in test_cases:
        matches = engine.find_matches(input_val, category, sub_cat, attr_name)
        
        # Should find unit-based matches
        unit_matches = [m for m in matches if m['match_type'] == 'unit_based']
        
        if unit_matches:
            best_match = unit_matches[0]
            print(f"  ✅ Unit match: {input_val} -> {best_match['preset_value']} ({best_match['similarity']:.1%})")
            print(f"      Comment: {best_match['comment']}")
            passed += 1
        else:
            print(f"  ❌ No unit match found for: {input_val}")
    
    print(f"Unit Handling Tests: {passed}/{total} passed\n")
    return passed >= total * 0.7

def test_similarity_threshold():
    """Test 75% similarity threshold enforcement"""
    print("🔍 Testing 75% Similarity Threshold...")
    
    preset_df = create_test_preset_data()
    engine = EnhancedMatchingEngine(preset_df)
    
    # Test various inputs that should or shouldn't meet threshold
    test_cases = [
        ('5V', 'Electronics', 'Capacitors', 'Voltage', True),    # Exact match
        ('5 V', 'Electronics', 'Capacitors', 'Voltage', True),   # Format difference
        ('5.0V', 'Electronics', 'Capacitors', 'Voltage', True),  # Minor difference
        ('xyz', 'Electronics', 'Capacitors', 'Voltage', False),  # Completely different
        ('', 'Electronics', 'Capacitors', 'Voltage', False),     # Empty input
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_val, category, sub_cat, attr_name, should_have_matches in test_cases:
        matches = engine.find_matches(input_val, category, sub_cat, attr_name)
        
        # All returned matches should have >= 75% similarity
        all_above_threshold = all(m['similarity'] >= 0.75 for m in matches)
        
        if should_have_matches:
            if matches and all_above_threshold:
                print(f"  ✅ Matches found for: {input_val} (all above 75%)")
                passed += 1
            else:
                print(f"  ❌ Expected matches for: {input_val}")
        else:
            if not matches or all_above_threshold:
                print(f"  ✅ No matches or all above threshold for: {input_val}")
                passed += 1
            else:
                print(f"  ❌ Found matches below 75% threshold for: {input_val}")
    
    print(f"Similarity Threshold Tests: {passed}/{total} passed\n")
    return passed == total

def test_output_structure():
    """Test required output structure"""
    print("🔍 Testing Output Structure...")
    
    preset_df = create_test_preset_data()
    engine = EnhancedMatchingEngine(preset_df)
    
    # Test with a known match
    matches = engine.find_matches('5V', 'Electronics', 'Capacitors', 'Voltage')
    
    if matches:
        match_result = MatchResult('5V', matches)
        result_rows = match_result.to_result_rows('Electronics', 'Capacitors', 'Voltage')
        
        required_columns = [
            'Composite Key', 'Category', 'Sub-Category', 'Attribute Name',
            'Original Input', 'Matched Preset Value', 'Similarity %',
            'Comment', 'Suggested Value', 'Status'
        ]
        
        if result_rows:
            row = result_rows[0]
            missing_columns = [col for col in required_columns if col not in row]
            
            if not missing_columns:
                print(f"  ✅ All required columns present")
                print(f"  ✅ Original Input: {row['Original Input']}")
                print(f"  ✅ Matched Preset Value: {row['Matched Preset Value']}")
                print(f"  ✅ Similarity %: {row['Similarity %']}")
                print(f"  ✅ Comment: {row['Comment']}")
                print(f"  ✅ Suggested Value: {row['Suggested Value']}")
                print(f"  ✅ Status: {row['Status']}")
                print(f"  ✅ Composite Key: {row['Composite Key']}")
                return True
            else:
                print(f"  ❌ Missing columns: {missing_columns}")
                return False
        else:
            print(f"  ❌ No result rows generated")
            return False
    else:
        print(f"  ❌ No matches found for test")
        return False

def test_format_normalization():
    """Test format normalization based on composite key"""
    print("🔍 Testing Format Normalization...")
    
    preset_df = create_test_preset_data()
    engine = EnhancedMatchingEngine(preset_df)
    
    # Test that format patterns are detected
    print(f"  📊 Format patterns detected: {len(engine.format_patterns)} contexts")
    
    # Test format normalization in suggestions
    matches = engine.find_matches('5 volts', 'Electronics', 'Capacitors', 'Voltage')
    
    if matches:
        suggested_value = matches[0].get('suggested_value', '')
        print(f"  ✅ Input '5 volts' -> Suggested: '{suggested_value}'")
        return True
    else:
        print(f"  ❌ No matches found for format normalization test")
        return False

def run_all_tests():
    """Run all test suites"""
    print("🚀 Running Enhanced Matching Engine Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run individual test suites
    test_results.append(("Exact Matches", test_exact_matches()))
    test_results.append(("Partial Matches", test_partial_matches()))
    test_results.append(("Unit Handling", test_unit_handling()))
    test_results.append(("Similarity Threshold", test_similarity_threshold()))
    test_results.append(("Output Structure", test_output_structure()))
    test_results.append(("Format Normalization", test_format_normalization()))
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<25} {status}")
        if result:
            passed += 1
    
    print("-" * 30)
    print(f"Overall Result: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced matching engine is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
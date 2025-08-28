#!/usr/bin/env python3
"""
Test script to demonstrate accuracy parsing functionality.
This script shows how different output formats are parsed to extract accuracy values.
"""

def parse_accuracy_from_output(output):
    """
    Parse accuracy from the command output.
    
    Args:
        output: Command output string
        
    Returns:
        float: Parsed accuracy or None if not found
    """
    try:
        # Look for accuracy patterns in the output
        lines = output.split('\n')
        
        # Common accuracy patterns to look for (in order of preference)
        accuracy_patterns = [
            'Top-1 Accuracy:',
            'Accuracy:',
            'Top1:',
            'top1_acc:',
            'test_acc:'
        ]
        
        for line in lines:
            line = line.strip()
            for pattern in accuracy_patterns:
                if pattern in line:
                    # Extract numeric value
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        value_part = parts[1].strip()
                        # Extract first number (percentage or decimal)
                        import re
                        numbers = re.findall(r'[\d.]+', value_part)
                        if numbers:
                            accuracy = float(numbers[0])
                            # Convert to percentage if it's a decimal
                            if accuracy < 1.0:
                                accuracy *= 100
                            return accuracy
        
        # If no pattern found, try to find any percentage-like number
        import re
        percentages = re.findall(r'(\d+\.?\d*)\s*%', output)
        if percentages:
            return float(percentages[0])
            
        # Try to find decimal numbers that might be accuracy
        # Look for patterns like "Accuracy=0.8765" or "Accuracy: 0.8765"
        accuracy_matches = re.findall(r'[Aa]ccuracy\s*[=:]\s*(\d+\.\d+)', output)
        if accuracy_matches:
            for acc_str in accuracy_matches:
                acc = float(acc_str)
                if 0 <= acc <= 100:
                    # Convert to percentage if it's a decimal
                    if acc < 1.0:
                        acc *= 100
                    return acc
        
        # Last resort: look for any decimal number that could be accuracy
        # but be more restrictive about the context
        decimals = re.findall(r'(\d+\.\d+)', output)
        if decimals:
            # Look for numbers that are likely accuracy (between 0 and 100)
            # and appear in contexts that suggest they're accuracy values
            for dec in decimals:
                acc = float(dec)
                # Only consider numbers that are reasonable accuracy values
                if 0.1 <= acc <= 100:
                    # Convert to percentage if it's a decimal
                    if acc < 1.0:
                        acc *= 100
                    # Additional check: look for context around this number
                    context_start = max(0, output.find(dec) - 20)
                    context_end = min(len(output), output.find(dec) + 20)
                    context = output[context_start:context_end].lower()
                    
                    # If the context contains accuracy-related words, this is likely accuracy
                    if any(word in context for word in ['acc', 'accuracy', 'top', 'test', 'result']):
                        return acc
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not parse accuracy from output: {e}")
        return None

def test_accuracy_parsing():
    """Test the accuracy parsing with different output formats."""
    
    test_cases = [
        {
            'name': 'Standard Top-1 Accuracy',
            'output': '''
            Training completed.
            Top-1 Accuracy: 76.54%
            Top-5 Accuracy: 92.31%
            ''',
            'expected': 76.54
        },
        {
            'name': 'Decimal Accuracy',
            'output': '''
            Test Results:
            Accuracy: 0.8234
            Loss: 0.4567
            ''',
            'expected': 82.34
        },
        {
            'name': 'Simple Percentage',
            'output': '''
            Final Results:
            78.9%
            ''',
            'expected': 78.9
        },
        {
            'name': 'Multiple Numbers',
            'output': '''
            Epoch 10: Loss=0.1234, Accuracy=0.8765, Learning Rate=0.001
            ''',
            'expected': 87.65
        },
        {
            'name': 'No Accuracy Found',
            'output': '''
            Training started.
            Loss: 0.1234
            Learning Rate: 0.001
            ''',
            'expected': None
        }
    ]
    
    print("Testing Accuracy Parsing Functionality")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Output: {repr(test_case['output'].strip())}")
        
        result = parse_accuracy_from_output(test_case['output'])
        expected = test_case['expected']
        
        if result == expected:
            print(f"✓ PASS: Parsed accuracy = {result}")
        else:
            # Handle floating point precision issues
            if result is not None and expected is not None:
                if abs(result - expected) < 0.01:  # Allow small difference due to floating point
                    print(f"✓ PASS: Parsed accuracy = {result} (close to expected {expected})")
                else:
                    print(f"✗ FAIL: Expected {expected}, got {result}")
            else:
                print(f"✗ FAIL: Expected {expected}, got {result}")
    
    print(f"\n{'=' * 50}")
    print("Accuracy parsing test completed!")

if __name__ == "__main__":
    test_accuracy_parsing()

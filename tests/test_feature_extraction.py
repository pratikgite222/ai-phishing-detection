"""
AI-Based Phishing Website Detection
Test Suite: Live URL Feature Extraction Verification

Tests the feature extraction engine on 5 diverse benchmark URLs:
A. Legitimate-looking: https://www.google.com
B. Legitimate-looking: https://www.amazon.com
C. Suspicious: http://192.168.1.1/login.php
D. Suspicious: http://paypal-login-security.example.com/verify/account
E. URL with query parameters: https://example.com/login?user=test&id=123

Verifies:
- All 54 features extracted
- Exact order matches models/feature_names.json
- Feature vector length is exactly 54
- Zero NaN or Infinite values
- Handles network redirection / timeouts gracefully
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extraction import extract_features, get_expected_feature_names
from src.validate_features import validate_feature_vector


def test_url_extraction():
    test_cases = [
        ("A. Legitimate-looking", "https://www.google.com"),
        ("B. Legitimate-looking", "https://www.amazon.com"),
        ("C. Suspicious (IP & script)", "http://192.168.1.1/login.php"),
        (
            "D. Suspicious (Brand spoofing & keywords)",
            "http://paypal-login-security.example.com/verify/account",
        ),
        (
            "E. URL with query parameters",
            "https://example.com/login?user=test&id=123",
        ),
    ]

    expected_names = get_expected_feature_names("models/feature_names.json")
    print(f"Loaded expected features list: {len(expected_names)} features.")

    print("\n" + "=" * 80)
    print("STARTING LIVE URL FEATURE EXTRACTION TESTS")
    print("=" * 80)

    overall_passed = True

    for label, url in test_cases:
        print(f"\n[{label}]")
        print(f"URL: {url}")

        exception_caught = None
        features_dict = None
        is_valid = False
        errors = []

        try:
            # Test live extraction with network check enabled (1.5s timeout)
            features_dict = extract_features(
                url, check_network=True, network_timeout=1.5, return_type="dict"
            )
            # Run formal validation
            is_valid, errors, _ = validate_feature_vector(
                url, check_network=False
            )
        except Exception as e:
            exception_caught = str(e)
            is_valid = False
            errors.append(f"Exception raised: {e}")

        # Vector Length
        vector_len = len(features_dict) if features_dict else 0
        print(f"Feature Vector Length: {vector_len} (Expected: 54)")

        # Validity Status
        status_str = "VALID (PASSED)" if is_valid else "INVALID (FAILED)"
        print(f"Vector Status: {status_str}")

        if exception_caught:
            print(f"Exception Caught: {exception_caught}")
            overall_passed = False
        else:
            print("Exceptions: None")

        if errors:
            print(f"Validation Errors: {errors}")
            overall_passed = False

        # Print all 54 extracted features cleanly
        if features_dict:
            print("\nExtracted Features (All 54 in exact order):")
            for i, name in enumerate(expected_names, 1):
                val = features_dict.get(name, "MISSING")
                print(f"  {i:2d}. {name:25s} = {val}")

        print("-" * 80)

    print("\n" + "=" * 80)
    if overall_passed:
        print("ALL URL FEATURE EXTRACTION TESTS PASSED SUCCESSFULLY!")
    else:
        print("TEST FAILURES DETECTED!")
    print("=" * 80)

    return overall_passed


if __name__ == "__main__":
    success = test_url_extraction()
    sys.exit(0 if success else 1)

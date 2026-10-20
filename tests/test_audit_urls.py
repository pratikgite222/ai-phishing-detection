"""
AI-Based Phishing Website Detection
Phase 9: Pre-Deployment Audit Test Suite

Tests:
1. Additional URL Diversity (beyond baseline 5 URLs):
   - Legitimate website (wikipedia.org)
   - Suspicious URL with security tokens
   - Public IP address with path
   - Brand spoofing in domain
   - URL shortener
   - Suspicious TLD (.xyz)
   - URL without scheme (e.g., github.com)
2. Error Handling & Edge Cases:
   - Empty string
   - Pure whitespace
   - Malformed URL
   - Unreachable domain
   - Timeout safety
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import predict_url


def run_audit_tests():
    print("=" * 80)
    print("RUNNING PRE-DEPLOYMENT AUDIT & EDGE CASE TEST SUITE")
    print("=" * 80)

    audit_cases = [
        ("1. Legitimate Website", "https://www.wikipedia.org"),
        ("2. Suspicious Keywords", "http://secure-update-account.com/billing/confirm"),
        ("3. Public IP URL", "http://203.0.113.15/secure/login"),
        ("4. Brand Spoofing", "http://apple-support-verify.com/security"),
        ("5. URL Shortener", "https://bit.ly/secure-login-doc"),
        ("6. Suspicious TLD", "http://free-giftcard.xyz"),
        ("7. URL without Scheme", "github.com/torvalds/linux"),
    ]

    print("\n--- A. ADDITIONAL URL INFERENCE TESTS ---")
    for label, url in audit_cases:
        try:
            res = predict_url(url, check_network=True, network_timeout=1.5)
            print(f"[{label}]")
            print(f"  URL:         {url}")
            print(f"  Normalized:  {res['normalized_url']}")
            print(f"  Prediction:  {res['label']} ({res['prediction']})")
            print(f"  Phish Prob:  {res['phishing_probability'] * 100:.2f}%")
            print(f"  Legit Prob:  {res['legitimate_probability'] * 100:.2f}%")
            print(f"  Features:    {res['feature_vector_length']}")
        except Exception as e:
            print(f"[{label}] FAILED with exception: {e}")
        print("-" * 50)

    print("\n--- B. ERROR HANDLING & RESILIENCE TESTS ---")
    error_cases = [
        ("Empty String", ""),
        ("Pure Whitespace", "   \t\n  "),
        ("Malformed String", "not-a-valid-domain---"),
        ("Unreachable Domain", "http://nonexistent-domain-abcxyz123456789.org"),
    ]

    for label, err_input in error_cases:
        print(f"[{label}] Input: {repr(err_input)}")
        try:
            res = predict_url(err_input, check_network=True, network_timeout=1.5)
            print(f"  Result: Handled without crash -> {res['label']} (Phish: {res['phishing_probability']*100:.1f}%)")
        except ValueError as ve:
            print(f"  Result: Safely caught expected ValueError -> {ve}")
        except Exception as e:
            print(f"  Result: Caught exception -> {e}")
        print("-" * 50)

    print("\n" + "=" * 80)
    print("PRE-DEPLOYMENT AUDIT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_audit_tests()

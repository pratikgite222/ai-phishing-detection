"""
AI-Based Phishing Website Detection
Integration Test Suite: End-to-End Prediction Pipeline

Tests the complete flow from raw URL string -> feature extraction -> model inference.
Test Cases:
A: https://www.google.com
B: https://www.amazon.com
C: http://192.168.1.1/login.php
D: http://paypal-login-security.example.com/verify/account
E: https://example.com/login?user=test&id=123

Note: These are integration test cases verifying pipeline connectivity and probability
computation. They do not constitute model accuracy benchmarks on unseen distributions.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import predict_url


def run_prediction_tests():
    test_urls = [
        ("A", "https://www.google.com"),
        ("B", "https://www.amazon.com"),
        ("C", "http://192.168.1.1/login.php"),
        ("D", "http://paypal-login-security.example.com/verify/account"),
        ("E", "https://example.com/login?user=test&id=123"),
    ]

    print("=" * 80)
    print("RUNNING END-TO-END PREDICTION PIPELINE INTEGRATION TESTS")
    print("=" * 80)

    all_passed = True
    results = []

    for label, url in test_urls:
        print(f"\n[Test Case {label}]")
        print(f"URL: {url}")

        exc_str = "None"
        res = None

        try:
            # Run prediction (with network check enabled, 1.5s timeout)
            res = predict_url(url, check_network=True, network_timeout=1.5)
        except Exception as e:
            exc_str = str(e)
            all_passed = False

        if res:
            vec_len = res.get("feature_vector_length")
            pred = res.get("prediction")
            pred_label = res.get("label")
            legit_p = res.get("legitimate_probability")
            phish_p = res.get("phishing_probability")

            print(f"Extracted Vector Length: {vec_len} (Expected: 54)")
            print(f"Prediction (Raw):        {pred}")
            print(f"Predicted Label:         {pred_label}")
            print(f"Legitimate Probability:  {legit_p * 100:.2f}% ({legit_p})")
            print(f"Phishing Probability:    {phish_p * 100:.2f}% ({phish_p})")
            print(f"Exception:               {exc_str}")

            results.append((label, url, vec_len, pred, pred_label, legit_p, phish_p, exc_str))
        else:
            print(f"FAILED with Exception:   {exc_str}")
            results.append((label, url, 0, None, "ERROR", 0.0, 0.0, exc_str))

        print("-" * 80)

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 5 INTEGRATION PREDICTION TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("FAILURES ENCOUNTERED DURING INTEGRATION TESTING!")
    print("=" * 80)

    return all_passed, results


if __name__ == "__main__":
    success, _ = run_prediction_tests()
    sys.exit(0 if success else 1)

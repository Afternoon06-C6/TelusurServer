#!/usr/bin/env python3
"""
Simple test script to validate the async server functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:4789"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_job_status_404():
    """Test job status endpoint with non-existent job"""
    try:
        response = requests.get(f"{BASE_URL}/job/fake-job-id/status", timeout=5)
        print(f"Job status (404 test): {response.status_code}")
        if response.status_code == 404:
            print("✓ 404 response as expected")
            return True
        else:
            print(f"Unexpected response: {response.text}")
            return False
    except Exception as e:
        print(f"Job status test failed: {e}")
        return False

def main():
    print("Testing async investigation server...")
    
    if not test_health():
        print("❌ Health check failed - server might not be running")
        return
    
    if not test_job_status_404():
        print("❌ Job status endpoint test failed")
        return
    
    print("✅ Basic server tests passed!")

if __name__ == "__main__":
    main()

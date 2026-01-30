#!/usr/bin/env python3
"""
Comprehensive test script for TelusurServer video processing
Tests the full workflow without requiring the SwiftUI frontend
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:4789"
TEST_VIDEO_PATH = "video1.mp4"
TEST_UUID = "test-session-123"
TEST_COLOR = "blue"  # Options: black, blue, grey, white

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_health_check():
    """Test 1: Verify server is running and healthy"""
    print_section("Test 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server Status: {data.get('status')}")
            print(f"   YOLO Model: {data.get('yolo_model')}")
            print(f"   Active Jobs: {data.get('active_jobs')}")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Is the server running?")
        print(f"   Make sure to start the server with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_upload_video():
    """Test 2: Upload video for processing"""
    print_section("Test 2: Upload Video for Processing")
    
    # Check if test video exists
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"❌ Test video not found: {TEST_VIDEO_PATH}")
        print(f"   Please ensure video1.mp4 exists in the current directory")
        return None
    
    print(f"Uploading: {TEST_VIDEO_PATH}")
    print(f"UUID: {TEST_UUID}")
    print(f"Target Color: {TEST_COLOR}")
    
    try:
        with open(TEST_VIDEO_PATH, 'rb') as video_file:
            files = {'videos': (TEST_VIDEO_PATH, video_file, 'video/mp4')}
            data = {
                'uuid': TEST_UUID,
                'topColor': TEST_COLOR
            }
            
            response = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"✅ Upload successful!")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {result.get('status')}")
            print(f"   Total Videos: {result.get('total_videos')}")
            return job_id
        else:
            print(f"❌ Upload failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def test_job_status(job_id):
    """Test 3: Monitor job processing status"""
    print_section("Test 3: Monitor Job Status")
    
    if not job_id:
        print("❌ No job ID provided")
        return False
    
    print(f"Polling status for Job ID: {job_id}")
    print("(This may take several minutes depending on video length...)\n")
    
    max_polls = 300  # 5 minutes max
    poll_interval = 2  # seconds
    
    try:
        for i in range(max_polls):
            response = requests.get(f"{BASE_URL}/job/{job_id}/status", timeout=10)
            
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.status_code}")
                return False
            
            status_data = response.json()
            status = status_data.get('status')
            progress = status_data.get('progress', 0)
            processed = status_data.get('processed_videos', 0)
            total = status_data.get('total_videos', 0)
            
            # Print progress update
            progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            print(f"\r[{progress_bar}] {progress*100:.1f}% | Status: {status} | Videos: {processed}/{total}", end="", flush=True)
            
            if status == 'completed':
                print("\n✅ Job completed successfully!")
                return True
            elif status == 'failed':
                error = status_data.get('error_message', 'Unknown error')
                print(f"\n❌ Job failed: {error}")
                return False
            
            time.sleep(poll_interval)
        
        print("\n❌ Job timed out after 5 minutes")
        return False
        
    except Exception as e:
        print(f"\n❌ Status monitoring failed: {e}")
        return False

def test_get_results(job_id):
    """Test 4: Retrieve processing results"""
    print_section("Test 4: Retrieve Results")
    
    if not job_id:
        print("❌ No job ID provided")
        return False
    
    try:
        response = requests.get(f"{BASE_URL}/job/{job_id}/results", timeout=10)
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Results retrieved successfully!")
            print(f"\nMessage: {results.get('message')}")
            print(f"Job ID: {results.get('job_id')}")
            
            processed_files = results.get('processed_files', [])
            print(f"\nProcessed Files: {len(processed_files)}")
            
            for i, file_data in enumerate(processed_files, 1):
                if 'error' in file_data:
                    print(f"\n  File {i}: ERROR - {file_data['error']}")
                    continue
                
                print(f"\n  File {i}:")
                print(f"    Original: {file_data.get('original_name')}")
                print(f"    Processed: {file_data.get('processed_filename')}")
                
                images = file_data.get('images', [])
                print(f"    Detected Individuals: {len(images)}")
                
                for j, img in enumerate(images, 1):
                    start_time = img.get('start_time', 0)
                    end_time = img.get('end_time', 0)
                    duration = end_time - start_time
                    print(f"      Person {j}:")
                    print(f"        Appeared: {start_time:.2f}s")
                    print(f"        Disappeared: {end_time:.2f}s")
                    print(f"        Duration: {duration:.2f}s")
                    print(f"        Start Image: {img.get('start')}")
                    print(f"        End Image: {img.get('end')}")
            
            return True
        else:
            print(f"❌ Failed to retrieve results: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Result retrieval failed: {e}")
        return False

def test_invalid_job():
    """Test 5: Test error handling with invalid job ID"""
    print_section("Test 5: Error Handling")
    
    fake_job_id = "fake-job-id-12345"
    print(f"Testing with invalid job ID: {fake_job_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/job/{fake_job_id}/status", timeout=5)
        
        if response.status_code == 404:
            print("✅ Correctly returned 404 for invalid job ID")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🎬" * 30)
    print("  TelusurServer Video Processing Test Suite")
    print("🎬" * 30)
    
    print(f"\nTest Configuration:")
    print(f"  Server URL: {BASE_URL}")
    print(f"  Test Video: {TEST_VIDEO_PATH}")
    print(f"  Test UUID: {TEST_UUID}")
    print(f"  Target Color: {TEST_COLOR}")
    
    # Test 1: Health Check
    if not test_health_check():
        print("\n❌ Server health check failed. Cannot proceed with tests.")
        print("   Please start the server with: python app.py")
        return
    
    # Test 5: Error Handling (run early since it's quick)
    test_invalid_job()
    
    # Test 2: Upload Video
    job_id = test_upload_video()
    if not job_id:
        print("\n❌ Video upload failed. Cannot proceed with remaining tests.")
        return
    
    # Test 3: Monitor Status
    if not test_job_status(job_id):
        print("\n❌ Job processing failed or timed out.")
        return
    
    # Test 4: Get Results
    test_get_results(job_id)
    
    # Final Summary
    print_section("Test Summary")
    print("✅ All tests completed!")
    print("\nNext steps:")
    print("  1. Check processed video in: ~/Library/Application Support/Telusur/Processed/")
    print("  2. Check cropped images in: ~/Library/Application Support/Telusur/Images/")
    print("  3. Integrate with Telusur macOS SwiftUI app")

if __name__ == "__main__":
    main()

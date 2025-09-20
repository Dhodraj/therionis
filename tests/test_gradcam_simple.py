#!/usr/bin/env python3
"""
Simple API test for GradCAM implementation
"""
import requests
import os
import json

def test_gradcam_api():
    """Test the API with GradCAM implementation"""
    
    print("🔬 Testing GradCAM via API...")
    
    # API endpoint
    url = "http://127.0.0.1:8000/analyze"
    
    # Use an available test image
    image_path = "./storage/original/6dfb271b-092c-4a75-be47-f80eb293c085_Normal-1.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        # Send image to API
        with open(image_path, 'rb') as f:
            files = {'file': ('test_gradcam.png', f, 'image/png')}
            data = {'model_name': 'chexpert'}
            
            print("📤 Sending image to API for GradCAM analysis...")
            response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            
            # Check predictions
            predictions = result.get('predictions', [])
            print(f"📊 Found {len(predictions)} predictions")
            
            if predictions:
                print("🔍 Top predictions:")
                for i, pred in enumerate(predictions[:3]):
                    condition = pred.get('condition', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    print(f"  {i+1}: {condition} -> {confidence:.3f}")
            
            # Check case info
            case_id = result.get('case_id', 'unknown')
            print(f"📋 Case ID: {case_id}")
            
            if case_id != 'unknown':
                annotated_path = f"./storage/annotated/{case_id}_annotated.png"
                if os.path.exists(annotated_path):
                    print(f"✅ Annotated image created: {annotated_path}")
                    return True
                else:
                    print("❌ Annotated image file not found")
                    return False
            else:
                print("❌ No case ID returned")
                return False
            
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_server_logs():
    """Check server logs for GradCAM messages"""
    
    print("\n📋 Checking server logs for GradCAM messages...")
    
    try:
        with open("./server_output.log", "r") as f:
            logs = f.read()
        
        # Look for GradCAM-related messages
        gradcam_lines = [line for line in logs.split('\n') if 'gradcam' in line.lower() or 'heatmap' in line.lower()]
        
        if gradcam_lines:
            print("🔍 GradCAM-related log entries:")
            for line in gradcam_lines[-5:]:  # Show last 5 entries
                print(f"  {line}")
        else:
            print("ℹ️  No specific GradCAM messages found in logs")
        
        # Check for fallback messages
        fallback_lines = [line for line in logs.split('\n') if 'fallback' in line.lower()]
        if fallback_lines:
            print("\n⚠️  Fallback messages:")
            for line in fallback_lines[-3:]:  # Show last 3
                print(f"  {line}")
                
    except Exception as e:
        print(f"❌ Could not read server logs: {e}")

if __name__ == "__main__":
    print("🧪 Simple GradCAM API Test")
    print("=" * 40)
    
    success = test_gradcam_api()
    check_server_logs()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 GradCAM API test PASSED!")
    else:
        print("❌ GradCAM API test FAILED!")
    
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
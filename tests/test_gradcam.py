#!/usr/bin/env python3
"""
Test script to verify the GradCAM implementation works correctly
"""
import sys
import os
import requests
import json
import numpy as np

# Add project root to path
sys.path.append('.')

def test_gradcam_api():
    """Test the API with GradCAM implementation"""
    
    print("Testing GradCAM implementation via API...")
    
    # API endpoint
    url = "http://127.0.0.1:8000/analyze"
    
    # Use one of the existing test images
    image_path = "./storage/original/6dfb271b-092c-4a75-be47-f80eb293c085_Normal-1.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        # Send image to API
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            data = {'model_name': 'chexpert'}
            
            print("📤 Sending image to API for GradCAM analysis...")
            response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            
            # Check predictions
            predictions = result.get('predictions', [])
            print(f"📊 Number of predictions: {len(predictions)}")
            
            if predictions:
                print("🔍 Predictions:")
                for i, pred in enumerate(predictions[:3]):  # Show top 3
                    condition = pred.get('condition', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    print(f"  {i+1}: {condition} -> {confidence:.3f}")
            
            # Check if annotated image was created
            case_id = result.get('case_id', 'unknown')
            annotated_url = result.get('annotated_image_url', '')
            
            print(f"📋 Case ID: {case_id}")
            print(f"🖼️  Annotated image URL: {annotated_url}")
            
            if case_id != 'unknown':
                annotated_path = f"./storage/annotated/{case_id}_annotated.png"
                if os.path.exists(annotated_path):
                    print("✅ Annotated image created successfully")
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
        print("❌ Request timed out - this might indicate the GradCAM is working but taking time")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_gradcam_direct():
    """Test GradCAM implementation directly (without full API)"""
    
    print("\nTesting GradCAM implementation directly...")
    
    try:
        # Import required modules
        from src.models.inference import ModelInferenceEngine
        from src.preprocessing.image_processor import ImagePreprocessor
        import torch
        from PIL import Image
        
        print("📦 Modules imported successfully")
        
        # Initialize components
        engine = ModelInferenceEngine()
        preprocessor = ImagePreprocessor()
        
        print("🔧 Components initialized")
        
        # Load test image
        image_path = "./storage/original/6dfb271b-092c-4a75-be47-f80eb293c085_Normal-1.png"
        
        with open(image_path, 'rb') as f:
            file_content = f.read()
        
        # Preprocess image
        image_array = preprocessor.load_image(file_content, "test.png")
        image_tensor = preprocessor.preprocess_for_inference(image_array, "xray")
        
        print(f"🖼️  Image preprocessed: {image_tensor.shape}")
        
        # Test regular inference
        print("🧠 Testing regular inference...")
        inference_results = engine.predict(image_tensor, "chexpert")
        predictions = inference_results["predictions"]
        
        print(f"✅ Inference completed, {len(predictions)} predictions")
        
        # Test GradCAM generation
        print("🎯 Testing GradCAM generation...")
        heatmap = engine.generate_gradcam_heatmap(image_tensor, "chexpert")
        
        print(f"✅ GradCAM generated with shape: {heatmap.shape}")
        
        # Check heatmap properties
        if heatmap.shape == (224, 224):
            print("✅ Heatmap has correct dimensions")
        else:
            print(f"⚠️  Unexpected heatmap shape: {heatmap.shape}")
        
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        print(f"📈 Heatmap range: [{heatmap_min:.3f}, {heatmap_max:.3f}]")
        
        # Check if it's not just a fallback (fallback has specific pattern)
        if heatmap_max > 0.5001:  # More than fallback max
            print("✅ Appears to be real GradCAM (not fallback)")
            return True
        elif heatmap_max == 0.5:  # Exactly fallback max
            print("⚠️  Appears to be fallback heatmap")
            return False
        else:
            print("✅ GradCAM generated with different values")
            return True
            
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔬 Starting GradCAM Implementation Tests")
    print("=" * 50)
    
    # Test 1: Direct implementation test
    direct_success = test_gradcam_direct()
    
    # Test 2: API integration test
    api_success = test_gradcam_api()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Direct GradCAM Test: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"   API Integration Test: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if direct_success and api_success:
        print("\n🎉 ALL TESTS PASSED! GradCAM implementation is working correctly!")
        return True
    elif direct_success:
        print("\n⚠️  GradCAM works directly but API integration has issues")
        return False
    else:
        print("\n❌ GradCAM implementation needs debugging")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
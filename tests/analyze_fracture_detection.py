#!/usr/bin/env python3
"""
Analyze fracture detection issues in the medical AI model
"""
import sys
import os
import requests
import json

sys.path.append('.')

def analyze_fracture_predictions():
    """Analyze all current cases for fracture predictions"""
    
    print("🔍 Analyzing Fracture Detection Issues")
    print("=" * 50)
    
    # Test images in storage
    test_images = [
        ("0e2ab70d-ce2a-4c67-b8d5-2f81588e0c22_Tuberculosis-1.png", "Tuberculosis"),
        ("184a7ee4-01a6-4dc8-8bdc-6243e68c8c30_Tuberculosis-3.png", "Tuberculosis"), 
        ("2975f77a-bf2e-4b04-a75b-7a1dc1958709_Normal-2.png", "Normal"),
        ("335fcd67-89d5-4471-aa6b-7d333cdf2bda_Normal-3.png", "Normal"),
        ("6f235fae-8a2d-4757-9dd4-cdef01f477a9_Normal-1.png", "Normal"),
        ("bb55ea51-45cc-42c6-9e6f-64eabba73e6c_Tuberculosis-2.png", "Tuberculosis")
    ]
    
    fracture_detections = []
    
    for image_file, expected_type in test_images:
        image_path = f"./storage/original/{image_file}"
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_file}")
            continue
            
        print(f"\n📋 Testing: {image_file} (Expected: {expected_type})")
        
        try:
            # Send to API
            url = "http://127.0.0.1:8000/analyze"
            with open(image_path, 'rb') as f:
                files = {'file': ('test.png', f, 'image/png')}
                response = requests.post(url, files=files, data={'model_name': 'chexpert'}, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                
                # Check for fracture predictions
                fracture_pred = None
                for pred in predictions:
                    if pred.get('condition') == 'Fracture':
                        fracture_pred = pred
                        break
                
                if fracture_pred:
                    confidence = fracture_pred['confidence']
                    rank = fracture_pred['rank']
                    print(f"🚨 FRACTURE DETECTED: {confidence:.1%} confidence (rank {rank})")
                    
                    # This is problematic for Normal and Tuberculosis cases
                    if expected_type in ['Normal', 'Tuberculosis']:
                        print(f"❌ FALSE POSITIVE: {expected_type} case incorrectly flagged as fracture")
                        fracture_detections.append({
                            'file': image_file,
                            'expected': expected_type,
                            'fracture_confidence': confidence,
                            'rank': rank,
                            'all_predictions': predictions
                        })
                else:
                    print(f"✅ No fracture detected")
                    
                # Show top predictions
                print(f"📊 Top predictions:")
                for i, pred in enumerate(predictions[:3]):
                    condition = pred.get('condition', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    print(f"   {i+1}. {condition}: {confidence:.1%}")
                    
            else:
                print(f"❌ API error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing {image_file}: {e}")
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"🚨 FRACTURE DETECTION ANALYSIS SUMMARY")
    print(f"Total false positive fracture detections: {len(fracture_detections)}")
    
    if fracture_detections:
        print(f"\n❌ PROBLEMATIC CASES:")
        for detection in fracture_detections:
            print(f"   • {detection['file']}")
            print(f"     Expected: {detection['expected']}")
            print(f"     Fracture confidence: {detection['fracture_confidence']:.1%}")
            print(f"     Rank: #{detection['rank']}")
            print()
    
    return fracture_detections

def analyze_model_pathologies():
    """Check what pathologies the model can detect"""
    
    print("🔬 Analyzing Model Pathology Categories")
    print("=" * 50)
    
    url = "http://127.0.0.1:8000/models"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models_info = response.json()
            
            for model_name, info in models_info.items():
                print(f"\n📋 Model: {model_name}")
                pathologies = info.get('pathologies', [])
                print(f"Total pathologies: {len(pathologies)}")
                
                print("\n🏥 All pathologies this model can detect:")
                for i, pathology in enumerate(pathologies, 1):
                    marker = "🚨" if pathology == "Fracture" else "🔸"
                    print(f"   {i:2d}. {marker} {pathology}")
                    
        else:
            print(f"❌ Could not get model info: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting model pathologies: {e}")

def main():
    """Run fracture analysis"""
    
    print("🩻 FRACTURE DETECTION ANALYSIS")
    print("Investigating false positive fracture predictions...")
    print()
    
    # Analyze current predictions
    false_positives = analyze_fracture_predictions()
    
    # Check model capabilities
    analyze_model_pathologies()
    
    # Recommendations
    print("\n" + "=" * 50)
    print("🎯 RECOMMENDATIONS:")
    
    if false_positives:
        print("❌ CRITICAL ISSUE: Model is producing false positive fracture detections")
        print("   Recommended actions:")
        print("   1. Increase fracture confidence threshold from current level")
        print("   2. Add medical context filtering for fracture vs other pathologies")
        print("   3. Consider using chest X-ray specific fracture detection criteria")
        print("   4. Review training data for fracture category contamination")
    else:
        print("✅ No fracture false positives detected in current dataset")
    
    return len(false_positives) == 0

if __name__ == "__main__":
    success = main()
    print(f"\nAnalysis result: {'✅ PASS' if success else '❌ ISSUES FOUND'}")
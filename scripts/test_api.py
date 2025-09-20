#!/usr/bin/env python3
"""
Test script for the Tele-Radiology API
This script demonstrates how to use the API endpoints
"""
import requests
import json
import time
from pathlib import Path


class TeleRadiologyAPITester:
    """
    Test client for the Tele-Radiology API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test the health check endpoint"""
        print("🏥 Testing health check...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check successful!")
                print(f"   Status: {data['status']}")
                print(f"   Models loaded: {', '.join(data['models_loaded'])}")
                print(f"   Device: {data['device']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_list_models(self):
        """Test the list models endpoint"""
        print("\n🤖 Testing model listing...")
        
        try:
            response = self.session.get(f"{self.base_url}/models")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Model listing successful!")
                
                for model_name, model_info in data['models'].items():
                    print(f"   Model: {model_name}")
                    print(f"     Name: {model_info['name']}")
                    print(f"     Pathologies: {len(model_info['pathologies'])}")
                    print(f"     Input size: {model_info['input_size']}")
                
                return True
            else:
                print(f"❌ Model listing failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Model listing error: {e}")
            return False
    
    def test_image_analysis(self, image_path: str, model_name: str = "chexpert"):
        """Test image analysis endpoint"""
        print(f"\n🔬 Testing image analysis with {image_path}...")
        
        try:
            if not Path(image_path).exists():
                print(f"❌ Image file not found: {image_path}")
                return False
            
            # Prepare file for upload
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                data = {'model_name': model_name}
                
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data
                )
                processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Image analysis successful!")
                print(f"   Case ID: {result['case_id']}")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   API call time: {processing_time:.2f}s")
                print(f"   Model: {result['model_info']['name']}")
                print(f"   Predictions ({len(result['predictions'])}):")
                
                for i, pred in enumerate(result['predictions'][:5]):  # Show top 5
                    confidence_pct = pred['confidence'] * 100
                    print(f"     {i+1}. {pred['condition']}: {confidence_pct:.1f}%")
                
                print(f"   Annotated image: {self.base_url}{result['annotated_image_url']}")
                
                return result['case_id']
            else:
                print(f"❌ Image analysis failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Image analysis error: {e}")
            return None
    
    def test_get_case(self, case_id: str):
        """Test retrieving a specific case"""
        print(f"\n📋 Testing case retrieval for {case_id}...")
        
        try:
            response = self.session.get(f"{self.base_url}/cases/{case_id}")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Case retrieval successful!")
                print(f"   Case ID: {data['case_id']}")
                print(f"   Original filename: {data['original_filename']}")
                print(f"   File size: {data['file_size']} bytes")
                print(f"   Image format: {data['image_format']}")
                print(f"   Created at: {data['created_at']}")
                print(f"   Original image: {self.base_url}{data['original_image_url']}")
                print(f"   Annotated image: {self.base_url}{data['annotated_image_url']}")
                
                return True
            else:
                print(f"❌ Case retrieval failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Case retrieval error: {e}")
            return False
    
    def test_list_cases(self):
        """Test listing all cases"""
        print("\n📚 Testing case listing...")
        
        try:
            response = self.session.get(f"{self.base_url}/cases?limit=10")
            
            if response.status_code == 200:
                cases = response.json()
                print(f"✅ Case listing successful! Found {len(cases)} cases")
                
                for case in cases[:3]:  # Show first 3 cases
                    print(f"   Case: {case['case_id'][:8]}...")
                    print(f"     File: {case['original_filename']}")
                    print(f"     Predictions: {len(case['predictions'])}")
                
                return True
            else:
                print(f"❌ Case listing failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Case listing error: {e}")
            return False
    
    def create_sample_image(self, save_path: str = "sample_chest_xray.jpg"):
        """Create a sample chest X-ray image for testing"""
        try:
            import numpy as np
            from PIL import Image
            
            # Create a synthetic chest X-ray-like image
            # This is just for testing - not a real medical image
            width, height = 512, 512
            
            # Create a gradient background (dark to light)
            image_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add chest cavity shape
            center_x, center_y = width // 2, height // 2
            for y in range(height):
                for x in range(width):
                    # Distance from center
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # Create chest-like pattern
                    if dist < 200:
                        intensity = int(50 + (200 - dist) * 0.5)
                        image_array[y, x] = [intensity, intensity, intensity]
                    else:
                        intensity = 30
                        image_array[y, x] = [intensity, intensity, intensity]
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 10, (height, width, 3))
            image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
            
            # Save the image
            img = Image.fromarray(image_array)
            img.save(save_path, 'JPEG', quality=85)
            
            print(f"📸 Created sample image: {save_path}")
            return save_path
            
        except ImportError:
            print("❌ Cannot create sample image: numpy or PIL not available")
            return None
        except Exception as e:
            print(f"❌ Error creating sample image: {e}")
            return None
    
    def run_full_test_suite(self, image_path: str = None):
        """Run the complete test suite"""
        print("🧪 Starting Tele-Radiology API Test Suite")
        print("=" * 50)
        
        # Test 1: Health check
        if not self.test_health_check():
            print("❌ Critical: Health check failed. Is the server running?")
            return False
        
        # Test 2: List models
        if not self.test_list_models():
            print("⚠️  Warning: Model listing failed")
        
        # Test 3: Image analysis
        if image_path is None:
            image_path = self.create_sample_image()
        
        if image_path:
            case_id = self.test_image_analysis(image_path)
            
            if case_id:
                # Test 4: Get specific case
                self.test_get_case(case_id)
        
        # Test 5: List cases
        self.test_list_cases()
        
        print("\n" + "=" * 50)
        print("🎉 Test suite completed!")
        
        return True


def main():
    """Main function to run the tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Tele-Radiology API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", help="Path to test image file")
    parser.add_argument("--model", default="chexpert", help="Model name to use")
    
    args = parser.parse_args()
    
    tester = TeleRadiologyAPITester(args.url)
    
    # Run the full test suite
    tester.run_full_test_suite(args.image)


if __name__ == "__main__":
    main()
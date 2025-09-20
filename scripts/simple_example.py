#!/usr/bin/env python3
"""
Simple example of using the Tele-Radiology API
"""
import requests
import sys
from pathlib import Path


def analyze_image(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Simple function to analyze a medical image
    
    Args:
        image_path: Path to the image file
        api_url: Base URL of the API
    """
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file '{image_path}' not found")
        return None
    
    try:
        # Prepare the file for upload
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            
            # Send request to analyze endpoint
            print(f"🔬 Analyzing image: {image_path}")
            response = requests.post(f"{api_url}/analyze", files=files)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Analysis completed successfully!")
            print(f"📋 Case ID: {result['case_id']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
            print(f"🤖 Model: {result['model_info']['name']}")
            
            # Display predictions
            predictions = result['predictions']
            if predictions:
                print(f"\n🔍 Found {len(predictions)} potential conditions:")
                for i, pred in enumerate(predictions, 1):
                    confidence_pct = pred['confidence'] * 100
                    print(f"   {i}. {pred['condition']}: {confidence_pct:.1f}%")
            else:
                print("\n✅ No significant pathologies detected")
            
            # Show where to find the annotated image
            annotated_url = f"{api_url}{result['annotated_image_url']}"
            print(f"\n🖼️  Annotated image available at: {annotated_url}")
            
            return result
            
        else:
            print(f"❌ Analysis failed with status code: {response.status_code}")
            print(f"Error details: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to API at {api_url}")
        print("Make sure the Tele-Radiology server is running")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python simple_example.py <image_path>")
        print("Example: python simple_example.py chest_xray.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("🏥 Tele-Radiology Simple Example")
    print("=" * 40)
    
    # Analyze the image
    result = analyze_image(image_path)
    
    if result:
        print("\n✨ Analysis complete! Check the annotated image for visual results.")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
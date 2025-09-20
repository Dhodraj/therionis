#!/usr/bin/env python3
"""
Setup script for the Tele-Radiology Backend
"""
import subprocess
import sys
import platform
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Minimum required version: Python 3.8")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install Python dependencies"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    
    if not in_venv:
        print("⚠️  No virtual environment detected.")
        print("Creating virtual environment...")
        
        venv_path = Path(__file__).parent / "venv"
        
        # Create virtual environment
        create_venv_cmd = f"{sys.executable} -m venv {venv_path}"
        if not run_command(create_venv_cmd, "Creating virtual environment"):
            return False
        
        # Get the correct pip path for the virtual environment
        if platform.system() == "Windows":
            pip_path = venv_path / "Scripts" / "pip"
            python_path = venv_path / "Scripts" / "python"
        else:
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"
        
        # Install dependencies in the virtual environment
        command = f"{pip_path} install -r {requirements_file}"
        success = run_command(command, "Installing Python dependencies in virtual environment")
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 Virtual environment created successfully!")
            print("=" * 60)
            print(f"📁 Virtual environment location: {venv_path}")
            print("\nTo activate the virtual environment:")
            if platform.system() == "Windows":
                print(f"   {venv_path}\\Scripts\\activate")
            else:
                print(f"   source {venv_path}/bin/activate")
            print("\nTo run the application:")
            print(f"   {python_path} main.py")
            print("\nOr use the provided scripts:")
            print("   make run    # if you have make installed")
            print("   ./run.sh    # if you prefer shell scripts")
        
        return success
    else:
        # We're already in a virtual environment, install normally
        command = f"{sys.executable} -m pip install -r {requirements_file}"
        return run_command(command, "Installing Python dependencies")


def setup_environment():
    """Setup environment files"""
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    # Create .env.example if it doesn't exist
    if not env_example.exists():
        print("📝 Creating .env.example file...")
        with open(env_example, 'w') as f:
            f.write("""# Tele-Radiology Backend Configuration
DEBUG=False
HOST=127.0.0.1
PORT=8000
DATABASE_URL=sqlite:///radiology.db

# Image processing settings
MAX_IMAGE_SIZE=2048
MAX_FILE_SIZE=52428800
CONFIDENCE_THRESHOLD=0.1
""")
        print("✅ Created .env.example")
    
    # Create .env if it doesn't exist
    if not env_file.exists():
        print("📝 Creating .env file...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from template")
    else:
        print("✅ .env file already exists")
    
    return True


def create_directories():
    """Create necessary directories"""
    project_root = Path(__file__).parent
    directories = [
        project_root / "storage",
        project_root / "storage" / "original",
        project_root / "storage" / "annotated",
        project_root / "logs"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    return True


def test_imports():
    """Test critical imports"""
    print("🧪 Testing critical imports...")
    
    critical_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("torchxrayvision", "TorchXRayVision"),
        ("fastapi", "FastAPI"),
        ("sqlalchemy", "SQLAlchemy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]
    
    failed_imports = []
    
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (import '{module}' failed)")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️  Some imports failed: {', '.join(failed_imports)}")
        print("   You may need to install additional dependencies")
        return False
    
    print("✅ All critical imports successful")
    return True


def verify_setup():
    """Verify that setup was successful"""
    print("\n🔍 Verifying setup...")
    
    project_root = Path(__file__).parent
    
    # Check files exist
    required_files = [
        "main.py",
        "requirements.txt",
        ".env",
        "config/settings.py",
        "src/api/main.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    # Check directories exist
    required_dirs = [
        "storage/original",
        "storage/annotated",
        "src/api",
        "src/models",
        "src/preprocessing",
        "src/visualization"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("✅ Setup verification successful")
    return True


def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start the server:")
    print("   python main.py")
    print()
    print("2. Test the API:")
    print("   python scripts/test_api.py")
    print()
    print("3. Try the simple example:")
    print("   python scripts/simple_example.py <image_path>")
    print()
    print("4. Access the API documentation:")
    print("   http://localhost:8000/docs")
    print()
    print("📚 For more information, check the README.md file")
    print()


def main():
    """Main setup function"""
    print("🏥 Tele-Radiology Backend Setup")
    print("=" * 40)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Step 3: Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment")
        sys.exit(1)
    
    # Step 4: Install dependencies
    print("\n📦 Installing dependencies...")
    print("This may take several minutes as it downloads PyTorch and other large packages...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        print("You may need to install them manually with:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 5: Test imports
    if not test_imports():
        print("⚠️  Some imports failed, but setup will continue")
    
    # Step 6: Verify setup
    if not verify_setup():
        print("❌ Setup verification failed")
        sys.exit(1)
    
    # Step 7: Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
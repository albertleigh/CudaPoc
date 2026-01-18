import os
import subprocess
import sys
import argparse
import platform
import json


# Configuration Constants
# Visual Studio paths
VS_INSTALLER_REL_PATH = ('Microsoft Visual Studio', 'Installer', 'vswhere.exe')
VS_VCVARSALL_REL_PATH = ('VC', 'Auxiliary', 'Build', 'vcvarsall.bat')
VS_REQUIRED_COMPONENT = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
VS_ARCHITECTURE = 'x64'

# vcpkg paths
VCPKG_TOOLCHAIN_REL_PATH = ('scripts', 'buildsystems', 'vcpkg.cmake')
VCPKG_COMMON_LOCATIONS = [
    'C:/opt/vcpkg',
    'C:/vcpkg',
]

# Environment variables
ENV_PROGRAM_FILES_X86 = 'ProgramFiles(x86)'
ENV_VCPKG_ROOT = 'VCPKG_ROOT'
ENV_TEMP_DIR = 'TEMP'
ENV_USER_PROFILE = 'USERPROFILE'

# Default values
DEFAULT_PROGRAM_FILES_X86 = 'C:\\Program Files (x86)'
DEFAULT_TEMP_DIR = '.'

# Temporary files
TEMP_ENV_SCRIPT_NAME = 'dump_env.bat'

# Build tools
NINJA_COMMAND = 'ninja'
CMAKE_COMMAND = 'cmake'
CMD_SHELL = 'cmd'


def setup_windows_environment():
    """Setup Visual Studio environment variables on Windows."""
    # Try to find vswhere to locate Visual Studio
    program_files_x86 = os.environ.get(ENV_PROGRAM_FILES_X86, DEFAULT_PROGRAM_FILES_X86)
    vswhere_path = os.path.join(program_files_x86, *VS_INSTALLER_REL_PATH)
    
    if not os.path.exists(vswhere_path):
        print('Warning: vswhere.exe not found, using current environment')
        return os.environ.copy()
    
    # Find latest Visual Studio installation
    try:
        result = subprocess.run(
            [vswhere_path, '-latest', '-products', '*', '-requires', 
             VS_REQUIRED_COMPONENT, '-property', 'installationPath'],
            capture_output=True, text=True, check=True
        )
        vs_path = result.stdout.strip()
        
        if not vs_path:
            print('Warning: Visual Studio not found, using current environment')
            return os.environ.copy()
        
        print(f'Found Visual Studio at: {vs_path}')
        
        # Find vcvarsall.bat
        vcvarsall = os.path.join(vs_path, *VS_VCVARSALL_REL_PATH)
        if not os.path.exists(vcvarsall):
            print(f'Warning: vcvarsall.bat not found at {vcvarsall}')
            return os.environ.copy()
        
        # Run vcvarsall.bat and capture the environment
        # Use x64 architecture for 64-bit builds
        print('Setting up Visual Studio environment...')
        
        # Create a batch script to dump environment variables
        temp_dir = os.environ.get(ENV_TEMP_DIR, DEFAULT_TEMP_DIR)
        temp_script = os.path.join(temp_dir, TEMP_ENV_SCRIPT_NAME)
        with open(temp_script, 'w') as f:
            f.write(f'@echo off\n')
            f.write(f'call "{vcvarsall}" {VS_ARCHITECTURE} >nul\n')
            f.write('if errorlevel 1 exit /b 1\n')
            f.write('set\n')
        
        try:
            result = subprocess.run(
                [CMD_SHELL, '/c', temp_script],
                capture_output=True, text=True, check=True
            )
            
            # Parse environment variables from output
            env = os.environ.copy()
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, _, value = line.partition('=')
                    env[key] = value
            
            print('Visual Studio environment configured successfully')
            return env
        finally:
            # Clean up temp script
            if os.path.exists(temp_script):
                os.remove(temp_script)
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'Warning: Failed to setup VS environment: {e}')
        print('Using current environment')
        return os.environ.copy()


def setup_linux_environment():
    """Setup build environment for Linux."""
    # Placeholder for Linux-specific environment setup
    # Add compiler detection, library paths, etc. as needed
    return os.environ.copy()


def setup_macos_environment():
    """Setup build environment for macOS."""
    # Placeholder for macOS-specific environment setup
    # Add Xcode tools detection, SDK paths, etc. as needed
    return os.environ.copy()


def setup_build_environment():
    """Setup platform-specific build environment."""
    system = platform.system()
    
    if system == 'Windows':
        return setup_windows_environment()
    elif system == 'Linux':
        return setup_linux_environment()
    elif system == 'Darwin':  # macOS
        return setup_macos_environment()
    else:
        print(f'Warning: Unsupported platform {system}, using current environment')
        return os.environ.copy()


def main():
    parser = argparse.ArgumentParser(description='Build the CudaPoc project')
    parser.add_argument('--release', action='store_true',
                        help='Build in Release mode (default: Debug)')
    parser.add_argument('--clean', action='store_true',
                        help='Clean build directory before building')
    parser.add_argument('--jobs', '-j', type=int, default=None,
                        help='Number of parallel build jobs (default: auto)')
    
    args = parser.parse_args()
    
    # Determine build type
    build_type = 'Release' if args.release else 'Debug'
    
    # Assume script is in <root>/scripts/
    script_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_path)
    
    # Build directory naming
    build_dir_name = f'cmake-build-{build_type.lower()}'
    build_dir = os.path.join(root_dir, build_dir_name)
    
    print(f'Project root: {root_dir}')
    print(f'Build type: {build_type}')
    print(f'Build directory: {build_dir}')
    
    # Clean if requested
    if args.clean and os.path.exists(build_dir):
        print(f'\nCleaning build directory: {build_dir}')
        import shutil
        shutil.rmtree(build_dir)
    
    # Create build directory if it doesn't exist
    if not os.path.exists(build_dir):
        print(f'\nCreating build directory: {build_dir}')
        os.makedirs(build_dir)
    
    # Setup platform-specific build environment
    env = setup_build_environment()
    
    # Configure CMake
    print('\n' + '='*60)
    print('Configuring CMake...')
    print('='*60)
    
    cmake_args = [
        CMAKE_COMMAND,
        '-S', root_dir,
        '-B', build_dir,
        f'-DCMAKE_BUILD_TYPE={build_type}',
    ]
    
    # Add vcpkg toolchain file
    # First check environment variable
    vcpkg_root = os.environ.get(ENV_VCPKG_ROOT)
    if vcpkg_root:
        vcpkg_toolchain = os.path.join(vcpkg_root, *VCPKG_TOOLCHAIN_REL_PATH)
        if os.path.exists(vcpkg_toolchain):
            cmake_args.append(f'-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain}')
            print(f'Using vcpkg toolchain from: {vcpkg_toolchain}')
    else:
        # Try common locations
        common_locations = [
            os.path.join(loc, *VCPKG_TOOLCHAIN_REL_PATH) 
            for loc in VCPKG_COMMON_LOCATIONS
        ]
        # Also try user profile directory
        user_profile = os.environ.get(ENV_USER_PROFILE, '')
        if user_profile:
            common_locations.append(
                os.path.join(user_profile, 'vcpkg', *VCPKG_TOOLCHAIN_REL_PATH)
            )
        
        for location in common_locations:
            if os.path.exists(location):
                cmake_args.append(f'-DCMAKE_TOOLCHAIN_FILE={location}')
                print(f'Using vcpkg toolchain from: {location}')
                break
    
    # Add generator if on Windows (use Ninja if available, otherwise Visual Studio)
    if platform.system() == 'Windows':
        # Try to use Ninja if available
        try:
            subprocess.run([NINJA_COMMAND, '--version'], capture_output=True, check=True)
            cmake_args.extend(['-G', 'Ninja'])
            print('Using Ninja generator')
        except (subprocess.CalledProcessError, FileNotFoundError):
            print('Using default generator (Visual Studio)')
    
    try:
        result = subprocess.run(cmake_args, cwd=root_dir, check=True, 
                              stdout=sys.stdout, stderr=sys.stderr, env=env)
    except subprocess.CalledProcessError as e:
        print(f'\nCMake configuration failed with exit code {e.returncode}')
        sys.exit(1)
    
    # Build
    print('\n' + '='*60)
    print('Building project...')
    print('='*60)
    
    build_args = [
        CMAKE_COMMAND,
        '--build', build_dir,
        '--config', build_type,
    ]
    
    if args.jobs:
        build_args.extend(['--parallel', str(args.jobs)])
    
    try:
        result = subprocess.run(build_args, cwd=root_dir, check=True,
                              stdout=sys.stdout, stderr=sys.stderr, env=env)
    except subprocess.CalledProcessError as e:
        print(f'\nBuild failed with exit code {e.returncode}')
        sys.exit(1)
    
    print('\n' + '='*60)
    print(f'Build completed successfully!')
    print(f'Build type: {build_type}')
    print(f'Build directory: {build_dir}')
    print('='*60)


if __name__ == '__main__':
    main()

import os
import subprocess
import sys
import platform

def run_gtests():
    # Assume script is in <root>/scripts/
    script_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_path)
    
    # Detect build directory
    build_dirs = ['cmake-build-debug', 'build']
    build_dir = None
    for d in build_dirs:
        path = os.path.join(root_dir, d)
        if os.path.isdir(path):
            build_dir = path
            break
    
    if not build_dir:
        print('Error: Could not find build directory (cmake-build-debug or build).')
        sys.exit(1)

    print(f'Searching for tests in: {build_dir}')
    
    test_executables = []
    is_windows = platform.system() == 'Windows'
    extension = '.exe' if is_windows else ''
    
    # Walk through the build directory to find test executables
    for dirpath, dirnames, filenames in os.walk(build_dir):
        for filename in filenames:
            if filename.startswith('test_') and filename.endswith(extension):
                # Skip compiler check files if any
                if 'CompilerId' in dirpath or 'CMakeFiles' in dirpath:
                    continue
                    
                full_path = os.path.join(dirpath, filename)
                test_executables.append(full_path)

    if not test_executables:
        print('No test executables found.')
        return

    print(f'Found {len(test_executables)} test executables: {[os.path.basename(t) for t in test_executables]}')
    
    failed_tests = []
    for exe in test_executables:
        print(f'\n{'='*60}')
        print(f'Running: {os.path.basename(exe)}')
        print(f'{'='*60}')
        
        try:
            # Run the test executable
            result = subprocess.run([exe], check=False)
            if result.returncode != 0:
                failed_tests.append(exe)
        except Exception as e:
            print(f'Failed to run {exe}: {e}')
            failed_tests.append(exe)

    print('\n' + '='*60)
    if failed_tests:
        print(f'Summary: {len(failed_tests)} out of {len(test_executables)} test suites FAILED.')
        for f in failed_tests:
            print(f'[FAILED] {os.path.basename(f)}')
        sys.exit(1)
    else:
        print(f'Summary: All {len(test_executables)} test suites PASSED.')
        sys.exit(0)

if __name__ == '__main__':
    run_gtests()


#!/usr/bin/env python3
"""
Format all C++ and CUDA source files using clang-format.
"""

import subprocess
import sys
from pathlib import Path
from typing import List


def find_source_files(root_dir: Path) -> List[Path]:
    """Find all C++ and CUDA source files in the project."""
    extensions = {'.cpp', '.cc', '.cu', '.h', '.hpp', '.cuh'}
    
    # Directories to exclude
    exclude_dirs = {
        'build', 'cmake-build-debug', 'cmake-build-release',
        '.git', 'vcpkg_installed', '.venv', '__pycache__',
        'third_party', 'external'
    }
    
    files = []
    for ext in extensions:
        for file_path in root_dir.rglob(f'*{ext}'):
            # Skip if file is in excluded directory
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            files.append(file_path)
    
    return sorted(files)


def format_files(files: List[Path], dry_run: bool = False) -> None:
    """Format files using clang-format."""
    if not files:
        print("No files to format.")
        return
    
    print(f"Found {len(files)} files to format.")
    
    if dry_run:
        print("\n=== DRY RUN MODE - No files will be modified ===\n")
    
    success_count = 0
    error_count = 0
    
    for file_path in files:
        try:
            if dry_run:
                # Check if formatting would change the file
                result = subprocess.run(
                    ['clang-format', '--dry-run', '-Werror', str(file_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✓ {file_path.relative_to(Path.cwd())}")
                    success_count += 1
                else:
                    print(f"⚠ {file_path.relative_to(Path.cwd())} - would be reformatted")
                    success_count += 1
            else:
                # Format in-place
                result = subprocess.run(
                    ['clang-format', '-i', str(file_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✓ {file_path.relative_to(Path.cwd())}")
                    success_count += 1
                else:
                    print(f"✗ {file_path.relative_to(Path.cwd())} - error")
                    if result.stderr:
                        print(f"  Error: {result.stderr}")
                    error_count += 1
        except FileNotFoundError:
            print("\nError: clang-format not found. Please install it first.")
            print("\nInstallation instructions:")
            print("  Ubuntu/Debian: sudo apt-get install clang-format")
            print("  Fedora/RHEL:   sudo dnf install clang-tools-extra")
            print("  Arch Linux:    sudo pacman -S clang")
            print("  macOS:         brew install clang-format")
            print("\nOr install a specific version:")
            print("  sudo apt-get install clang-format-18")
            sys.exit(1)
        except Exception as e:
            print(f"✗ {file_path.relative_to(Path.cwd())} - {e}")
            error_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {success_count} files {'checked' if dry_run else 'formatted'}, {error_count} errors")
    if dry_run and success_count > 0:
        print("\nTo apply changes, run without --dry-run flag")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Format all C++ and CUDA files using clang-format'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check formatting without modifying files'
    )
    parser.add_argument(
        '--dir',
        type=Path,
        default=Path.cwd(),
        help='Root directory to search (default: current directory)'
    )
    
    args = parser.parse_args()
    
    if not args.dir.exists():
        print(f"Error: Directory {args.dir} does not exist")
        sys.exit(1)
    
    print(f"Searching for source files in: {args.dir}")
    files = find_source_files(args.dir)
    format_files(files, dry_run=args.dry_run)


if __name__ == '__main__':
    main()

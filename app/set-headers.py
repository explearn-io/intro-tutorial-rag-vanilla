#!/usr/bin/env python3
"""
File: add_headers.py
Description: Script to recursively add standard headers to all Python files in a directory
Author: Arturo Gomez-Chavez
Creation Date: 30.06.2025
Institution/Organization: Constructor University GmbH
Contributors/Editors:
License: MIT License - See LICENSE.MD file for details
Contact & Support:
- Email: [support@example.com]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

def get_header_template(filename: str, description: str = "[Brief description of the file's purpose]") -> str:
    """
    Generate the standard header template for a given filename
    
    Args:
        filename: Name of the file
        description: Description of the file's purpose
        
    Returns:
        Formatted header string
    """
    header = f'''"""
File: {filename}
Description: {description}
Author: Arturo Gomez-Chavez
Creation Date: 07.07.2025
Institution/Organization: NA
Contributors/Editors:
License: MIT License - See LICENSE.MD file for details
Contact & Support:
- Email: [support@example.com]
"""

'''
    return header

def has_existing_header(content: str) -> bool:
    """
    Check if file already has a header with author information
    
    Args:
        content: File content as string
        
    Returns:
        True if header exists, False otherwise
    """
    lines = content.split('\n')
    
    # Check first 20 lines for existing header patterns
    header_section = '\n'.join(lines[:20])
    
    # Look for common header indicators
    header_indicators = [
        'Author:',
        'Creation Date:',
        'Institution/Organization:',
        'File:',
        'Description:'
    ]
    
    return any(indicator in header_section for indicator in header_indicators)

def get_file_description(filename: str) -> str:
    """
    Generate a basic description based on filename
    
    Args:
        filename: Name of the file
        
    Returns:
        Generated description string
    """
    descriptions = {
        'main.py': 'FastAPI application entry point and service configuration',
        'services.py': 'FastAPI service endpoints for image registration and batch processing',
        'schemas.py': 'Pydantic models and data schemas for API requests and responses',
        'executor.py': 'Command execution service for running registration tools',
        'batch_processor.py': 'Batch processing service for sequential fourier-soft2d operations',
        'batch_stitching_processor.py': 'Batch image stitching processor for progressive image combination',
        'utils.py': 'Utility functions for sequential batch processing operations',
        '__init__.py': 'Package initialization file',
        'requirements.txt': 'Python package dependencies',
        'dockerfile': 'Docker container configuration',
        'docker-compose.yml': 'Docker Compose service configuration',
        'readme.md': 'Project documentation and setup instructions'
    }
    
    # Check exact matches first
    lower_filename = filename.lower()
    if lower_filename in descriptions:
        return descriptions[lower_filename]
    
    # Generate description based on file patterns
    if filename.endswith('_test.py') or filename.endswith('test_.py'):
        return f"Unit tests for {filename.replace('_test.py', '').replace('test_', '')}"
    elif filename.endswith('_config.py'):
        return "Configuration settings and parameters"
    elif filename.endswith('_models.py'):
        return "Data models and database schemas"
    elif filename.endswith('_views.py'):
        return "View controllers and request handlers"
    elif filename.endswith('_utils.py'):
        return "Utility functions and helper methods"
    elif 'api' in lower_filename:
        return "API endpoint definitions and handlers"
    elif 'client' in lower_filename:
        return "Client interface and communication handlers"
    elif 'server' in lower_filename:
        return "Server implementation and request processing"
    elif 'database' in lower_filename or 'db' in lower_filename:
        return "Database operations and data access layer"
    else:
        return "[Brief description of the file's purpose]"

def add_header_to_file(file_path: Path, dry_run: bool = False) -> Dict[str, any]:
    """
    Add header to a single file
    
    Args:
        file_path: Path to the file
        dry_run: If True, only simulate the operation
        
    Returns:
        Dictionary with operation results
    """
    result = {
        'file': str(file_path),
        'success': False,
        'action': 'skipped',
        'message': ''
    }
    
    try:
        # Read existing content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if header already exists
        if has_existing_header(content):
            result['action'] = 'skipped'
            result['message'] = 'Header already exists'
            result['success'] = True
            return result
        
        # Generate header
        filename = file_path.name
        description = get_file_description(filename)
        header = get_header_template(filename, description)
        
        # Handle shebang line
        new_content = content
        if content.startswith('#!'):
            lines = content.split('\n')
            shebang = lines[0]
            rest_content = '\n'.join(lines[1:])
            new_content = f"{shebang}\n{header}{rest_content}"
        else:
            new_content = f"{header}{content}"
        
        if dry_run:
            result['action'] = 'would_add'
            result['message'] = f'Would add header with description: {description}'
            result['success'] = True
        else:
            # Write the file with header
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            result['action'] = 'added'
            result['message'] = f'Header added with description: {description}'
            result['success'] = True
    
    except Exception as e:
        result['action'] = 'error'
        result['message'] = f'Error: {str(e)}'
        result['success'] = False
    
    return result

def find_python_files(directory: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """
    Find all Python files in directory recursively
    
    Args:
        directory: Root directory to search
        exclude_patterns: Patterns to exclude (e.g., ['__pycache__', '.git'])
        
    Returns:
        List of Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.git', '.venv', 'venv', 'node_modules', '.pytest_cache']
    
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Remove excluded directories from search
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                python_files.append(file_path)
    
    return sorted(python_files)

def main():
    parser = argparse.ArgumentParser(
        description="Add standard headers to Python files recursively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_headers.py /path/to/project
  python add_headers.py . --dry-run
  python add_headers.py /project --exclude __pycache__ .git
  python add_headers.py /project --include-only services.py utils.py
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory to process recursively'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=['__pycache__', '.git', '.venv', 'venv', 'node_modules', '.pytest_cache'],
        help='Patterns to exclude from processing'
    )
    
    parser.add_argument(
        '--include-only',
        nargs='*',
        help='Process only these specific filenames'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output for each file'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)
    
    # Find Python files
    print(f"Scanning directory: {directory}")
    python_files = find_python_files(directory, args.exclude)
    
    # Filter by include-only if specified
    if args.include_only:
        python_files = [f for f in python_files if f.name in args.include_only]
    
    if not python_files:
        print("No Python files found to process")
        sys.exit(0)
    
    print(f"Found {len(python_files)} Python files to process")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE - No files will be modified ---")
    
    # Process files
    results = {
        'added': [],
        'skipped': [],
        'would_add': [],
        'errors': []
    }
    
    for file_path in python_files:
        result = add_header_to_file(file_path, args.dry_run)
        results[result['action']].append(result)
        
        if args.verbose or result['action'] == 'error':
            status_symbol = {
                'added': '✓',
                'skipped': '○',
                'would_add': '→',
                'error': '✗'
            }.get(result['action'], '?')
            
            print(f"{status_symbol} {result['file']}: {result['message']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Files processed: {len(python_files)}")
    
    if args.dry_run:
        print(f"  Would add headers: {len(results['would_add'])}")
    else:
        print(f"  Headers added: {len(results['added'])}")
    
    print(f"  Skipped (existing): {len(results['skipped'])}")
    print(f"  Errors: {len(results['errors'])}")
    
    # Show errors if any
    if results['errors']:
        print("\nERRORS:")
        for error in results['errors']:
            print(f"  ✗ {error['file']}: {error['message']}")
    
    # Show files that would be modified in dry-run
    if args.dry_run and results['would_add']:
        print(f"\nFiles that would be modified:")
        for item in results['would_add']:
            print(f"  → {item['file']}")
    
    print("="*60)

if __name__ == "__main__":
    main()

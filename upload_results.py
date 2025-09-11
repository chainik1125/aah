#!/usr/bin/env python3
"""
Script to upload result files from RunPod to various services.
Run this from within the RunPod instance.
"""

import os
import subprocess
import sys
import glob
from datetime import datetime

def upload_to_transfer_sh(file_path):
    """
    Upload file to transfer.sh (anonymous, expires in 14 days)
    No account needed!
    """
    print(f"Uploading {file_path} to transfer.sh...")
    try:
        result = subprocess.run(
            f'curl --upload-file "{file_path}" https://transfer.sh/{os.path.basename(file_path)}',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            print(f"✓ Success! Download URL: {url}")
            return url
        else:
            print(f"✗ Failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def upload_to_fileio(file_path, expires='1d'):
    """
    Upload file to file.io (anonymous, customizable expiry)
    Expires: 1d, 1w, 2w, 1m, etc.
    """
    print(f"Uploading {file_path} to file.io (expires: {expires})...")
    try:
        result = subprocess.run(
            f'curl -F "file=@{file_path}" "https://file.io/?expires={expires}"',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            import json
            response = json.loads(result.stdout)
            if response.get('success'):
                url = response.get('link')
                print(f"✓ Success! Download URL: {url}")
                print(f"  (Note: file.io links expire after first download or {expires})")
                return url
            else:
                print(f"✗ Failed: {response.get('message')}")
        else:
            print(f"✗ Failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def upload_to_tmpfiles(file_path):
    """
    Upload to tmpfiles.org (anonymous, expires in 1 hour)
    Good for quick transfers
    """
    print(f"Uploading {file_path} to tmpfiles.org...")
    try:
        result = subprocess.run(
            f'curl -F "file=@{file_path}" https://tmpfiles.org/api/v1/upload',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            import json
            response = json.loads(result.stdout)
            if response.get('status') == 'success':
                # Convert URL to direct download link
                url = response['data']['url'].replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                print(f"✓ Success! Download URL: {url}")
                print(f"  (Expires in 1 hour)")
                return url
            else:
                print(f"✗ Failed: {response}")
        else:
            print(f"✗ Failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def create_download_links_file(urls, output_file='download_links.txt'):
    """Save all download links to a file"""
    with open(output_file, 'w') as f:
        f.write(f"Download Links - Generated {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        for filename, url in urls.items():
            f.write(f"{filename}:\n{url}\n\n")
    print(f"\nAll links saved to: {output_file}")

def main():
    print("=" * 60)
    print("RUNPOD FILE UPLOADER")
    print("=" * 60)
    
    # Find HTML files
    html_files = glob.glob('large_files/plots/*.html')
    
    if not html_files:
        print("No HTML files found in large_files/plots/")
        return
    
    print(f"\nFound {len(html_files)} HTML file(s):")
    for f in html_files:
        size = os.path.getsize(f) / (1024 * 1024)  # Size in MB
        print(f"  - {os.path.basename(f)} ({size:.2f} MB)")
    
    print("\nChoose upload service:")
    print("1. transfer.sh (recommended - expires in 14 days)")
    print("2. file.io (expires after 1 download or custom time)")
    print("3. tmpfiles.org (quick - expires in 1 hour)")
    print("4. Upload all to transfer.sh")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    urls = {}
    
    if choice == '4':
        # Upload all files
        for file_path in html_files:
            print(f"\n[{html_files.index(file_path)+1}/{len(html_files)}]")
            url = upload_to_transfer_sh(file_path)
            if url:
                urls[os.path.basename(file_path)] = url
    else:
        # Select specific file
        if len(html_files) > 1:
            print("\nSelect file to upload:")
            for i, f in enumerate(html_files):
                print(f"{i+1}. {os.path.basename(f)}")
            file_idx = int(input("Enter number: ")) - 1
            file_path = html_files[file_idx]
        else:
            file_path = html_files[0]
        
        # Upload based on choice
        if choice == '1':
            url = upload_to_transfer_sh(file_path)
        elif choice == '2':
            expires = input("Expiry time (1d/1w/2w/1m) [default: 1d]: ").strip() or '1d'
            url = upload_to_fileio(file_path, expires)
        elif choice == '3':
            url = upload_to_tmpfiles(file_path)
        else:
            print("Invalid choice")
            return
        
        if url:
            urls[os.path.basename(file_path)] = url
    
    # Save links to file
    if urls:
        create_download_links_file(urls)
        print("\n" + "=" * 60)
        print("✓ Upload complete! Copy the URL(s) above to download on your local machine.")
        print("=" * 60)

if __name__ == "__main__":
    main()
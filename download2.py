import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
pd.set_option("display.max_columns", None)
import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
import hashlib
import urllib3
import random
import sys
import codecs

# Set console to UTF-8 mode
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


# Disable SSL warnings if needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Base API URL
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

def create_session_with_retries():
    session = requests.Session()
    
    retry_strategy = Retry(
        total=5,  # Increased retries
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,  # Exponential backoff
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Updated parameter name
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set session-wide timeout
    session.timeout = 30
    
    return session

def get_with_retry(session, url, headers=None, params=None, max_retries=3):
    """Helper function to handle SSL errors with retry logic"""
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                wait_time = (2 ** attempt) * 10  # Exponential backoff, longer for 403
                print(f"403 Forbidden (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            raise
        except (requests.exceptions.SSLError, 
                requests.exceptions.ConnectionError) as e:
            print(f"SSL/Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                raise
        except Exception as e:
            print(f"Other error: {e}")
            raise

# Get all object IDs
def get_all_objects():
    session = create_session_with_retries()
    headers = {
        'User-Agent': 'MetMuseum-ML-Dataset/1.0',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }
    
    time.sleep(0.1)  # Increased delay
    
    try:
        response = get_with_retry(session, f"{BASE_URL}/objects", headers=headers)
        data = response.json()
        return data.get('objectIDs', [])
    except Exception as e:
        print(f"Error getting all objects: {e}")
        return []

# Get object details with image URLs
def get_object_details(object_id):
    session = create_session_with_retries()
    headers = {
        'User-Agent': 'MetMuseum-ML-Dataset/1.0',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }

    time.sleep(0.1)  # Increased delay
    
    try:
        response = get_with_retry(session, f"{BASE_URL}/objects/{object_id}", headers=headers)
        return response.json()
    except Exception as e:
        print(f"Error getting object details for {object_id}: {e}")
        return {}

# Filter for objects with images and specific criteria
def filter_objects_for_ml(sample_size=500, only_open_access=True, existing_ids =None):
    search_params = {
        'hasImages': 'true',
        'q': 'Paintings',  # Search for paintings specifically
    }
    
    session = create_session_with_retries()
    headers = {
        'User-Agent': 'MetMuseum-ML-Dataset/1.0',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }
    
    time.sleep(0.1)
    
    try:
        response = get_with_retry(session, f"{BASE_URL}/search", headers=headers, params=search_params)
        data = response.json()
        object_ids = data.get('objectIDs', [])

        print(f"Found {len(object_ids)} total objects with images")
        
        # Filter out existing IDs before sampling
        if existing_ids:
            object_ids = [obj_id for obj_id in object_ids if obj_id not in existing_ids]
            print(f"Filtered to {len(object_ids)} new objects")
        
        # Sample if needed
        if len(object_ids) > sample_size:
            object_ids = np.random.choice(object_ids, sample_size, replace=False)
            print(f"Sampled {len(object_ids)} objects for processing")
        
        return object_ids
    except Exception as e:
        print(f"Error filtering objects: {e}")
        return []

def get_existing_downloads():
    """Get set of object IDs that have already been downloaded"""
    existing = set()
    
    # Check processed images directory
    processed_dir = "Data/Images/Processed"
    if os.path.exists(processed_dir):
        for filename in os.listdir(processed_dir):
            if filename.endswith('.jpg'):
                object_id = filename.replace('.jpg', '')
                existing.add(int(object_id))
    
    return existing

def download_and_process_image(object_data, target_size=(224, 224)):
    """Download and preprocess image for ML"""
    USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    # Add more if needed
    ]
    
    # Handle both dictionary and DataFrame row inputs
    if hasattr(object_data, 'to_dict'):
        object_data = object_data.to_dict()
    
    object_id = object_data.get('object_id', object_data.get('objectID'))
    image_url = object_data.get('image_url', object_data.get('primaryImage'))
    
    if not image_url:
        print(f"No image URL for object {object_id}")
        return {'object_id': object_id, 'success': False, 'error': 'No image URL'}
    
    # Create new session for image download
    session = create_session_with_retries()
    
    # Enhanced headers to mimic a real browser and avoid 403 errors
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.metmuseum.org/',
        'Sec-Fetch-Dest': 'image',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'cross-site',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    try:
        # Create directories if they don't exist
        os.makedirs("Data/Images/Raw", exist_ok=True)
        os.makedirs("Data/Images/Processed", exist_ok=True)
        
        print(f"Downloading image for object {object_id}...")
        print(f"Image URL: {image_url}")
        
        # Add delay before download to be respectful
        time.sleep(0.1)
        
        # Download image with retry logic
        response = get_with_retry(session, image_url, headers=headers)
        
        # Save raw image
        raw_path = f"Data/Images/Raw/{object_id}.jpg"
        with open(raw_path, 'wb') as f:
            f.write(response.content)
        
        # Process for ML
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to decode image for object {object_id}")
            return {'object_id': object_id, 'success': False, 'error': 'Image decode failed'}
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for consistency
        processed_image = cv2.resize(image_rgb, target_size)
        
        # Save processed image
        processed_path = f"Data/Images/Processed/{object_id}.jpg"
        cv2.imwrite(processed_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        
        print(f"Successfully processed image for object {object_id}")
        
        return {
            'object_id': object_id,
            'raw_path': raw_path,
            'processed_path': processed_path,
            'success': True
        }
    
    except Exception as e:
        print(f"Error processing {object_id}: {e}")
        return {'object_id': object_id, 'success': False, 'error': str(e)}

def update_metadata(new_data, metadata_file='data/metadata.csv'):
    """Update metadata file with new entries"""
    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file)
        # Append new data and remove duplicates based on object_id
        updated_df = pd.concat([existing_df, pd.DataFrame([new_data])], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset='object_id', keep='last')
    else:
        updated_df = pd.DataFrame([new_data])
    
    updated_df.to_csv(metadata_file, index=False)
    return updated_df

# Parallel downloading with reduced workers to avoid overwhelming the server
def download_images_parallel(object_data_list, max_workers=3):
    """Download images in parallel with conservative threading"""
    print(f"Starting parallel download of {len(object_data_list)} images with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_and_process_image, object_data_list))
    
    successful = [r for r in results if r and r['success']]
    failed = [r for r in results if r and not r['success']]
    
    print(f"Download complete: {len(successful)} successful, {len(failed)} failed")
    return successful

def download_images_sequential(dataset_df):
    """Download images one by one for more stable downloads"""
    print(f"Starting sequential download of {len(dataset_df)} images...")
    
    # Get existing downloads
    existing_downloads = get_existing_downloads()
    print(f"Found {len(existing_downloads)} existing downloads")
    
    # Track counts instead of keeping lists in memory
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Create failed downloads file if needed
    failed_log_path = 'data/failed_downloads.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    
    for count, (idx, row) in enumerate(dataset_df.iterrows(), 1):
        object_id = row['object_id']
        
        # Skip if already downloaded
        if object_id in existing_downloads:
            print(f"Skipping image {idx + 1}/{len(dataset_df)} (already downloaded)")
            skipped_count += 1
            continue

        print(f"Processing image {count}/{len(dataset_df)}")
        result = download_and_process_image(row)
        
        if result['success']:
            successful_count += 1
            # Update metadata immediately after successful download
            metadata_entry = row.to_dict()
            metadata_entry.update({
                'processed_path': result['processed_path'],
                'raw_path': result['raw_path'],
                'download_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            update_metadata(metadata_entry)
            print(f"Updated metadata for object {object_id}")
        else:
            failed_count += 1
            # Log failed download immediately
            failed_df = pd.DataFrame([result])
            failed_df.to_csv(failed_log_path, mode='a', header=not os.path.exists(failed_log_path), index=False)
        
        # Rate limiting between downloads
        time.sleep(1.0)
    
    print(f"\nDownload complete:")
    print(f"- {successful_count} successful")
    print(f"- {failed_count} failed")
    print(f"- {skipped_count} skipped (already downloaded)")
    
    if failed_count > 0:
        print("\nFailed downloads have been logged to 'data/failed_downloads.csv'")
    
    return successful_count

def create_ml_dataset(filtered_object_ids, target_classification="Paintings", only_public_domain=False):
    # Create data directory
    os.makedirs('data', exist_ok=True)
        # Load existing metadata if available
    metadata_file = 'data/metadata.csv'
    existing_downloads = get_existing_downloads()

    if os.path.exists(metadata_file):
        existing_df = pd.read_csv(metadata_file)
        print(f"Loaded {len(existing_df)} existing metadata entries")
        # Convert existing object IDs to set for faster lookup
        existing_ids = set(existing_df['object_id'])
    else:
        existing_df = pd.DataFrame()
        existing_ids = set()

    # Filter out already downloaded objects upfront
    filtered_object_ids = [obj_id for obj_id in filtered_object_ids if obj_id not in existing_ids and obj_id not in existing_downloads]
    print(f"Filtered down to {len(filtered_object_ids)} new objects to process...")
    if len(filtered_object_ids) == 0:
        print("No new objects to process!")
        return existing_df
        # Collect new metadata
    new_metadata = []
    
    print(f"Processing {len(filtered_object_ids)} objects, filtering for classification: {target_classification}...")
    
    if only_public_domain:
        print("Only including public domain (Open Access) artworks...")
    
    for i, object_id in enumerate(filtered_object_ids):
        # Skip if we already have metadata for this object
        if object_id in existing_ids:
            print(f"Skipping metadata for object {object_id} (already exists)")
            continue
        print(f"Processing object {i+1}/{len(filtered_object_ids)}: {object_id}")
        
        obj_data = get_object_details(object_id)
        
        # Check if object has image AND matches classification
        if obj_data.get('primaryImage'):  # Has image
            classification = obj_data.get('classification', 'Unknown')
            is_public_domain = obj_data.get('isPublicDomain', False)
            
            # Filter by classification (case-insensitive)
            if target_classification.lower() in classification.lower():
                
                # Filter by public domain status if requested
                if only_public_domain and not is_public_domain:
                    print(f"Skipped (not public domain): {obj_data.get('title', 'Untitled')}")
                    continue
                
                new_metadata.append({
                    'object_id': obj_data['objectID'],
                    'title': obj_data.get('title', 'Untitled'),
                    'department': obj_data.get('department', 'Unknown'),
                    'object_name': obj_data.get('objectName', 'Unknown'),
                    'culture': obj_data.get('culture', 'Unknown'),
                    'period': obj_data.get('period', 'Unknown'),
                    'medium': obj_data.get('medium', 'Unknown'),
                    'classification': classification,
                    'is_public_domain': is_public_domain,
                    'image_url': obj_data['primaryImage'],
                    'object_date': obj_data.get('objectDate', 'Unknown'),
                    'artist_display_name': obj_data.get('artistDisplayName', 'Unknown')
                })
                domain_status = "Public Domain" if is_public_domain else "Rights Reserved"
                title = obj_data.get('title', 'Untitled').encode('ascii', 'replace').decode('ascii')
                print(f"Added painting: {title} ({domain_status})")
                #print(f"Added painting: {obj_data.get('title', 'Untitled')} ({domain_status})")
            else:
                print(f"Skipped (classification: {classification})")
        
        # More conservative rate limiting
        time.sleep(0.1)
    
    if new_metadata:
        new_df = pd.DataFrame(new_metadata)

        # Combine with existing metadata
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove any duplicates, keeping the newest entry
            combined_df = combined_df.drop_duplicates(subset='object_id', keep='last')
        else:
            combined_df = new_df
        # Save combined metadata
        combined_df.to_csv('data/metadata.csv', index=False)
        public_domain_count = combined_df['is_public_domain'].sum() if only_public_domain else len(df)
        print(f"\nSaved metadata for {len(combined_df)} paintings ({public_domain_count} public domain)")
        return combined_df
    else:
        if not existing_df.empty:
            print("No new paintings found, keeping existing metadata")
            return existing_df
        else:
            print("No paintings with images found")
            return pd.DataFrame()

if __name__ == "__main__":
    try:
        # Load existing data first
        metadata_file = 'data/metadata.csv'
        existing_downloads = get_existing_downloads()
        
        if os.path.exists(metadata_file):
            existing_df = pd.read_csv(metadata_file)
            existing_ids = set(existing_df['object_id'])
        else:
            existing_df = pd.DataFrame()
            existing_ids = set()
            
        # Combine both existing metadata and downloads
        all_existing_ids = existing_ids.union(existing_downloads)
        print(f"Found {len(all_existing_ids)} total existing processed objects")
        
        print("Filtering objects (excluding existing)...")
        all_filtered_ids = filter_objects_for_ml(
            sample_size=500, 
            only_open_access=True,
            existing_ids=all_existing_ids
        )
        print(f"Found {len(all_filtered_ids)} new objects to process")
        
        if len(all_filtered_ids) > 0:
            print("Creating dataset...")
            dataset = create_ml_dataset(all_filtered_ids, target_classification="Paintings", only_public_domain=True)
            
            if not dataset.empty:
                print(f"\nDownloading {len(dataset)} new images...")
                successful_count = download_images_sequential(dataset)
                print(f"\nImage download complete! Successfully downloaded {successful_count} new images")
            else:
                print("No new images found to download")
        else:
            print("No new objects found to process")
            
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
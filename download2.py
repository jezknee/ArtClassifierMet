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
def filter_objects_for_ml(sample_size=10, only_open_access=True):
    search_params = {
        'hasImages': 'true',
        'q': 'Paintings',  # Search for paintings specifically
    }
    
    # Add Open Access filter to get only public domain images
    if only_open_access:
        search_params['isOnView'] = 'true'  # Often correlates with open access
    
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
        
        # Sample if needed
        if len(object_ids) > sample_size:
            object_ids = np.random.choice(object_ids, sample_size, replace=False)
        
        return object_ids
    except Exception as e:
        print(f"Error filtering objects: {e}")
        return []

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
    
    successful_downloads = []
    failed_downloads = []
    
    for idx, row in dataset_df.iterrows():
        print(f"Processing image {idx + 1}/{len(dataset_df)}")
        result = download_and_process_image(row)
        
        if result['success']:
            successful_downloads.append(result)
        else:
            failed_downloads.append(result)
        
        # Rate limiting between downloads - increased delay
        time.sleep(1.0)  # Increased from 0.1 to 1 second
    
    print(f"Sequential download complete: {len(successful_downloads)} successful, {len(failed_downloads)} failed")
    
    if failed_downloads:
        print("Failed downloads:")
        for failed in failed_downloads:
            print(f"  Object {failed['object_id']}: {failed.get('error', 'Unknown error')}")
    
    return successful_downloads

def create_ml_dataset(filtered_object_ids, target_classification="Paintings", only_public_domain=True):
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Collect metadata
    metadata_list = []
    
    print(f"Processing {len(filtered_object_ids)} objects, filtering for classification: {target_classification}...")
    if only_public_domain:
        print("Only including public domain (Open Access) artworks...")
    
    for i, object_id in enumerate(filtered_object_ids):
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
                
                metadata_list.append({
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
                print(f"Added painting: {obj_data.get('title', 'Untitled')} ({domain_status})")
            else:
                print(f"Skipped (classification: {classification})")
        
        # More conservative rate limiting
        time.sleep(0.1)
    
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        df.to_csv('data/metadata.csv', index=False)
        public_domain_count = df['is_public_domain'].sum() if only_public_domain else len(df)
        print(f"\nSaved metadata for {len(metadata_list)} paintings ({public_domain_count} public domain)")
        return df
    else:
        print("No paintings with images found")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Test with small sample
    try:
        print("Filtering objects...")
        filtered_ids = filter_objects_for_ml(sample_size=10000, only_open_access=True)  # Increased sample size
        print(f"Found {len(filtered_ids)} objects")
        
        if len(filtered_ids) > 0:
            print("Creating dataset...")
            dataset = create_ml_dataset(filtered_ids, target_classification="Paintings", only_public_domain=True)
            print(dataset.head())
            print("Dataset creation complete!")
            
            if not dataset.empty:
                print("\nDownloading images...")
                
                # Choose download method (sequential is more stable)
                use_parallel = False  # Set to True for parallel downloads
                
                if use_parallel:
                    successful_downloads = download_images_parallel(dataset.to_dict('records'))
                else:
                    successful_downloads = download_images_sequential(dataset)
                
                print(f"\nImage download complete! Successfully downloaded {len(successful_downloads)} images")
                
                # Save download results
                if successful_downloads:
                    download_df = pd.DataFrame(successful_downloads)
                    download_df.to_csv('data/download_results.csv', index=False)
                    print("Download results saved to 'data/download_results.csv'")
            else:
                print("No dataset to download images for")
        else:
            print("No objects found to process")
            
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
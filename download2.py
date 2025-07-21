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
    
    time.sleep(0.05)  # Increased delay
    
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

    time.sleep(0.05)  # Increased delay
    
    try:
        response = get_with_retry(session, f"{BASE_URL}/objects/{object_id}", headers=headers)
        return response.json()
    except Exception as e:
        print(f"Error getting object details for {object_id}: {e}")
        return {}

# Filter for objects with images and specific criteria
def filter_objects_for_ml(sample_size=10):
    search_params = {
        'hasImages': 'true',
        'q': '*',
    }
    
    session = create_session_with_retries()
    headers = {
        'User-Agent': 'MetMuseum-ML-Dataset/1.0',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }
    
    time.sleep(0.05)
    
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
    object_id = object_data['objectID']
    image_url = object_data.get('primaryImage')
    
    if not image_url:
        return None
    
    # Create new session for image download
    session = create_session_with_retries()
    headers = {
        'User-Agent': 'MetMuseum-ML-Dataset/1.0',
    }
    
    try:
        # Create directories if they don't exist
        os.makedirs("Data/Images/Raw", exist_ok=True)
        os.makedirs("Data/Images/Processed", exist_ok=True)
        
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
def download_images_parallel(object_data_list, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_and_process_image, object_data_list))
    
    return [r for r in results if r and r['success']]

def create_ml_dataset(filtered_object_ids):
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Collect metadata
    metadata_list = []
    
    print(f"Processing {len(filtered_object_ids)} objects...")
    
    for i, object_id in enumerate(filtered_object_ids):
        print(f"Processing object {i+1}/{len(filtered_object_ids)}: {object_id}")
        
        obj_data = get_object_details(object_id)
        
        if obj_data.get('primaryImage'):  # Has image
            metadata_list.append({
                'object_id': obj_data['objectID'],
                'department': obj_data.get('department', 'Unknown'),
                'object_name': obj_data.get('objectName', 'Unknown'),
                'culture': obj_data.get('culture', 'Unknown'),
                'period': obj_data.get('period', 'Unknown'),
                'medium': obj_data.get('medium', 'Unknown'),
                'classification': obj_data.get('classification', 'Unknown'),
                'image_url': obj_data['primaryImage']
            })
        
        # More conservative rate limiting
        time.sleep(0.05)
    
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        df.to_csv('data/metadata.csv', index=False)
        print(f"Saved metadata for {len(metadata_list)} objects")
        return df
    else:
        print("No objects with images found")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Test with small sample
    try:
        print("Filtering objects...")
        filtered_ids = filter_objects_for_ml(sample_size=5)
        print(f"Found {len(filtered_ids)} objects")
        
        if len(filtered_ids) > 0:
            print("Creating dataset...")
            dataset = create_ml_dataset(filtered_ids)
            print(dataset.head())
            print("Dataset creation complete!")
        else:
            print("No objects found to process")
            
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
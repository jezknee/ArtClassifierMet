import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

# Base API URL
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

def create_session_with_retries():
    
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Get all object IDs
def get_all_objects():
    session = create_session_with_retries()

    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive'
    }
    time.sleep(0.015)
    response = session.get(f"{BASE_URL}/objects", headers=headers, timeout=30)
    return response.json()['objectIDs']

# Get object details with image URLs
def get_object_details(object_id):
    session = create_session_with_retries()
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive'
    }

    time.sleep(0.015)
    response = session.get(f"{BASE_URL}/objects/{object_id}", headers=headers, timeout=30)
    return response.json()

# Filter for objects with images and specific criteria
def filter_objects_for_ml(sample_size=10):
    # Use search API to get objects with images
    search_params = {
        'hasImages': 'true',
        'q': '*',  # Get all
    }
    time.sleep(0.015)
    session = create_session_with_retries()
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive'
    }
    response = session.get(f"{BASE_URL}/search", params=search_params, headers=headers, timeout=30)
    object_ids = response.json()['objectIDs']
    
    # Sample if needed
    if len(object_ids) > sample_size:
        object_ids = np.random.choice(object_ids, sample_size, replace=False)
    
    return object_ids

import cv2
from concurrent.futures import ThreadPoolExecutor
import hashlib

def download_and_process_image(object_data, target_size=(224, 224)):
    """Download and preprocess image for ML"""
    object_id = object_data['objectID']
    image_url = object_data.get('primaryImage')
    
    if not image_url:
        return None
    
    try:
        # Download image
        response = session.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Save raw image
        raw_path = f"Data/Images/Raw/{object_id}.jpg"
        with open(raw_path, 'wb') as f:
            f.write(response.content)
        
        # Process for ML
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
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

# Parallel downloading
def download_images_parallel(object_data_list, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_and_process_image, object_data_list))
    
    return [r for r in results if r and r['success']]

def create_ml_dataset(filtered_object_ids):
    # Collect metadata
    metadata_list = []
    
    for object_id in filtered_object_ids:
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
        
        # Rate limiting - API allows 80 requests/second
        time.sleep(0.015)
    
    df = pd.DataFrame(metadata_list)
    df.to_csv('data/metadata.csv', index=False)
    return df
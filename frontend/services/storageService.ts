import { API_ENDPOINTS } from './api';
import { StorageUploadResult } from '../types/chat';

import { fetchWithAuth } from '@/lib/auth';
// @ts-ignore
const fetch = fetchWithAuth;

export const storageService = {
  /**
   * Upload files to storage service
   * @param files List of files to upload
   * @param folder Optional folder path
   * @returns Upload result
   */
  async uploadFiles(
    files: File[],
    folder: string = 'attachments'
  ): Promise<StorageUploadResult> {
    // Create FormData object
    const formData = new FormData();
    
    // Add files
    files.forEach(file => {
      formData.append('files', file);
    });
    
    // Add folder parameter
    formData.append('folder', folder);
    
    // Send request
    const response = await fetch(API_ENDPOINTS.storage.upload, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      // Special handling for 405 errors which indicate routing problems
      if (response.status === 405) {
        throw new Error(`Routing error: Upload endpoint not found. Please check service configuration.`);
      }
      throw new Error(`Failed to upload files to storage: ${response.statusText}`);
    }
    
    return await response.json();
  },
  
  /**
   * Get the URL of a single file
   * @param objectName File object name
   * @returns File URL
   */
  async getFileUrl(objectName: string): Promise<string> {
    const response = await fetch(API_ENDPOINTS.storage.file(objectName));
    
    if (!response.ok) {
      throw new Error(`Failed to get file URL from storage: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.url;
  },

  /**
   * Preprocess files
   * @param files List of files to preprocess
   * @param query Query string for preprocessing
   * @param folder Optional folder path
   * @returns Response from preprocessing service
   */
  async preprocessFiles(
    files: File[],
    query: string,
    folder: string = 'attachments'
  ) {
    // Create FormData object
    const formData = new FormData();
    
    // Add files
    files.forEach(file => {
      formData.append('files', file);
    });
    
    // Add query parameter
    formData.append('query', query);
    
    // Add folder parameter
    formData.append('folder', folder);
    
    // Send request to runtime API endpoint
    const response = await fetch(API_ENDPOINTS.storage.preprocess, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      // Special handling for 405 errors which indicate routing problems
      if (response.status === 405) {
        throw new Error(`Routing error: Preprocessing endpoint not found. Please check service configuration.`);
      }
      throw new Error(`Failed to preprocess files: ${response.statusText}`);
    }
    
    return response;
  }
};
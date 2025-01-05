"use server";

import minioClient from './minioClient';

import { v4 as uuidv4 } from 'uuid';

export const generateUploadUrl = async (originalFileName: string, bucketName: string) => {
  const expiry = 24 * 60 * 60; // 24 hours

  try {
    const bucketExists = await minioClient.bucketExists(bucketName);
    if (!bucketExists) {
      console.log(`Bucket '${bucketName}' does not exist. Creating...`);
      await minioClient.makeBucket(bucketName, 'us-east-1');
      console.log(`Bucket '${bucketName}' created successfully.`);
    }

    const uniqueId = uuidv4();

    const objectKey = `${uniqueId}-${originalFileName}`;

    const presignedUrl = await minioClient.presignedPutObject(
      bucketName,
      objectKey,
      expiry
    );

    // const presignedExternalUrl = presignedUrl.replace(
    //   `http://${process.env.MINIO_HOST}:${process.env.MINIO_PORT}`,
    //   `http://${process.env.MINIO_EXTERNAL_HOST}:${process.env.MINIO_EXTERNAL_PORT}`
    // );

    console.log('Generated Pre-signed Upload URL:', presignedUrl);

    console.log('Generated Pre-signed External Upload URL:', presignedUrl);

    console.log('Generated Object Key:', objectKey);
    return { presignedUrl, objectKey };
  } catch (error) {
    console.log('Error generating pre-signed upload URL:', error);
    return { error: `Failed to generate pre-signed upload URL ${error}` };
  }
};


export const uploadFile = async (fileForm: FormData, url: string) => {
  try {

    const file = fileForm.get('file');

    const response = await fetch(url, {
      method: 'PUT',
      body: file,
    });
    
    console.log("Response status:", response.status);
    console.log("Response headers:", response.headers);
    const responseBody = await response.text();
    console.log("Response body:", responseBody);
    
    if (response.ok) {
      console.log('File uploaded successfully');
      return { success: true };
    } else {
      console.error('Failed to upload file:', response.statusText);
      return { error: `Failed to upload file: ${responseBody}` };
    }
    
  } catch (error) {
    console.error('Error uploading file:', error);
    return { error: 'Failed to upload file' };
  }
}
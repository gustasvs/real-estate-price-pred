"use server";

import minioClient from './minioClient';

import { v4 as uuidv4 } from 'uuid';

export const generateUploadUrl = async (originalFileName: string, bucketName: string) => {
  const expiry = 24 * 60 * 60; // URL expiry time in seconds (24 hours)

  try {
    // Ensure the bucket exists, and create it if it doesn't
    const bucketExists = await minioClient.bucketExists(bucketName);
    if (!bucketExists) {
      console.log(`Bucket '${bucketName}' does not exist. Creating...`);
      await minioClient.makeBucket(bucketName, 'us-east-1'); // Replace 'us-east-1' with your preferred region
      console.log(`Bucket '${bucketName}' created successfully.`);
    }

    // Generate a unique object key
    const uniqueId = uuidv4(); // Create a unique ID
    const objectKey = `${uniqueId}-${originalFileName}`;

    // Generate pre-signed PUT URL
    const presignedUrl = await minioClient.presignedPutObject(
      bucketName,
      objectKey,
      expiry
    );

    console.log('Generated Pre-signed Upload URL:', presignedUrl);
    return { presignedUrl, objectKey };
  } catch (error) {
    console.error('Error generating pre-signed upload URL:', error);
    return { error: 'Failed to generate pre-signed upload URL' };
  }
};
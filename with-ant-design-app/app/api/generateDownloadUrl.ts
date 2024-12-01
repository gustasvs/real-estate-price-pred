"use server";

import minioClient from './minioClient';

// Function to generate a pre-signed download URL
export const generateDownloadUrl = async (fileName: string, bucketName: string) => {
  const expiry = 24 * 60 * 60; // URL expiry time in seconds (24 hours)

  try {
    // Ensure the bucket exists before attempting to generate the URL
    const bucketExists = await minioClient.bucketExists(bucketName);
    if (!bucketExists) {
      console.error(`Bucket '${bucketName}' does not exist.`);
      return { error: `Bucket '${bucketName}' does not exist.` };
    }

    // Generate the pre-signed download URL
    const presignedUrl = await minioClient.presignedGetObject(
      bucketName,
      fileName,
      expiry
    );
    // console.log('Pre-signed download URL:', presignedUrl);
    return presignedUrl;
  } catch (error) {
    console.error('Error generating pre-signed download URL:', error);
    return { error: 'Failed to generate pre-signed download URL' };
  }
};

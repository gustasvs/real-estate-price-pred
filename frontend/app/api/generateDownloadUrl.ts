"use server";

import minioClient from './minioClient';


export const generateDownloadUrl = async (fileName: string, bucketName: string) => {

  // cheeky fix while minio can't be accessed from the frontend
  return "/api/getImage?fileName=" + fileName + "&bucketName=" + bucketName;
  
  const expiry = 24 * 60 * 60; // 24 hours

  try {
    const bucketExists = await minioClient.bucketExists(bucketName);
    if (!bucketExists) {
      console.error(`Bucket '${bucketName}' does not exist.`);
      return { error: `Bucket '${bucketName}' does not exist.` };
    }

    const presignedUrl = await minioClient.presignedGetObject(
      bucketName,
      fileName,
      expiry
    );
    const presignedExternalUrl = presignedUrl.replace(
      `http://${process.env.MINIO_HOST}:${process.env.MINIO_PORT}`,
      // `http://${process.env.MINIO_EXTERNAL_HOST}/files`
      `http://localhost:3000/files`
    );

    return presignedUrl;
  } catch (error) {
    console.error('Error generating pre-signed download URL:', error);
    return { error: 'Failed to generate pre-signed download URL' };
  }
};

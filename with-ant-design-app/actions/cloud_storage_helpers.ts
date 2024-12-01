// Import required modules
import { useSession } from "next-auth/react";
import { auth } from "../auth";

// Set up MinIO credentials and host information
const minioUrl = `http://${process.env.MINIO_HOST}:${process.env.MINIO_PORT}`;

export const saveImageOnCloud = async (formData: FormData) => {
  const session = await auth();

  const user = session?.user;

  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  const userId = user.id;

  try {
    const minioResponse = await fetch(`${minioUrl}/bucket-name/upload`, {
        method: "POST",
        headers: {
          'Authorization': `Basic ${Buffer.from(`${process.env.MINIO_ROOT_USER}:${process.env.MINIO_ROOT_PASSWORD}`).toString('base64')}`,
        },
        body: formData,
      }); 
    const minioData = await minioResponse.json();

    return minioData;
  } catch (error) {
    console.error("Error posting image to MinIO:", error);
    return { error: "Failed to post image to MinIO" };
  }
};

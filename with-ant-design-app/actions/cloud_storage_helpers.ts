"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";


// Get one group
export const saveImageOnCloud = async (formData: FormData) => {
  const session = await auth();
  console.log("session in groups", session);
  const user = session?.user;
  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  const userId = user.id;
  
  try {
    const cloudinaryResponse = await fetch(`https://api.cloudinary.com/v1_1/${process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME}/image/upload`, {
        method: "POST",
        body: formData,
      });
    const cloudinaryData = await cloudinaryResponse.json();

    return cloudinaryData;


  } catch (error) {
    console.error("Error posting image to cloud:", error);
    return { error: "Failed to post image to cloud" };
  }
};

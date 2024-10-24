"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";

export const updateUserProfile = async (profileData: { name: string }) => {
    const session = await auth();
    const user = session?.user;
  
    if (!user || !user.id) {
      return { error: "Unauthorized" };
    }
  
    try {
      const updatedUser = await db.user.update({
        where: { id: user.id },
        data: { name: profileData.name },
      });
      return updatedUser;
    } catch (error) {
      console.error("Error updating user profile:", error);
      return { error: "Failed to update user profile" };
    }
  };
  
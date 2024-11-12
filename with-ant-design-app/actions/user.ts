"use server";

import { useSession } from "next-auth/react";
import { db } from "../db";
import { auth } from "../auth";
import { saltAndHashPassword } from "../utils/helper";

const checkPassword = async (password: string, confirmPassword: string) => {
  if (password !== confirmPassword) {
    return { error: "Passwords do not match" };
  }
  return true;
};

export const updateUserProfile = async (profileData: {
  id: string;
  name: string | null;
  email: string | null;
  password: string;
  confirmPassword: string;
  image: string | null;
}) => {
  const session = await auth();
  const user = session?.user;

  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  if (user.id !== profileData.id) {
    return { error: "Unauthorized" };
  }

  let passwordHash: string | undefined;

  if (profileData.password) {
    const passwordCheck = await checkPassword(
      profileData.password,
      profileData.confirmPassword
    );
    if (passwordCheck !== true) {
      return { error: `Error: passwords doesnt match` };
    }
    passwordHash = saltAndHashPassword(profileData.password); // Hash the new password
  }

  try {
    const updatedUser = await db.user.update({
      where: { id: user.id },
      data: {
        // name: profileData.name,
        // email: profileData.email,
        // image: profileData.image,
        ...(profileData.name && { name: profileData.name }),
        ...(profileData.email && { email: profileData.email }),
        ...(profileData.image && { image: profileData.image }),
        ...(passwordHash && { hashedPassword: passwordHash }), // Only update if password is provided
      },
    });
    return { user: updatedUser };
  } catch (error) {
    console.error("Error updating user profile:", error);
    return { error: "Failed to update user profile" };
  }
};


export const updateUserTheme = async (theme: string) => {
  const session = await auth();
  const user = session?.user;

  if (!user || !user.id) {
    return { error: "Unauthorized" };
  }

  try {
    const updatedUser = await db.user.update({
      where: { id: user.id },
      data: {
        theme,
      },
    });
    return { user: updatedUser };
  } catch (error) {
    console.error("Error updating user theme:", error);
    return { error: "Failed to update user theme" };
  }
}

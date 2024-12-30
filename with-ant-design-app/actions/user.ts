"use server";

import { db } from "../db";
import { generateRandomHashedString, saltAndHashPassword } from "../utils/helper";

import { getServerSession } from "next-auth";
import { authOptions } from "../auth";
import sendVerificationEmail from "./send_mail";

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
  fontSize: string | null;
  theme: string | null;
}) => {
  const session = await getServerSession(authOptions);
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

  // user is trying to update email
  if (profileData.email && profileData.email !== user.email) {
    const existingUser = await db.user.findUnique({
      where: { email: profileData.email },
    });
    if (existingUser && existingUser.id !== user.id) {
      return { error: "Email already in use" };
    }

    const verificationToken = await db.verificationToken.create({
      data: {
        identifier: user.id,
        token: generateRandomHashedString(),
        expires: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours from now
      },
    });

    console.log("verificationToken", verificationToken);

    await sendVerificationEmail(profileData.email, verificationToken.token, verificationToken.id, true);

    await db.user.update({
      where: { id: user.id },
      data: {
        email: profileData.email,
        emailVerified: null,
      },
    });

    return { message: "E-pasts veiksmīgi nomainīts. Lūdzu atveriet savu e-pastu un apstipriniet jauno e-pastu! Jums nebūs iespēja atkārtotie ieiet sistēmā kamēr e-pasts netiks apstiprināts." };
  }


  try {
    const updatedUser = await db.user.update({
      where: { id: user.id },
      data: {
        updatedAt: new Date(),
        // name: profileData.name,
        // email: profileData.email,
        // image: profileData.image,
        ...(profileData.name && { name: profileData.name }),
        ...(profileData.email && { email: profileData.email }),
        ...(profileData.image && { image: profileData.image }),
        ...(passwordHash && { hashedPassword: passwordHash }), // Only update if password is provided
      
        ...(profileData.fontSize && { fontSize: profileData.fontSize }),
        ...(profileData.theme && { theme: profileData.theme }),
      },
    });
    return { user: updatedUser };
  } catch (error) {
    console.error("Error updating user profile:", error);
    return { error: "Failed to update user profile" };
  }
};


export const updateUserTheme = async (theme: string) => {
  const session = await getServerSession(authOptions);
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

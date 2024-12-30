"use client";

import { revalidatePath } from "next/cache";

import { signIn, signOut } from "next-auth/react";

import { db } from "../db";


const getUserByEmail = async (email: string) => {
  try {
    const user = await db.user.findUnique({
      where: {
        email,
      },
    });
    return user;
  } catch (error) {
    console.log(error);
    return null;
  }
};

export const login = async (provider: string) => {
  await signIn(provider, { redirectTo: "/" });
};

export const logout = async () => {

  await signOut({ redirect: false, callbackUrl: "/" });
};
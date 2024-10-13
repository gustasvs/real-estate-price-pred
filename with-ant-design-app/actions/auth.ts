"use server";

import { revalidatePath } from "next/cache";
import { signIn, signOut } from "../auth";
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
  revalidatePath("/");
};

export const logout = async () => {

  await signOut({ redirectTo: "/" });
  revalidatePath("/");
};

export const loginWithCreds = async (formData: FormData) => {
  const rawFormData = {
    email: formData.get("email"),
    password: formData.get("password"),
    role: "ADMIN",
    redirectTo: "/",
  };

  console.log("rawFormData", rawFormData);

  const existingUser = await getUserByEmail(formData.get("email") as string);
  console.log("existingUser", existingUser);

  try {
    await signIn("credentials", rawFormData);
  } catch (error: any) {
    if (error instanceof Error) {
      if (error.message.includes("CredentialsSignin")) {
        return { error: "Invalid credentials!" };
      } else {
        return { error: "Something went wrong!" };
      }
    }

    throw error;
  }
  revalidatePath("/");
};

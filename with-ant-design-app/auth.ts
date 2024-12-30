import NextAuth from "next-auth";

import Twitter from "next-auth/providers/twitter";
import Github from "next-auth/providers/github";
import { PrismaAdapter } from "@auth/prisma-adapter";
import Credentials from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { db } from "./db";
import { generateRandomHashedString, saltAndHashPassword } from "./utils/helper";
import sendVerificationEmail from "./actions/send_mail";
import { redirect } from "next/dist/server/api-utils";

console.log("auth.ts");

export const authOptions = {
  adapter: PrismaAdapter(db),
  callbacks: {
    // this is what gets returned when await auth() is called
    
    session: async ({ session, token }) => {
      // console.log("session callback", session, token);
      if (token.sub) {
        // Fetch the updated user data from the database using the user's ID (token.sub)
        const updatedUser = await db.user.findUnique({
          where: { id: token.sub },
        });

        if (updatedUser) {
          session.user = {
            ...session.user,
            ...updatedUser,
            email: updatedUser.email || "", // Ensure email is always a string
            id: token.sub, // Ensure the user ID is always included
          };
        }
      }

      if (token.error) {
        session.error = token.error;
      }

      return session;
    },
    jwt: async ({ user, token }) => {
      // console.log("jwt callback", user, token);

      if (user) {

        if (user.error) {
          token.error = user.error;
        }

        token.sub = user.id;

        token.theme = user.theme || "dark";

        token.picture = "none"; // do NOT include picture in token
        token.expires = Math.floor(Date.now() / 1000) + (24 * 60 * 60); // 24 hours
      }
      // console.log("token returned after jwt callback", token);
      return token;
    },
  },
  session: { strategy: "jwt", maxAge: 24 * 60 * 60 }, // 24 hours
  secret: process.env.JWT_AUTH_SECRET || "secret",
  providers: [
    // Github({
    //   clientId: process.env.AUTH_GITHUB_ID || "",
    //   clientSecret: process.env.AUTH_GITHUB_SECRET || "", 
    // }),
    // Twitter({
    //   clientId: process.env.TWITTER_ID,
    //   clientSecret: process.env.TWITTER_SECRET
    // }),
    Credentials({
      name: "Credentials",
      credentials: {
        email: {
          label: "Email",
          type: "email",
          placeholder: "email@example.com",
        },
        password: { label: "Password", type: "password" },
        confirmPassword: { label: "Confirm Password", type: "password", optional: true },
      },
      authorize: async (credentials) => {
        console.log("called authorise with credentials", credentials);
        if (!credentials) {
          return null;
        }
        if (!credentials || !credentials.email || !credentials.password) {
          return { error: "Visi ievadlauki nav aizpildīti." }
        }

        const email = credentials.email as string;
        const password = credentials.password as string;
        const confirmPassword = credentials.confirmPassword as string;

        // check if user is trying to register
        if (confirmPassword) {
          if (password !== confirmPassword) {
            return { error: "Parole atkārtoti nesakrīt ar paroli!" }
          }
          const hash = saltAndHashPassword(password);
          let user: any = await db.user.findUnique({
            where: {
              email,
            },
          });
          if (user) {
            // user already exists
            return { error: "Lietotājs ar šādu e-pastu jau eksistē" };
          }
          // create new user
          user = await db.user.create({
            data: {
              email,
              hashedPassword: hash,
            },
          });

          const verificationToken = await db.verificationToken.create({
            data: {
              identifier: user.id,
              token: generateRandomHashedString(),
              expires: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours from now
            },
          });

          await sendVerificationEmail(email, verificationToken.token, verificationToken.id);

          // redirect to verify email page

          return { error: "Lūdzu apstipriniet savu e-pastu" };
        }

        let user: any = await db.user.findUnique({
          where: {
            email,
          },
        });

        console.log("user", user);

        if (!user) {
          console.log("User not found");
          return { error: "Lietotājs ar šādu e-pastu nav atrasts" };
        }

        if (!user.emailVerified) {
          console.log("Email not verified");
          return { error: "Jūsu e-pasts nav apstiprināts. Pārbaudiet savu e-pastu un sekojiet norādījumiem!" };
        }

        const isMatch = bcrypt.compareSync(
          credentials.password as string,
          user.hashedPassword
        );
        console.log("isMatch", isMatch);
        // const isMatch = true;
        if (!isMatch) {
          return { error: "Nepareizs e-pasts vai parole" };
        }

        return user;
      },
    }),
  ],
};

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST }

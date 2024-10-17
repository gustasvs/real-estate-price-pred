import NextAuth from "next-auth";

import Twitter from "next-auth/providers/twitter";
import Github from "next-auth/providers/github";
import { PrismaAdapter } from "@auth/prisma-adapter";
import Credentials from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { db } from "./db";
import { saltAndHashPassword } from "./utils/helper";

console.log("auth.ts");

export const {
  handlers: { GET, POST },
  signIn,
  signOut,
  auth,
} = NextAuth({
  adapter: PrismaAdapter(db),
  callbacks: {
    // this is what gets returned when await auth() is called
    
    session: async ({ session, token }) => {
      if (session?.user && token.sub) {
          session.user.id = token.sub;
      }
      return session;
    },
    jwt: async ({ user, token }) => {
      console.log("jwt callback", user, token);
      if (user) {
        token.sub = user.id;
      }
      return token;
    },
  },
  session: { strategy: "jwt" },
  secret: process.env.JWT_AUTH_SECRET || "secret",
  providers: [
    Github({
      clientId: process.env.AUTH_GITHUB_ID || "",
      clientSecret: process.env.AUTH_GITHUB_SECRET || "", 
    }),
    Twitter({
      clientId: process.env.TWITTER_ID,
      clientSecret: process.env.TWITTER_SECRET
    }),
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
          return null;
        }

        const email = credentials.email as string;
        const password = credentials.password as string;
        const confirmPassword = credentials.confirmPassword as string;

        // check if user is trying to register
        if (confirmPassword) {
          if (password !== confirmPassword) {
            return null;
          }
          const hash = saltAndHashPassword(password);
          let user: any = await db.user.findUnique({
            where: {
              email,
            },
          });
          if (user) {
            // user already exists
            return null;
          }
          // create new user
          user = await db.user.create({
            data: {
              email,
              hashedPassword: hash,
            },
          });
          return user;
        }

        let user: any = await db.user.findUnique({
          where: {
            email,
          },
        });

        if (!user) {
          console.log("User not found");
          return null;
        }

        console.log("User found, checking password", user);
        const isMatch = bcrypt.compareSync(
          credentials.password as string,
          user.hashedPassword
        );
        console.log("isMatch", isMatch);
        // const isMatch = true;
        if (!isMatch) {
          return null;
        }

        return user;
      },
    }),
  ],
});

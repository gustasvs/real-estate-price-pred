import NextAuth from "next-auth"
import { JWT } from "next-auth/jwt";
import { PrismaAdapter } from "@next-auth/prisma-adapter";

declare module "next-auth" {
  /**
   * Returned by `useSession`, `getSession` and received as a prop on the `SessionProvider` React Context
   */
  interface Session {
    user: {
      theme: string | null;
      fontSize: string | null;
      id : string;
      email: string;
      name: string;
      image: string;
      
    }
    adaptedUser: {
        /** The user's postal address. */
        theme: string | null;
        fontSize: string | null;
        
      }
  }
  
  interface User {
    theme: string | null;
        fontSize: string | null;
  }

  interface AdaptedUser {
    theme: string | null;
        fontSize: string | null;
  }
}

declare module "next-auth/jwt" {
    interface JWT {
        id: string;
        theme: string | null;
        fontSize: number | null;
    }
  }
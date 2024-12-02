import { getToken } from "next-auth/jwt";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const protectedRoutes = ["/middleware"];

export default async function middleware(request: NextRequest) {
  const token = await getToken({ req: request, secret: process.env.JWT_AUTH_SECRET || "secret" });
  // console.log("Middleware.ts Token", token);

  if (!token && protectedRoutes.some(route => request.nextUrl.pathname.startsWith(route))) {
    const absoluteURL = new URL("/", request.nextUrl.origin);
    console.error("Unauthorized access attempt, redirecting to home.");
    return NextResponse.redirect(absoluteURL.toString());
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};

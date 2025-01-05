import { getToken } from "next-auth/jwt";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const protectedRoutes = ["/middleware", "/profile", "/groups", "/files"];
const fileAccessRoutes = ["/files"];

export default async function middleware(request: NextRequest) {
  const token = await getToken({ req: request, secret: process.env.JWT_AUTH_SECRET || "secret" });
  // console.log("Middleware.ts Token", token);

  if (!token && protectedRoutes.some(route => request.nextUrl.pathname.startsWith(route))) {
    const absoluteURL = new URL("/", request.nextUrl.origin);
    console.error("Unauthorized access attempt, redirecting to home.");
    return NextResponse.redirect(absoluteURL.toString());
  }

  if (fileAccessRoutes.some(route => request.nextUrl.pathname.startsWith(route))) {
    const baseUrl = `http://${process.env.MINIO_HOST}:${process.env.MINIO_PORT}`;
    // should remove /files from the path
    const newUrl = new URL(baseUrl + request.nextUrl.pathname.replace("/files", ""));
    console.log("Redirecting file access through backend service.");
    return NextResponse.rewrite(newUrl.toString());
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};

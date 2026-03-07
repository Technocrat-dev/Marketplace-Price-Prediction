export { auth as middleware } from "@/auth";

export const config = {
    matcher: [
        // Protect all routes except auth API, static files, and favicon
        "/((?!api/auth|_next/static|_next/image|favicon.ico).*)",
    ],
};

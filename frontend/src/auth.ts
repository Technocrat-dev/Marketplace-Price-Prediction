import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

export const { handlers, signIn, signOut, auth } = NextAuth({
    providers: [
        Google({
            clientId: process.env.AUTH_GOOGLE_ID!,
            clientSecret: process.env.AUTH_GOOGLE_SECRET!,
            authorization: {
                params: {
                    prompt: "consent",
                    access_type: "offline",
                    response_type: "code",
                },
            },
        }),
    ],
    session: { strategy: "jwt" },
    trustHost: true,
    pages: {
        signIn: "/login",
    },
    callbacks: {
        authorized({ auth, request }) {
            const isLoggedIn = !!auth?.user;
            const path = request.nextUrl.pathname;

            // Allow login page and auth error page
            if (path === "/login") return true;

            // Block everything else if not signed in
            return isLoggedIn;
        },
        async jwt({ token, profile }) {
            if (profile) {
                token.id = profile.sub;
            }
            return token;
        },
        async session({ session, token }) {
            if (session.user && token.id) {
                session.user.id = token.id as string;
            }
            return session;
        },
    },
});

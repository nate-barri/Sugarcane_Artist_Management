import { StackClientApp } from "@stackframe/stack"

export const stackClientApp = new StackClientApp({
  projectId: process.env.NEXT_PUBLIC_STACK_PROJECT_ID || "d361ac68-22e9-4507-93a5-ec9697dba04a",
  publishableClientKey:
    process.env.NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY || "pck_25df23aqkpdymvp0ahkwa2b7b0320pc10mrjdmg9x0ze0",
  tokenStore: "nextjs-cookie",
  urls: {
    signIn: "/login",
    afterSignIn: "/import",
    afterSignOut: "/login",
  },
})

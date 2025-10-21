import type { Metadata } from "next"
import { Inter, Roboto_Mono } from "next/font/google"
import { StackProvider, StackTheme } from "@stackframe/stack"
import { stackServerApp } from "@/stack"
import { AuthProvider } from "@/components/auth-provider"
import type React from "react"
import { Suspense } from "react"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
})

const robotoMono = Roboto_Mono({
  subsets: ["latin"],
  variable: "--font-roboto-mono",
})

export const metadata: Metadata = {
  title: "Dashboard App",
  description: "Social Media Analytics Dashboard",
}

function AuthProviderFallback() {
  return <div className="flex items-center justify-center min-h-screen">Loading...</div>
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  const projectId = process.env.NEXT_PUBLIC_STACK_PROJECT_ID
  const publishableClientKey = process.env.NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY

  return (
    <html lang="en" className={`${inter.variable} ${robotoMono.variable}`}>
      <body className="antialiased" suppressHydrationWarning>
        <StackProvider app={stackServerApp} projectId={projectId} publishableClientKey={publishableClientKey}>
          <StackTheme>
            <Suspense fallback={<AuthProviderFallback />}>
              <AuthProvider>{children}</AuthProvider>
            </Suspense>
          </StackTheme>
        </StackProvider>
      </body>
    </html>
  )
}

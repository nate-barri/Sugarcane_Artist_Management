import type { Metadata } from "next"
import { Inter, Roboto_Mono } from "next/font/google"
import { StackProvider, StackTheme } from "@stackframe/stack"
import { stackServerApp } from "@/stack"
import { AuthProvider } from "@/components/auth-provider"
import type React from "react"
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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${robotoMono.variable}`}>
      <body className="antialiased">
        <StackProvider app={stackServerApp}>
          <StackTheme>
            <AuthProvider>{children}</AuthProvider>
          </StackTheme>
        </StackProvider>
      </body>
    </html>
  )
}

"use client"

import { StackProvider, StackTheme } from "@stackframe/stack"
import { stackClientApp } from "@/stack-client"
import type React from "react"

export function StackAuthProvider({ children }: { children: React.ReactNode }) {
  return (
    <StackProvider app={stackClientApp}>
      <StackTheme>{children}</StackTheme>
    </StackProvider>
  )
}

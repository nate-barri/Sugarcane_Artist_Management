// inside your existing middleware.ts
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export function middleware(req: NextRequest) {
  // ✅ Add this line near the top:
  if (req.nextUrl.pathname.startsWith("/api")) return NextResponse.next()

  // ...keep ALL your existing logic below...
  return NextResponse.next()
}

// ✅ Add (or merge) this config at the bottom
export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
}

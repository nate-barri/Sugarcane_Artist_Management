import { neon } from "@neondatabase/serverless"

let sql: any = null

export function getDb() {
  if (!sql) {
    const databaseUrl = process.env.DATABASE_URL
    if (!databaseUrl) {
      throw new Error("DATABASE_URL environment variable is not set")
    }
    sql = neon(databaseUrl)
  }
  return sql
}

export async function executeQuery(query: string, params: any[] = []) {
  const sql = getDb()
  return sql.query(query, params)
}

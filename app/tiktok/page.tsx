"use client";

import Sidebar from "@/components/sidebar";
import { useEffect, useMemo, useState } from "react";

// Recharts
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

// ===== Types =====
type VideoRow = { video_title: string; total_views: number };
type MonthlyRow = { publish_year: number; publish_month: number; total_views: number };
type Overview = {
  engagement_rate?: number; // 0–1 or 0–100 both supported
  total_views?: number;
  total_likes?: number;
  total_shares?: number;
};

// ===== Helpers: CSV Parsing (handles quotes, commas, BOM) =====
function parseCSV(text: string): Record<string, string>[] {
  const cleaned = text.replace(/^\uFEFF/, ""); // strip BOM
  const lines = cleaned.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (!lines.length) return [];
  const headers = splitCSVLine(lines[0]).map((h) => h.trim());

  return lines.slice(1).map((line) => {
    const cells = splitCSVLine(line);
    const row: Record<string, string> = {};
    headers.forEach((h, i) => (row[h] = (cells[i] ?? "").trim()));
    return row;
  });

  function splitCSVLine(line: string): string[] {
    const out: string[] = [];
    let cur = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        if (inQuotes && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (c === "," && !inQuotes) {
        out.push(cur);
        cur = "";
      } else {
        cur += c;
      }
    }
    out.push(cur);
    return out;
  }
}

// ===== Flexible mappers =====
function mapRowsToVideos(rows: Record<string, string>[]): VideoRow[] {
  if (!rows.length) return [];
  const keys = Object.keys(rows[0]);

  const titleKey =
    keys.find((k) => /^(video[_\s-]*title|title|caption|content)$/i.test(k)) ??
    keys.find((k) => typeof rows[0][k] === "string") ??
    "video_title";

  let viewsKey = keys.find((k) => /views?|total[_\s-]*views?/i.test(k));
  if (!viewsKey) {
    const numericScore = (k: string) =>
      rows.reduce((acc, r) => {
        const n = Number(String(r[k] ?? "").replace(/,/g, ""));
        return acc + (Number.isFinite(n) ? 1 : 0);
      }, 0);
    viewsKey = [...keys].sort((a, b) => numericScore(b) - numericScore(a))[0];
  }

  return rows
    .map((r) => {
      const title = r[titleKey!] ?? "";
      const raw = String(r[viewsKey!] ?? "").replace(/,/g, "");
      const views = Number(raw);
      return { video_title: String(title), total_views: Number.isFinite(views) ? views : NaN };
    })
    .filter((r) => r.video_title && Number.isFinite(r.total_views));
}

function mapRowsToMonthly(rows: Record<string, string>[]): MonthlyRow[] {
  if (!rows.length) return [];
  const keys = Object.keys(rows[0]);
  const findKey = (regexes: RegExp[]) =>
    keys.find((k) => regexes.some((re) => re.test(k))) ?? "";

  const yearKey = findKey([/publish[_\s-]*year/i, /^year$/i]);
  const monthKey = findKey([/publish[_\s-]*month/i, /^month$/i]);
  const viewsKey =
    keys.find((k) => /total[_\s-]*views?|views?/i.test(k)) ??
    keys.find((k) => /count|value|number/i.test(k)) ??
    keys[0];

  const out: MonthlyRow[] = [];

  if (yearKey && monthKey) {
    for (const r of rows) {
      const y = Number(String(r[yearKey]).replace(/\D/g, ""));
      const m = Number(String(r[monthKey]).replace(/\D/g, ""));
      const v = Number(String(r[viewsKey]).replace(/,/g, ""));
      if (Number.isFinite(y) && Number.isFinite(m) && Number.isFinite(v)) {
        out.push({ publish_year: y, publish_month: m, total_views: v });
      }
    }
  } else {
    const monthLikeKey = findKey([/^month$/i, /date/i, /period/i]);
    for (const r of rows) {
      const label = r[monthLikeKey] ?? "";
      let y = NaN, m = NaN;
      const iso = label.match(/(\d{4})[-/](\d{1,2})/);
      if (iso) {
        y = Number(iso[1]);
        m = Number(iso[2]);
      } else {
        const mName = label.match(/(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*/i);
        const yNum = label.match(/(20\d{2})/);
        if (mName && yNum) {
          const idx =
            "jan feb mar apr may jun jul aug sep oct nov dec"
              .split(" ")
              .findIndex((s) => mName[1].toLowerCase().startsWith(s)) + 1;
          m = idx;
          y = Number(yNum[1]);
        }
      }
      const v = Number(String(r[viewsKey]).replace(/,/g, ""));
      if (Number.isFinite(y) && Number.isFinite(m) && Number.isFinite(v)) {
        out.push({ publish_year: y, publish_month: m, total_views: v });
      }
    }
  }

  out.sort((a, b) => a.publish_year - b.publish_year || a.publish_month - b.publish_month);
  return out;
}

/** Map overview_metrics.csv into { engagement_rate, total_views, total_likes, total_shares }.
 *  Supports either:
 *   A) wide format (columns already named like engagement_rate,total_views,... on row 1), or
 *   B) long format with two columns like metric,value (e.g., "metric,value" rows)
 */
function mapRowsToOverview(rows: Record<string, string>[]): Overview {
  if (!rows.length) return {};

  const keys = Object.keys(rows[0]);
  const lowerKeys = keys.map((k) => k.toLowerCase());

  // Wide-format detection
  const hasWide =
    lowerKeys.some((k) => /engagement[_\s-]*rate/.test(k)) ||
    lowerKeys.some((k) => /total[_\s-]*views?/.test(k)) ||
    lowerKeys.some((k) => /total[_\s-]*likes?/.test(k)) ||
    lowerKeys.some((k) => /total[_\s-]*shares?/.test(k));

  if (hasWide) {
    const row = rows[0];
    const pickNum = (re: RegExp) => {
      const key = keys.find((k) => re.test(k.toLowerCase()));
      if (!key) return undefined;
      const raw = String(row[key]).replace(/[, %]/g, "");
      const n = Number(raw);
      return Number.isFinite(n) ? n : undefined;
    };

    let er = pickNum(/engagement[_\s-]*rate/);
    // Normalize ER: if it's 0–100, convert to 0–1 for display helper later
    if (er !== undefined && er > 1.0001) er = er / 100;

    return {
      engagement_rate: er,
      total_views: pickNum(/total[_\s-]*views?/),
      total_likes: pickNum(/total[_\s-]*likes?/),
      total_shares: pickNum(/total[_\s-]*shares?/),
    };
  }

  // Long format (metric,value)
  const metricKey =
    keys.find((k) => /^metric|name|measure$/i.test(k)) ?? keys[0];
  const valueKey =
    keys.find((k) => /^value|amount|total|number|count$/i.test(k)) ??
    keys[1] ??
    keys[0];

  const out: Overview = {};
  for (const r of rows) {
    const m = (r[metricKey] || "").toLowerCase().trim();
    const valRaw = String(r[valueKey] ?? "");
    const valNum = Number(valRaw.replace(/[, %]/g, ""));
    if (!Number.isFinite(valNum)) continue;

    if (/engagement/.test(m)) out.engagement_rate = valNum > 1.0001 ? valNum / 100 : valNum;
    else if (/views?/.test(m)) out.total_views = valNum;
    else if (/likes?/.test(m)) out.total_likes = valNum;
    else if (/shares?/.test(m)) out.total_shares = valNum;
  }
  return out;
}

// ===== Component =====
export default function TikTokDashboard() {
  const [videos, setVideos] = useState<VideoRow[]>([]);
  const [monthly, setMonthly] = useState<MonthlyRow[]>([]);
  const [overview, setOverview] = useState<Overview>({});
  const [errTop, setErrTop] = useState<string | null>(null);
  const [errMonthly, setErrMonthly] = useState<string | null>(null);
  const [errOverview, setErrOverview] = useState<string | null>(null);

  // Load Top Videos CSV
  useEffect(() => {
    fetch("/top_videos_by_views.csv", { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((txt) => setVideos(mapRowsToVideos(parseCSV(txt))))
      .catch((e) => setErrTop(String(e?.message || e)));
  }, []);

  // Load Monthly CSV
  useEffect(() => {
    fetch("/monthly_stats.csv", { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((txt) => setMonthly(mapRowsToMonthly(parseCSV(txt))))
      .catch((e) => setErrMonthly(String(e?.message || e)));
  }, []);

  // Load Overview metrics CSV
  useEffect(() => {
    fetch("/overview_metrics.csv", { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((txt) => setOverview(mapRowsToOverview(parseCSV(txt))))
      .catch((e) => setErrOverview(String(e?.message || e)));
  }, []);

  const top10 = useMemo(
    () => [...videos].sort((a, b) => b.total_views - a.total_views).slice(0, 10),
    [videos]
  );

  // ===== Formatters =====
  const fmtInt = (n?: number) =>
    typeof n === "number" && Number.isFinite(n)
      ? n.toLocaleString()
      : "—";
  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—";
    // expects 0–1; if user stored 0–100 we normalized earlier
    return `${(n * 100).toFixed(2)}%`;
  };
  const fmtCompact = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M`
    : n >= 1_000     ? `${(n / 1_000).toFixed(1)}K`
    : `${n}`;

  return (
    <div className="flex min-h-screen bg-[#123458] text-white">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">TikTok Dashboard</h1>
        </header>

        {/* KPIs driven by overview_metrics.csv */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Engagement Rate</h2>
            <p className="text-3xl font-bold text-gray-900">
              {fmtPct(overview.engagement_rate)}
            </p>
            {errOverview && (
              <p className="text-xs text-red-600 mt-1">Failed to load overview: {errOverview}</p>
            )}
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Views</h2>
            <p className="text-3xl font-bold text-gray-900">
              {fmtInt(overview.total_views)}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Likes</h2>
            <p className="text-3xl font-bold text-gray-900">
              {fmtInt(overview.total_likes)}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Shares</h2>
            <p className="text-3xl font-bold text-gray-900">
              {fmtInt(overview.total_shares)}
            </p>
          </div>
        </section>

        {/* Charts */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Highest-Performing Videos by Total Views (Top 10) */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">
              Highest-Performing Videos by Total Views
            </h2>

            <div className="h-96">
              {errTop ? (
                <div className="text-red-600">
                  Failed to load CSV: {errTop}. Check that
                  <code className="mx-1 text-[#123458]">/public/top_videos_by_views.csv</code>
                  exists, opens at <code>/top_videos_by_views.csv</code>, and has a title column and a numeric views column.
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={top10}
                    layout="vertical"
                    margin={{ top: 12, right: 24, bottom: 12, left: 24 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" tickFormatter={fmtCompact} />
                    <YAxis
                      type="category"
                      dataKey="video_title"
                      width={220}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip
                      formatter={(v: any) =>
                        Number.isFinite(v) ? v.toLocaleString() : String(v)
                      }
                    />
                    <Legend />
                    <Bar dataKey="total_views" name="Views" fill="#0c4d8fff" />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Total Views by Month (Line) */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Total Views by Month</h2>

            <div className="h-96">
              {errMonthly ? (
                <div className="text-red-600">
                  Failed to load CSV: {errMonthly}. Check that
                  <code className="mx-1 text-[#123458]">/public/monthly_stats.csv</code>
                  exists, opens at <code>/monthly_stats.csv</code>, and has year/month + total_views columns.
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart
                    data={monthly.map((m) => ({
                      month: `${m.publish_year}-${String(m.publish_month).padStart(2, "0")}`,
                      total_views: m.total_views,
                    }))}
                    margin={{ top: 16, right: 24, bottom: 32, left: 24 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" angle={-45} textAnchor="end" height={60} />
                    <YAxis tickFormatter={(v) => (Number.isFinite(v) ? v.toLocaleString() : String(v))} />
                    <Tooltip
                      labelFormatter={(l) => `Month: ${l}`}
                      formatter={(v: any) => (Number.isFinite(v) ? v.toLocaleString() : String(v))}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="total_views"
                      name="Month"
                      stroke = "#2c0379ff"
                      strokeWidth={2}
                      dot
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

"use client";

import Sidebar from "@/components/sidebar";
import { useEffect, useState } from "react";
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
  ScatterChart,
  Scatter,
} from "recharts";

export default function SpotifyDashboard() {
  const [overview, setOverview] = useState<any>({});
  const [topTracks, setTopTracks] = useState<any[]>([]);
  const [monthly, setMonthly] = useState<any[]>([]);
  const [engagement, setEngagement] = useState<any[]>([]);
  const [dailyStreams, setDailyStreams] = useState<any[]>([]);
  const [songReleases, setSongReleases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch overview, top tracks, monthly data, and engagement
        const [overviewRes, topTracksRes, monthlyRes, engagementRes, dailyStreamsRes] = await Promise.all([
          fetch("/api/analytics/spotify/overview"),
          fetch("/api/analytics/spotify/top-tracks"),
          fetch("/api/analytics/spotify/monthly"),
          fetch("/api/analytics/spotify/engagement"),
          fetch("/api/analytics/spotify/daily-streams-with-releases"), // Fetch Daily Streams with Song Releases
        ]);

        if (!overviewRes.ok) throw new Error("Failed to fetch overview");
        const overviewData = await overviewRes.json();
        
        // Convert all necessary values to numbers
        setOverview({
          total_streams: Number(overviewData.overview.total_streams),
          total_followers: Number(overviewData.overview.total_followers),
          total_listeners: Number(overviewData.overview.total_listeners),
          top_tracks_count: Number(overviewData.overview.top_tracks_count),
        });

        if (!topTracksRes.ok) throw new Error("Failed to fetch top tracks");
        const topTracksData = await topTracksRes.json();
        setTopTracks(topTracksData.tracks || []);

        if (!monthlyRes.ok) throw new Error("Failed to fetch monthly data");
        const monthlyData = await monthlyRes.json();
        setMonthly(monthlyData.monthly || []);

        if (!engagementRes.ok) throw new Error("Failed to fetch engagement data");
        const engagementData = await engagementRes.json();
        setEngagement(engagementData.engagement_distribution || []);

        // Fetch daily streams and song releases
        if (!dailyStreamsRes.ok) throw new Error("Failed to fetch daily streams and releases");
        const dailyStreamsData = await dailyStreamsRes.json();
        setDailyStreams(dailyStreamsData.daily_streams || []);
        setSongReleases(dailyStreamsData.song_releases || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    };

    fetchAllData();
  }, []);

  const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—");
  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—";
    return `${n.toFixed(2)}%`;
  };
  const fmtCompact = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`;

  if (loading) {
    return (
      <div className="flex min-h-screen bg-[#123458] text-white">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <p className="text-xl">Loading dashboard data...</p>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen bg-[#123458] text-white">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <div className="bg-red-900 p-6 rounded-lg">
            <p className="text-lg font-semibold">Error loading dashboard</p>
            <p className="text-sm mt-2">{error}</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-[#123458] text-white">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">Spotify Analytics Dashboard</h1>
        </header>

        {/* KPIs Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Streams</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.total_streams)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Listeners</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtCompact(overview.total_listeners)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Followers</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.total_followers)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Top Tracks</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.top_tracks_count)}</p>
          </div>
        </section>

        {/* Top Tracks Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Tracks by Streams</h2>
          <div className="h-96">
            {topTracks.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topTracks} layout="vertical" margin={{ top: 5, right: 30, left: 200, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={fmtCompact} />
                  <YAxis dataKey="track" type="category" width={190} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmtInt(v as number)} />
                  <Bar dataKey="streams" fill="#0c4d8f" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Daily Streams with Song Releases Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Daily Streams with Song Releases</h2>
          <div className="h-96">
            {dailyStreams.length > 0 && songReleases.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={dailyStreams}
                  margin={{ top: 20, right: 30, left: 40, bottom: 60 }} // Increased margins for readabilitys
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}
                    angle={-45} // Rotate XAxis labels for better readability
                    textAnchor="end"
                  />
                  <YAxis />
                  <Tooltip formatter={(value) => fmtInt(value)} />
                  <Legend />
                  <Line type="monotone" dataKey="streams" stroke="#1DB954" strokeWidth={2} name="Daily Streams" />

                  {/* Scatter plot for Song Releases */}
                  <ScatterChart>
                    <Scatter
                      name="Song Releases"
                      data={songReleases.map((release) => ({
                        date: release.release_date,
                        streams: dailyStreams.find((stream) => stream.date === release.release_date)?.streams || 0,
                      }))}
                      fill="#FF4444"
                      line
                      shape="circle"
                      strokeWidth={2}
                    />
                  </ScatterChart>
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

      </main>
    </div>
  );
}

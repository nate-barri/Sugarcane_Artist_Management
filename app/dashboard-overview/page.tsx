"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Sidebar from "@/components/sidebar";
import { generateReport } from "@/utils/reportGenerator";

export default function Dashboard() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const [spotifyFollowers, setSpotifyFollowers] = useState<number | null>(null)
  const [facebookReach, setFacebookReach] = useState<number | null>(null);

  const [youtubeEngagements, setYoutubeEngagements] = useState<number | null>(null);
  const [metaEngagements, setMetaEngagements] = useState<number | null>(null);
  const [spotifyEngagements, setSpotifyEngagements] = useState<number | null>(null);
  const [tiktokEngagements, setTiktokEngagements] = useState<number | null>(null);
  const [topPlatformEngagement, setTopPlatformEngagement] = useState<string | null>(null);
  const [totalViews, setTotalViews] = useState<number | null>(null);
  const [totalFacebookReach, setTotalFacebookReach] = useState<number | null>(null);
  const [growthPercentage, setGrowthPercentage] = useState<number | null>(null);

  useEffect(() => {
    async function fetchAllEngagements() {
      try {
        // assign response name from fetch
        const [
          spotifyRes,
          youtubeRes,
          metaRes,
          tiktokRes,
          totalViewsRes
        ] = await Promise.all([
          fetch("/api/analytics/dashboard-overview/spotify-dash"),
          fetch("/api/analytics/dashboard-overview/youtube-dash"),
          fetch("/api/analytics/dashboard-overview/facebook-dash"),
          fetch("/api/analytics/dashboard-overview/tiktok-dash"),
          fetch("/api/analytics/dashboard-overview/crossplat-dash")
        ]);

        // Parse JSON
        const [
          spotifyData,
          youtubeData,
          metaData,
          tiktokData,
          totalViewsData
        ] = await Promise.all([
          spotifyRes.json(),
          youtubeRes.json(),
          metaRes.json(),
          tiktokRes.json(),
          totalViewsRes.json()
        ]);

        setSpotifyFollowers(spotifyData.total_followers ?? 0);
        setSpotifyEngagements(spotifyData.totalSpotifyEngagement ?? 0);
        setYoutubeEngagements(youtubeData.total_youtube_engagement ?? 0);
        setMetaEngagements(metaData.total_meta_engagement ?? 0);
        setTiktokEngagements(tiktokData.totalTiktokEngagement ?? 0);
        setTotalViews(totalViewsData?.totalViews ?? 0);
        setTotalFacebookReach(metaData.total_meta_reach ?? 0);
        setGrowthPercentage(spotifyData.avg_growth ?? 0);

        // Find platform with max engagement
        const comparePlatforms = [
          { name: "YouTube", value: youtubeData.total_youtube_engagement ?? 0 },
          { name: "Meta", value: metaData.total_meta_engagement ?? 0 },
          { name: "Spotify", value: spotifyData.totalSpotifyEngagement ?? 0 },
          { name: "TikTok", value: tiktokData.totalTiktokEngagement ?? 0 }
        ];
        const top = comparePlatforms.reduce((prev, curr) => (curr.value > prev.value ? curr : prev));
        setTopPlatformEngagement(top.name);

      } catch (error) {
        console.error("Error fetching engagements:", error);
        // Default/crash is 0
        setSpotifyFollowers(0);
        setYoutubeEngagements(0);
        setMetaEngagements(0);
        setSpotifyEngagements(0);
        setTiktokEngagements(0);
        setTotalViews(0);
      }
    }

    fetchAllEngagements();
  }, []);
  useEffect(() => {
    console.log("[v0] Authentication check starting...");
    const authToken = localStorage.getItem("authToken");
    console.log("[v0] AuthToken found:", authToken);

    if (!authToken) {
      console.log("[v0] No auth token, redirecting to login...");
      router.push("/login");
    } else {
      console.log("[v0] Auth token exists, setting authenticated to true");
      setIsAuthenticated(true);
    }
    setLoading(false);

    // (Optional) cleanup if you later switch to an auth listener:
    // return () => unsubscribe?.();
  }, [router]);

  console.log("[v0] Render - Loading:", loading, "Authenticated:", isAuthenticated);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    console.log("[v0] Not authenticated, returning null");
    return null; // route exists; not a 404
  }

  return (
    <div className="flex min-h-screen bg-[#D3D3D3] text-[#123458]">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#123458]  text-center">Dashboard Overview</h1>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center">Total Followers | Spotify</h2>
            <p className="text-3xl font-bold text-gray-900">
              {spotifyFollowers !== null ? spotifyFollowers.toLocaleString() : "Loading..."}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center">Total Reach</h2>
            <p className="text-3xl font-bold text-gray-900">
              {/* Facebook Reach */}
              {totalFacebookReach !== null ? totalFacebookReach.toLocaleString() : "Loading..."}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center">Total Views</h2>
            <p className="text-3xl font-bold text-gray-900">
              {/* Tiktok + Youtube Views */}
              {totalViews !== null ? totalViews.toLocaleString() : "Loading..."}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458] text-center" >Total Engagement | YouTube</h2>
            <p className="text-3xl font-bold text-gray-900">
              {/* Likes, Shares, Comments, Dislikes */}
              {youtubeEngagements !== null ? youtubeEngagements.toLocaleString() : "Loading..."}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center ">Total Engagement | Meta</h2>
            <p className="text-3xl font-bold text-gray-900">
              {/* Shares, Comments, Reactions */}
              {metaEngagements !== null ? metaEngagements.toLocaleString() : "Loading..."}

            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center">Total Engagement | Spotify</h2>
            <p className="text-3xl font-bold text-gray-900">
              {/* Spotify_stats, listeners + stream + followers gained */}
              {spotifyEngagements !== null ? spotifyEngagements.toLocaleString() : "Loading..."}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center">Total Engagement | TikTok</h2>
            <p className="text-3xl font-bold text-gray-900">
              {/* likes, shares, comments, saves */}
              {tiktokEngagements !== null ? tiktokEngagements.toLocaleString() : "Loading..."}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center">Top Engaging Platform</h2>
            <p className="text-3xl font-bold text-gray-900">
              {/* Compares tiktok, spotify, meta, and youtube engagements */}
              {topPlatformEngagement !== null ? topPlatformEngagement : "Loading..."}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-center items-center aspect-square w-full max-w-xs">
            <h2 className="text-lg font-medium text-[#123458]  text-center">Listeners Growth | Spotify</h2>
            <p className="text-3xl font-bold text-gray-900">{growthPercentage !== null ? growthPercentage : "Loading..."}%</p>
          </div>
        </section>

        <div className="flex justify-end">
          <button
            onClick={() => generateReport("Dashboard Overview")}
            className="bg-[#0f2946] hover:bg-[#001F3F] text-[#FFFFFF] font-bold py-3 px-6 rounded-lg shadow-lg flex items-center transition-colors duration-200"
          >
            GENERATE REPORT
          </button>
        </div>
      </main>
    </div>
  );
}

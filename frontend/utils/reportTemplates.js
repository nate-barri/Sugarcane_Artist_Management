// reportTemplates.js

export function addYouTubeReport(doc) {
  doc.setFontSize(14);
  doc.text("YouTube Metrics:", 10, 50);
  doc.text("Total Views: ____________", 15, 60);
  doc.text("Total Watch Time: ____________", 15, 70);
  doc.text("Subscribers Growth: ____________", 15, 80);

  doc.text("Charts:", 10, 100);
  doc.rect(15, 110, 80, 40);
  doc.text("Views Overtime", 20, 125);
  doc.rect(110, 110, 80, 40);
  doc.text("Audience Retention", 115, 125);
}

export function addFacebookReport(doc){
  doc.setFontSize(14);
  doc.text("Meta Facebook Metrics:", 10, 50);
  doc.text("Page Likes: ____________", 15, 60);
  doc.text("Post Reach: ____________", 15, 70);
  doc.text("Engagement Rate: ____________", 15, 80);
  doc.text("Page Views: ____________", 15, 90);

  doc.text("Charts:", 10, 100);
  doc.rect(15, 110, 80, 40);
  doc.text("Page Likes Growth", 20, 125);
  doc.rect(110, 110, 80, 40);
  doc.text("Post Engagement", 115, 125);
}

export function addSpotifyReport(doc){
  doc.setFontSize(14);
  doc.text("Spotify Metrics:", 10, 50);
  doc.text("Total Streams: ____________", 15, 60);
  doc.text("Monthly Listeners: ____________", 15, 70);
  doc.text("Followers: ____________", 15, 80);
  doc.text("Playlist Adds: ____________", 15, 90);

  doc.text("Charts:", 10, 100);
  doc.rect(15, 110, 80, 40);
  doc.text("Streams Overtime", 20, 125);
  doc.rect(110, 110, 80, 40);
  doc.text("Top Tracks Performance", 115, 125);
}

export function addInstagramReport(doc){
  doc.setFontSize(14);
  doc.text("Instagram Metrics:", 10, 50);
  doc.text("Followers: ____________", 15, 60);
  doc.text("Post Reach: ____________", 15, 70);
  doc.text("Engagement Rate: ____________", 15, 80);
  doc.text("Story Views: ____________", 15, 90);

  doc.text("Charts:", 10, 100);
  doc.rect(15, 110, 80, 40);
  doc.text("Follower Growth", 20, 125);
  doc.rect(110, 110, 80, 40);
  doc.text("Post Performance", 115, 125);
}

export function addTiktokReport(doc){
  doc.setFontSize(14);
  doc.text("TikTok Metrics:", 10, 50);
  doc.text("Followers: ____________", 15, 60);
  doc.text("Video Views: ____________", 15, 70);
  doc.text("Likes: ____________", 15, 80);
  doc.text("Shares: ____________", 15, 90);

  doc.text("Charts:", 10, 100);
  doc.rect(15, 110, 80, 40);
  doc.text("Video Performance", 20, 125);
  doc.rect(110, 110, 80, 40);
  doc.text("Follower Growth", 115, 125);
}

/*export function addDashboardReport(doc){
  doc.setFontSize(14);
  doc.text("Dashboard Metrics:", 10, 50);
  doc.text("Total Subscribers: ____________", 15, 60);
  doc.text("Total Views: ____________", 15, 70);
  doc.text("Total Watch Time: ____________", 15, 80);
  doc.text("Total Spotify Streams: ____________", 15, 90);
  doc.text("Audience Growth: ____________", 15, 90);

  doc.text("Charts:", 10, 100);
  doc.rect(15, 110, 80, 40);
  doc.text("Overall Engagement", 20, 125);
  doc.rect(110, 110, 80, 40);
  doc.text("Spotify Streams Overtime", 115, 125);
}*/
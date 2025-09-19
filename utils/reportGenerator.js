import { jsPDF } from "jspdf";
import { addYouTubeReport, addFacebookReport, addSpotifyReport, addInstagramReport, addTiktokReport, addDashboardReport, addPredictiveAnalyticsDashboardReport } from "./reportTemplates";

export function generateReport(platform) {
  const doc = new jsPDF();
  const date = new Date().toLocaleString();

  doc.setFontSize(18);
  doc.text(`${platform} Dashboard Report`, 10, 20);
  doc.setFontSize(12);
  doc.text(`Generated on: ${date}`, 10, 30);

  switch (platform.toLowerCase()) {
    case "youtube":
      addYouTubeReport(doc);
      break;
    case "facebook":
      addFacebookReport(doc);
      break;
    case "spotify":
      addSpotifyReport(doc);
      break;
    case "instagram":
      addInstagramReport(doc);
      break;
    case "tiktok":
      addTiktokReport(doc);
      break;
    case "dashboard":
      addDashboardReport(doc);
      break;
    case "predictiveanalyticsdashboard":
      addPredictiveAnalyticsDashboardReport(doc);
      break;
    default:
      doc.text("No template available for this platform.", 10, 50);
  }

  doc.save(`${platform}-report.pdf`);
}

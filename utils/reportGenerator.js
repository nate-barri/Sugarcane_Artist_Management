// utils/reportGenerator.js
import { jsPDF } from "jspdf";

export function generateReport(platform) {
  const doc = new jsPDF();
  const date = new Date().toLocaleString();

  doc.setFontSize(16);
  doc.text(`${platform} Report`, 10, 20);
  doc.setFontSize(12);
  doc.text(`Generated on: ${date}`, 10, 30);

  doc.save(`${platform}-report.pdf`);
}

// npm install jspdf

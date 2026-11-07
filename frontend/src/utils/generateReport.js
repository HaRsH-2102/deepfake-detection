import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

function normalizeImage(image) {
  if (!image) {
    return null;
  }
  return image.startsWith("data:") ? image : `data:image/png;base64,${image}`;
}

function addDarkBackground(doc) {
  doc.setFillColor(10, 10, 15);
  doc.rect(0, 0, doc.internal.pageSize.getWidth(), doc.internal.pageSize.getHeight(), "F");
  doc.setTextColor(248, 250, 252);
}

export async function generateReport({
  prediction,
  confidence,
  mode,
  timestamp,
  gradcamBase64,
  limeBase64,
  exifFlags = [],
  frameConfidences = [],
  topFakeFrames = [],
}) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const verdictColor = prediction === "FAKE" ? [239, 68, 68] : [34, 197, 94];

  addDarkBackground(doc);
  doc.setFont("courier", "bold");
  doc.setFontSize(22);
  doc.text("DEEPFAKE DETECTOR", 40, 52);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.setTextColor(148, 163, 184);
  doc.text(`Generated ${timestamp.toLocaleString()}`, 40, 74);

  doc.setTextColor(...verdictColor);
  doc.setFont("courier", "bold");
  doc.setFontSize(34);
  doc.text(prediction, 40, 128);

  doc.setTextColor(248, 250, 252);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(15);
  doc.text(`Confidence: ${(confidence * 100).toFixed(2)}%`, 40, 158);
  doc.text(`Mode: ${mode}`, 40, 182);

  doc.setDrawColor(59, 130, 246);
  doc.roundedRect(36, 210, pageWidth - 72, 110, 18, 18);
  doc.setFont("courier", "bold");
  doc.setFontSize(12);
  doc.text("METADATA SIGNALS", 52, 236);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  const metadataLines = exifFlags.length ? exifFlags : ["No suspicious metadata detected"];
  metadataLines.slice(0, 5).forEach((flag, index) => {
    doc.text(`- ${flag}`, 52, 262 + index * 18);
  });

  doc.addPage();
  addDarkBackground(doc);
  doc.setFont("courier", "bold");
  doc.setFontSize(22);
  doc.text("EXPLAINABILITY VISUALS", 40, 52);

  const gradcamImage = normalizeImage(gradcamBase64);
  const limeImage = normalizeImage(limeBase64);
  if (gradcamImage) {
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("Grad-CAM", 40, 88);
    doc.addImage(gradcamImage, "PNG", 40, 100, 240, 240);
  }
  if (limeImage) {
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text("LIME", 315, 88);
    doc.addImage(limeImage, "PNG", 315, 100, 240, 240);
  }

  if (mode === "video") {
    doc.addPage();
    addDarkBackground(doc);
    doc.setFont("courier", "bold");
    doc.setFontSize(22);
    doc.text("VIDEO FORENSICS SUMMARY", 40, 52);

    autoTable(doc, {
      startY: 80,
      head: [["Frame", "Confidence"]],
      body: frameConfidences.slice(0, 20).map((value, index) => [index + 1, value.toFixed(3)]),
      theme: "grid",
      styles: {
        fillColor: [18, 18, 26],
        textColor: [248, 250, 252],
        lineColor: [59, 130, 246],
      },
      headStyles: {
        fillColor: [30, 41, 59],
      },
      margin: { left: 40, right: 40 },
    });

    const currentY = doc.lastAutoTable?.finalY ? doc.lastAutoTable.finalY + 30 : 180;
    topFakeFrames.slice(0, 3).forEach((frame, index) => {
      const top = currentY + index * 155;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(12);
      doc.text(
        `Frame ${frame.frame_index} | ${(frame.confidence * 100).toFixed(1)}%`,
        40,
        top,
      );
      if (frame.frame_image) {
        doc.addImage(normalizeImage(frame.frame_image), "PNG", 40, top + 12, 120, 120);
      }
      if (frame.gradcam_image) {
        doc.addImage(normalizeImage(frame.gradcam_image), "PNG", 190, top + 12, 120, 120);
      }
    });
  }

  return doc.output("blob");
}

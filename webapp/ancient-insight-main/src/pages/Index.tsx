import { useState, useCallback } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import LeftPanel from "@/components/LeftPanel";
import CenterPanel from "@/components/CenterPanel";
import RightPanel from "@/components/RightPanel";
import type { OCRResponse } from "@/types/ocr";

const Index = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [activeLineIndex, setActiveLineIndex] = useState<number | null>(null);

  const handleReset = useCallback(() => {
    setIsProcessing(false);
    setOcrResult(null);
    setImageUrl(null);
    setActiveLineIndex(null);
  }, []);

  const handleImageUploaded = useCallback(async (file: File, url: string) => {
    setImageUrl(url);
    setIsProcessing(true);
    setOcrResult(null);
    setActiveLineIndex(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("API request failed");
      }

      const data: OCRResponse = await response.json();
      setOcrResult(data);
      const firstLineIndex =
        data.lines && data.lines.length > 0 ? data.lines[0].line_index : null;
      setActiveLineIndex(firstLineIndex);
    } catch (error) {
      console.error("Error during OCR:", error);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  return (
    <DashboardLayout>
      {/* Left Column: Acquisition & Settings */}
      <LeftPanel 
        onImageUploaded={handleImageUploaded}
        isProcessing={isProcessing}
        onClear={handleReset}
      />

      {/* Center Column: Visual Inspection */}
      <CenterPanel 
        imageUrl={imageUrl}
        ocrResult={ocrResult}
        isProcessing={isProcessing}
        activeLineIndex={activeLineIndex}
        onActiveLineChange={setActiveLineIndex}
        onClear={handleReset}
      />

      {/* Right Column: Deciphered Output & Logs */}
      <RightPanel 
        ocrResult={ocrResult}
        isProcessing={isProcessing}
        activeLineIndex={activeLineIndex}
      />
    </DashboardLayout>
  );
};

export default Index;

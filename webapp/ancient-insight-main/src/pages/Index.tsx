import { useState, useCallback } from "react";
import InputModule from "@/components/InputModule";
import OutputModule from "@/components/OutputModule";
import type { OCRResponse } from "@/types/ocr";

const Index = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeLineIndex, setActiveLineIndex] = useState<number | null>(null);

  const handleReset = useCallback(() => {
    setIsProcessing(false);
    setIsComplete(false);
    setOcrResult(null);
    setErrorMessage(null);
    setActiveLineIndex(null);
  }, []);

  const handleImageUploaded = useCallback(async (file: File, _url: string) => {
    setIsProcessing(true);
    setIsComplete(false);
    setOcrResult(null);
    setErrorMessage(null);
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
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "OCR failed. Check whether the backend server is running.",
      );
    } finally {
      setIsProcessing(false);
      setIsComplete(true);
    }
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden grid grid-cols-1 md:grid-cols-2">
      {/* Left: Input */}
      <div className="relative border-r border-border/40">
        <InputModule
          onImageUploaded={handleImageUploaded}
          isProcessing={isProcessing}
          isComplete={isComplete}
          ocrResult={ocrResult}
          errorMessage={errorMessage}
          activeLineIndex={activeLineIndex}
          onActiveLineChange={setActiveLineIndex}
          onClear={handleReset}
        />
      </div>

      {/* Right: Output */}
      <div className="relative">
        <OutputModule
          isProcessing={isProcessing}
          isComplete={isComplete}
          ocrResult={ocrResult}
          errorMessage={errorMessage}
          activeLineIndex={activeLineIndex}
          onActiveLineChange={setActiveLineIndex}
        />
      </div>

      {/* Top bar */}
      <div className="fixed top-0 left-0 right-0 flex items-center justify-between px-8 lg:px-12 py-4 z-10 pointer-events-none">
        <h1 className="font-display text-sm tracking-[0.3em] uppercase text-foreground/40 font-light">
          Historica
        </h1>
        <span className="text-[9px] tracking-[0.2em] uppercase text-muted-foreground/40 font-body">
          AI Translation Engine
        </span>
      </div>
    </div>
  );
};

export default Index;

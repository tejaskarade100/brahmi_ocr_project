import { useState, useCallback } from "react";
import InputModule from "@/components/InputModule";
import OutputModule from "@/components/OutputModule";

const FAKE_PROCESSING_DELAY = 3500;

const Index = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  const handleImageUploaded = useCallback((_file: File, _url: string) => {
    setIsProcessing(true);
    setIsComplete(false);

    setTimeout(() => {
      setIsProcessing(false);
      setIsComplete(true);
    }, FAKE_PROCESSING_DELAY);
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden grid grid-cols-1 md:grid-cols-2">
      {/* Left: Input */}
      <div className="relative border-r border-border/40">
        <InputModule
          onImageUploaded={handleImageUploaded}
          isProcessing={isProcessing}
        />
      </div>

      {/* Right: Output */}
      <div className="relative">
        <OutputModule isProcessing={isProcessing} isComplete={isComplete} />
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

import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Maximize2, Minimize2, Layers, RefreshCw, ZoomIn, ZoomOut } from "lucide-react";
import { cn } from "@/lib/utils";
import type { OCRResponse } from "@/types/ocr";

interface CenterPanelProps {
  imageUrl: string | null;
  ocrResult: OCRResponse | null;
  isProcessing: boolean;
  activeLineIndex: number | null;
  onActiveLineChange: (index: number | null) => void;
  onClear: () => void;
}

const CenterPanel: React.FC<CenterPanelProps> = ({
  imageUrl,
  ocrResult,
  isProcessing,
  activeLineIndex,
  onActiveLineChange,
  onClear,
}) => {
  const [zoom, setZoom] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showProcessed, setShowProcessed] = useState(true);
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);

  const processedImageUrl = ocrResult?.base64_image;

  const [renderFrame, setRenderFrame] = useState<{ left: number; top: number; width: number; height: number } | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const updateRenderFrame = useCallback(() => {
    if (!containerRef.current || !imageRef.current || !imageRef.current.naturalWidth) return;
    
    const container = containerRef.current;
    const img = imageRef.current;
    
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    const scale = Math.min(
      (containerWidth * 0.9) / img.naturalWidth,
      (containerHeight * 0.9) / img.naturalHeight
    );
    
    const width = img.naturalWidth * scale;
    const height = img.naturalHeight * scale;
    
    setRenderFrame({
      left: (containerWidth - width) / 2,
      top: (containerHeight - height) / 2,
      width,
      height
    });
  }, []);

  useEffect(() => {
    updateRenderFrame();
    window.addEventListener('resize', updateRenderFrame);
    return () => window.removeEventListener('resize', updateRenderFrame);
  }, [updateRenderFrame, imageUrl]);

  const getOverlayStyle = (bbox: any) => {
    if (!renderFrame || !imageRef.current?.naturalWidth) return { display: 'none' };
    
    const scaleX = renderFrame.width / imageRef.current.naturalWidth;
    const scaleY = renderFrame.height / imageRef.current.naturalHeight;
    
    return {
      left: renderFrame.left + bbox.x_min * scaleX,
      top: renderFrame.top + bbox.y_min * scaleY,
      width: (bbox.x_max - bbox.x_min) * scaleX,
      height: (bbox.y_max - bbox.y_min) * scaleY,
    };
  };

  const handleSliderMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = "touches" in e ? e.touches[0].clientX : e.clientX;
    const position = ((x - rect.left) / rect.width) * 100;
    setSliderPosition(Math.max(0, Math.min(100, position)));
  };

  const resetView = () => {
    setZoom(1);
    setSliderPosition(50);
  };

  return (
    <div className="flex-1 relative flex flex-col bg-slate-100/50">
      {/* Toolbar */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 px-4 py-2 bg-white/90 backdrop-blur-md rounded-full border border-slate-200 shadow-sm">
        <ToolbarButton icon={<ZoomIn size={14} />} onClick={() => setZoom(z => Math.min(z + 0.2, 3))} />
        <ToolbarButton icon={<ZoomOut size={14} />} onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))} />
        <div className="w-[1px] h-4 bg-slate-200 mx-1" />
        <ToolbarButton 
          icon={<Layers size={14} />} 
          active={showOverlays} 
          onClick={() => setShowOverlays(!showOverlays)} 
          label="Overlay"
        />
        <div className="w-[1px] h-4 bg-slate-200 mx-1" />
        <ToolbarButton 
          icon={<RefreshCw size={14} />} 
          onClick={onClear} 
          label="Reset"
        />
      </div>

      {/* Main Viewport */}
      <div 
        ref={containerRef}
        className="flex-1 relative overflow-hidden flex items-center justify-center cursor-crosshair"
        onMouseMove={(e) => e.buttons === 1 && handleSliderMove(e)}
      >
        <AnimatePresence mode="wait">
          {!imageUrl ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center space-y-4"
            >
              <div className="relative inline-block">
                <motion.div 
                  animate={{ rotate: 360 }}
                  transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                  className="w-32 h-32 rounded-full border border-dashed border-slate-300"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-2 h-2 bg-research-gold rounded-full" />
                </div>
              </div>
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400 font-bold">Waiting for Data Input</p>
            </motion.div>
          ) : (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative w-full h-full flex items-center justify-center"
              style={{ scale: zoom }}
            >
              {/* Original Image */}
              <img 
                ref={imageRef}
                src={imageUrl} 
                alt="Original" 
                className="max-w-[90%] max-h-[90%] object-contain shadow-2xl rounded-lg border border-white"
                onLoad={updateRenderFrame}
              />

              {/* Before/After Comparison */}
              {processedImageUrl && showProcessed && renderFrame && (
                <div 
                  className="absolute overflow-hidden shadow-2xl rounded-lg pointer-events-none"
                  style={{ 
                    left: renderFrame.left,
                    top: renderFrame.top,
                    width: renderFrame.width,
                    height: renderFrame.height,
                    clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` 
                  }}
                >
                  <img 
                    src={processedImageUrl} 
                    alt="Processed" 
                    className="w-full h-full object-contain grayscale brightness-125 contrast-150"
                  />
                </div>
              )}

              {/* Comparison Slider Explainer */}
              {processedImageUrl && showProcessed && (
                <>
                  <div className="absolute top-10 left-10 text-[10px] uppercase tracking-[0.2em] text-research-cyan font-bold pointer-events-none bg-white/80 px-3 py-1 rounded-full shadow-sm border border-slate-100">
                    Processed View
                  </div>
                  <div className="absolute top-10 right-10 text-[10px] uppercase tracking-[0.2em] text-slate-500 font-bold pointer-events-none bg-white/80 px-3 py-1 rounded-full shadow-sm border border-slate-100">
                    Original View
                  </div>
                </>
              )}

              {/* Scan Slider Handle (Blue Vertical Line) */}
              {processedImageUrl && showProcessed && (
                <div 
                  className="absolute top-0 bottom-0 w-[2px] bg-research-cyan/30 z-10 cursor-ew-resize"
                  style={{ left: `${sliderPosition}%` }}
                >
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-white shadow-lg border border-slate-200 flex items-center justify-center">
                      <div className="w-1 h-4 bg-research-cyan/50 rounded-full" />
                    </div>
                    <span className="whitespace-nowrap text-[8px] uppercase tracking-widest text-research-cyan font-bold bg-white px-2 py-1 rounded shadow-sm border border-slate-100">Slide to Compare</span>
                  </div>
                </div>
              )}

              {/* Animated Scan Line */}
              {isProcessing && (
                <div className="absolute inset-0 pointer-events-none">
                  <div className="scan-line" />
                </div>
              )}

              {/* OCR Line Overlays */}
              {showOverlays && ocrResult?.lines?.map((line, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className={cn(
                    "absolute border border-research-cyan/50 bg-research-cyan/10 cursor-pointer transition-all rounded-sm",
                    activeLineIndex === line.line_index && "border-research-cyan bg-research-cyan/20 ring-4 ring-research-cyan/10"
                  )}
                  style={getOverlayStyle(line.bbox)}
                  onClick={() => onActiveLineChange(line.line_index)}
                />
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Info Bar */}
      <div className="p-4 flex items-center justify-between border-t border-slate-200 bg-white">
        <div className="flex gap-4">
          <InfoMetric label="IMAGE-RESOLUTION" value="3400 x 1200 PX" />
          <InfoMetric label="COLOR-DEPTH" value="16-BIT GRAYSCALE" />
        </div>
        <div className="flex items-center gap-4">
          <ToolbarButton 
            icon={<Layers size={14} />} 
            active={showProcessed} 
            onClick={() => setShowProcessed(!showProcessed)} 
            label="Toggle Scanner"
          />
          <div className="flex items-center gap-2 pl-4 border-l border-slate-100">
            <div className={cn("w-2 h-2 rounded-full", isProcessing ? "bg-research-cyan animate-pulse" : "bg-green-500")} />
            <span className="text-[10px] uppercase tracking-[0.2em] text-slate-400 font-bold">
              {isProcessing ? "Processing..." : "Ready"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

const ToolbarButton = ({ icon, onClick, active = false, label }: { icon: React.ReactNode, onClick: () => void, active?: boolean, label?: string }) => (
  <button 
    onClick={onClick}
    className={cn(
      "flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all duration-200",
      active ? "text-research-cyan bg-research-cyan/5 border border-research-cyan/20" : "text-slate-500 hover:bg-slate-50"
    )}
  >
    {icon}
    {label && <span className="text-[10px] uppercase font-bold tracking-widest">{label}</span>}
  </button>
);

const InfoMetric = ({ label, value }: { label: string, value: string }) => (
  <div className="flex flex-col">
    <span className="text-[8px] text-slate-400 tracking-widest font-bold">{label}</span>
    <span className="text-[10px] text-slate-600 font-mono font-bold">{value}</span>
  </div>
);

export default CenterPanel;

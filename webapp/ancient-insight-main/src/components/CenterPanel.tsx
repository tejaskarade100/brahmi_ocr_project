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
  const containerRef = useRef<HTMLDivElement>(null);

  const resetView = () => {
    setZoom(1);
  };

  return (
    <div className="flex-1 relative flex flex-col bg-slate-100/50">
      {/* Toolbar */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 px-4 py-2 bg-white/90 backdrop-blur-md rounded-full border border-slate-200 shadow-sm">
        <ToolbarButton icon={<ZoomIn size={14} />} onClick={() => setZoom(z => Math.min(z + 0.2, 3))} />
        <ToolbarButton icon={<ZoomOut size={14} />} onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))} />
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
        className="flex-1 relative overflow-hidden flex items-center justify-center"
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
                src={imageUrl} 
                alt="Original" 
                className="max-w-[90%] max-h-[90%] object-contain shadow-2xl rounded-lg border border-white"
              />

              {/* Animated Scan Line */}
              {isProcessing && (
                <div className="absolute inset-0 pointer-events-none">
                  <div className="scan-line" />
                </div>
              )}
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

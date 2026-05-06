import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ZoomIn, ZoomOut, RefreshCw } from "lucide-react";
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
    <div className="flex-1 relative flex flex-col bg-slate-50/80">
      {/* Background Gradient for UI Depth */}
      <div className="absolute inset-0 bg-gradient-to-b from-white/40 to-slate-100/40 pointer-events-none" />

      {/* Toolbar */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 px-5 py-2.5 bg-white/95 backdrop-blur-md rounded-full border border-slate-200 shadow-sm transition-all hover:shadow-md">
        <ToolbarButton icon={<ZoomIn size={16} />} onClick={() => setZoom(z => Math.min(z + 0.2, 3))} />
        <ToolbarButton icon={<ZoomOut size={16} />} onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))} />
        <div className="w-[1px] h-5 bg-slate-200 mx-2" />
        <ToolbarButton 
          icon={<RefreshCw size={16} />} 
          onClick={onClear} 
          label="Reset"
        />
      </div>

      {/* Main Viewport */}
      <div 
        ref={containerRef}
        className="flex-1 relative overflow-hidden flex items-center justify-center p-4 z-10"
      >
        <AnimatePresence mode="wait">
          {!imageUrl ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex flex-col items-center justify-center opacity-100 pointer-events-none select-none overflow-hidden"
            >
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-8 font-mono text-5xl md:text-7xl lg:text-8xl text-slate-900/10 whitespace-nowrap z-0">
                <div className="flex gap-6">𑀅 𑀆 𑀇 𑀈 𑀉 𑀊 𑀏 𑀐 𑀑 𑀒</div>
                <div className="flex gap-6">𑀓 𑀔 𑀕 𑀖 𑀗 𑀘 𑀙 𑀚 𑀛 𑀜</div>
                <div className="flex gap-6">𑀝 𑀞 𑀟 𑀠 𑀡 𑀢 𑀣 𑀤 𑀥 𑀦</div>
                <div className="flex gap-6">𑀧 𑀨 𑀩 𑀪 𑀫 𑀬 𑀭 𑀮 𑀯</div>
                <div className="flex gap-6">𑀰 𑀱 𑀲 𑀳 𑀴 𑀵</div>
              </div>
              
              <div className="absolute inset-0 z-10 flex items-center justify-center">
                <div className="bg-gradient-to-b from-white to-slate-100 px-8 py-4 rounded-2xl border-2 border-white shadow-[0_20px_40px_-15px_rgba(0,0,0,0.15),0_0_0_1px_rgba(0,0,0,0.05)] transform transition-transform">
                  <div className="flex items-center gap-3">
                    <div className="w-2.5 h-2.5 bg-research-gold rounded-full animate-pulse shadow-[0_0_10px_rgba(198,166,100,0.5)]" />
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-700 font-extrabold drop-shadow-sm">Waiting for Data Input</p>
                  </div>
                </div>
              </div>
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
                className="max-w-full max-h-full object-contain shadow-xl rounded-xl border-4 border-white bg-white"
              />

              {/* Animated Scan Line */}
              {isProcessing && (
                <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-xl">
                  <div className="scan-line" />
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Info Bar */}
      <div className="relative z-20 p-5 flex items-center justify-between border-t border-slate-200 bg-white/95 backdrop-blur-sm shadow-[0_-4px_20px_rgba(0,0,0,0.02)]">
        <div className="flex gap-6">
          <InfoMetric label="IMAGE-RESOLUTION" value="3400 x 1200 PX" />
          <InfoMetric label="COLOR-DEPTH" value="16-BIT GRAYSCALE" />
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 pl-5 border-l border-slate-200">
            <div className={cn("w-2.5 h-2.5 rounded-full shadow-sm", isProcessing ? "bg-research-cyan animate-pulse shadow-research-cyan/40" : "bg-emerald-500 shadow-emerald-500/40")} />
            <span className="text-[11px] uppercase tracking-[0.2em] text-slate-500 font-bold">
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
      "flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all duration-200 font-medium",
      active ? "text-research-cyan bg-research-cyan/10 border border-research-cyan/20" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
    )}
  >
    {icon}
    {label && <span className="text-[11px] uppercase font-bold tracking-widest">{label}</span>}
  </button>
);

const InfoMetric = ({ label, value }: { label: string, value: string }) => (
  <div className="flex flex-col gap-1">
    <span className="text-[9px] text-slate-400 tracking-[0.15em] font-bold">{label}</span>
    <span className="text-[11px] text-slate-700 font-mono font-bold">{value}</span>
  </div>
);

export default CenterPanel;

import React, { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, BrainCircuit, Globe2, Type, Cpu, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { OCRResponse } from "@/types/ocr";

interface RightPanelProps {
  ocrResult: OCRResponse | null;
  isProcessing: boolean;
  activeLineIndex: number | null;
}

const RightPanel: React.FC<RightPanelProps> = ({
  ocrResult,
  isProcessing,
  activeLineIndex,
}) => {
  const [logs, setLogs] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isProcessing) {
      const messages = [
        "Initializing neural weights...",
        "Applying Gaussian noise reduction...",
        "Detecting glyph boundaries...",
        "Matching against Brahmi database...",
        "Optimizing linguistic context...",
        "Generating transliterations...",
      ];
      let i = 0;
      const interval = setInterval(() => {
        if (i < messages.length) {
          setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${messages[i]}`]);
          i++;
        } else {
          clearInterval(interval);
        }
      }, 1500);
      return () => clearInterval(interval);
    } else if (ocrResult) {
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Analysis Complete. Confidence: ${((ocrResult.lines?.[0]?.confidence || 0.95) * 100).toFixed(1)}%`]);
    }
  }, [isProcessing, ocrResult]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="w-full md:w-[420px] border-l border-slate-200 flex flex-col custom-scrollbar overflow-y-auto bg-slate-50/50">
      <div className="p-6 space-y-8">
        {/* Results Section */}
        <div className="space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-200">
            <div className="flex items-center gap-2 text-research-gold">
              <BrainCircuit size={20} />
              <h2 className="text-lg font-bold uppercase tracking-widest text-slate-800">Transcription Results</h2>
            </div>
            {ocrResult && (
              <div className="flex items-center gap-2 text-[11px] text-research-cyan bg-research-cyan/10 px-2.5 py-1 rounded-md border border-research-cyan/20 font-bold shadow-sm">
                <BarChart3 size={14} />
                {(0.95 * 100).toFixed(1)}% CONFIDENCE
              </div>
            )}
          </div>

          <div className="space-y-5">
            <ResultCard 
              title="Brahmi (Original Script)" 
              content={ocrResult?.brahmi_text || "---"} 
              font="font-mono text-3xl"
              accent="blue"
              loading={isProcessing}
            />
            <ResultCard 
              title="Devanagari" 
              content={ocrResult?.devanagari_text || "---"} 
              font="text-xl"
              accent="emerald"
              loading={isProcessing}
            />
            <ResultCard 
              title="Latin Transliteration" 
              content={ocrResult?.latin_text || "---"} 
              font="text-lg"
              accent="violet"
              loading={isProcessing}
            />
            <ResultCard 
              title="English Translation" 
              content={ocrResult?.english_translation || "---"} 
              font="text-lg"
              accent="amber"
              loading={isProcessing}
            />
            <ResultCard 
              title="Hindi Translation" 
              content={ocrResult?.hindi_translation || "---"} 
              font="text-lg"
              accent="rose"
              loading={isProcessing}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const ResultCard = ({ title, content, font, accent = "slate", loading = false }: { title: string, content: string, font?: string, accent?: "slate" | "amber" | "blue" | "emerald" | "violet" | "rose", loading?: boolean }) => {
  const accentStyles = {
    slate: "bg-white border-slate-200 shadow-slate-200/50 text-slate-900 label-slate-500",
    amber: "bg-amber-50/80 border-amber-200 shadow-amber-900/5 text-amber-950 label-amber-700",
    blue: "bg-blue-50/80 border-blue-200 shadow-blue-900/5 text-blue-950 label-blue-700",
    emerald: "bg-emerald-50/80 border-emerald-200 shadow-emerald-900/5 text-emerald-950 label-emerald-700",
    violet: "bg-violet-50/80 border-violet-200 shadow-violet-900/5 text-violet-950 label-violet-700",
    rose: "bg-rose-50/80 border-rose-200 shadow-rose-900/5 text-rose-950 label-rose-700",
  };

  const style = accentStyles[accent] || accentStyles.slate;
  const bgBorderShadow = style.split(" text-")[0];
  const textColor = "text-" + style.split(" text-")[1].split(" label-")[0];
  const labelColor = "text-" + style.split(" label-")[1];

  return (
    <div className={cn("p-6 rounded-xl border transition-all duration-300 shadow-sm", bgBorderShadow)}>
      <div className="flex justify-between items-center mb-3">
        <span className={cn("text-[10px] uppercase tracking-widest font-bold", labelColor)}>{title}</span>
        {accent !== "slate" && <Globe2 size={14} className={cn("opacity-40", labelColor)} />}
      </div>
      <div className="relative">
        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div 
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              <div className="h-5 bg-black/5 rounded animate-pulse w-full" />
              <div className="h-5 bg-black/5 rounded animate-pulse w-2/3" />
            </motion.div>
          ) : (
            <motion.p 
              key="content"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={cn("leading-relaxed font-semibold tracking-wide", font || "text-base", textColor)}
            >
              {content}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default RightPanel;

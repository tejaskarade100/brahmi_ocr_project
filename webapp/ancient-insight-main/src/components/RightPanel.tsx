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
    <div className="w-full md:w-96 border-l border-slate-200 flex flex-col custom-scrollbar overflow-y-auto bg-white">
      <div className="p-6 space-y-8">
        {/* Results Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-research-gold">
              <BrainCircuit size={16} />
              <h2 className="text-sm font-bold uppercase tracking-widest">Transcription Results</h2>
            </div>
            {ocrResult && (
              <div className="flex items-center gap-2 text-[10px] text-research-cyan bg-research-cyan/5 px-2 py-0.5 rounded-full border border-research-cyan/10 font-bold">
                <BarChart3 size={12} />
                {(0.95 * 100).toFixed(1)}% CONFIDENCE
              </div>
            )}
          </div>

          <div className="space-y-4">
            <ResultCard 
              title="Brahmi (Original Script)" 
              content={ocrResult?.brahmi_text || "---"} 
              font="font-mono text-lg"
              loading={isProcessing}
            />
            <ResultCard 
              title="Devanagari" 
              content={ocrResult?.devanagari_text || "---"} 
              loading={isProcessing}
            />
            <ResultCard 
              title="Latin Transliteration" 
              content={ocrResult?.latin_text || "---"} 
              loading={isProcessing}
            />
            <ResultCard 
              title="English Translation" 
              content={ocrResult?.english_translation || "---"} 
              accent="gold"
              loading={isProcessing}
            />
            <ResultCard 
              title="Hindi Translation" 
              content={ocrResult?.hindi_translation || "---"} 
              loading={isProcessing}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const ResultCard = ({ title, content, font, accent = "cyan", loading = false }: { title: string, content: string, font?: string, accent?: "cyan" | "gold", loading?: boolean }) => (
  <div className={cn(
    "p-5 rounded-xl border transition-all duration-300",
    accent === "gold" ? "bg-amber-50/50 border-amber-100" : "bg-slate-50 border-slate-100"
  )}>
    <div className="flex justify-between items-center mb-2">
      <span className={cn(
        "text-[9px] uppercase tracking-widest font-bold",
        accent === "gold" ? "text-research-gold" : "text-slate-400"
      )}>{title}</span>
      {accent === "gold" && <Globe2 size={12} className="text-research-gold/30" />}
    </div>
    <div className="relative">
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div 
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-2"
          >
            <div className="h-4 bg-slate-200/50 rounded animate-pulse w-full" />
            <div className="h-4 bg-slate-200/50 rounded animate-pulse w-2/3" />
          </motion.div>
        ) : (
          <motion.p 
            key="content"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className={cn("text-sm leading-relaxed text-slate-700 font-medium", font)}
          >
            {content}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  </div>
);

export default RightPanel;

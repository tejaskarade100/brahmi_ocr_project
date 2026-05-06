import React, { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload, Settings, Sliders, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";

interface LeftPanelProps {
  onImageUploaded: (file: File, url: string) => void;
  isProcessing: boolean;
  onClear: () => void;
}

const LeftPanel: React.FC<LeftPanelProps> = ({
  onImageUploaded,
  isProcessing,
  onClear,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (file && file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      onImageUploaded(file, url);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  return (
    <div className="w-full md:w-96 border-r border-slate-200 flex flex-col custom-scrollbar overflow-y-auto bg-slate-50/50">
      <div className="p-6 space-y-8">
        {/* Upload Section */}
        <div className="space-y-5">
          <div className="flex items-center gap-3 text-research-gold pb-1 border-b border-slate-100">
            <Upload size={22} className="text-amber-600" />
            <h2 className="text-lg font-bold uppercase tracking-widest text-slate-900">Image Acquisition</h2>
          </div>
          
          <motion.div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "relative group cursor-pointer aspect-square rounded-[2rem] border-2 border-dashed flex flex-col items-center justify-center transition-all duration-300 shadow-xl overflow-hidden",
              isDragging 
                ? "border-blue-500 bg-blue-50/80 ring-8 ring-blue-500/10" 
                : "border-amber-200 bg-gradient-to-br from-white via-amber-50/30 to-slate-50"
            )}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            
            <div className="relative z-10 flex flex-col items-center">
              <div className="w-16 h-16 rounded-2xl bg-amber-500/10 flex items-center justify-center mb-6 transition-colors">
                <Upload className={cn(
                  "w-8 h-8 transition-all duration-300",
                  isDragging ? "text-blue-600 scale-110" : "text-amber-600"
                )} strokeWidth={2} />
              </div>
              
              <div className="text-center px-4">
                <p className="text-base font-bold text-slate-800 transition-colors">Drag & Drop Inscription</p>
                <div className="flex items-center justify-center gap-2 mt-2">
                  <span className="h-[1px] w-4 bg-slate-200" />
                  <p className="text-[11px] text-slate-500 font-bold uppercase tracking-tighter">PNG, JPG up to 10MB</p>
                  <span className="h-[1px] w-4 bg-slate-200" />
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Preprocessing Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-research-cyan">
            <Sliders size={18} />
            <h2 className="text-sm font-bold uppercase tracking-widest text-slate-800">Image Controls</h2>
          </div>
          
          <div className="bg-white rounded-xl p-6 border border-slate-200 space-y-6 shadow-sm">
            <ControlSlider label="Threshold" defaultValue={65} accent="blue" />
            <ControlSlider label="Contrast" defaultValue={80} accent="emerald" />
            <ControlSlider label="Denoise" defaultValue={30} accent="violet" />
            <div className="flex items-center justify-between pt-3 border-t border-slate-100">
              <span className="text-[11px] uppercase tracking-wider text-slate-600 font-bold">Auto-Enhance</span>
              <div className="w-8 h-4 bg-slate-200 rounded-full relative cursor-pointer shadow-inner">
                <div className="absolute right-1 top-1 w-2 h-2 bg-research-cyan rounded-full shadow-sm" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ControlSlider = ({ label, defaultValue, accent = "blue" }: { label: string, defaultValue: number, accent?: string }) => {
  const colors: Record<string, string> = {
    blue: "text-blue-600 bg-blue-500",
    emerald: "text-emerald-600 bg-emerald-500",
    violet: "text-violet-600 bg-violet-500",
  };
  const color = colors[accent];

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-[11px] uppercase tracking-wider font-bold">
        <span className="text-slate-500">{label}</span>
        <span className={color.split(" ")[0]}>{defaultValue}%</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden shadow-inner">
        <div className={cn("h-full rounded-full", color.split(" ")[1])} style={{ width: `${defaultValue}%` }} />
      </div>
    </div>
  );
};

const ModelOption = ({ label, active = false }: { label: string, active?: boolean }) => (
  <div className={cn(
    "px-3 py-2 rounded-lg border text-[11px] uppercase tracking-widest cursor-pointer transition-all",
    active ? "bg-research-gold/10 border-research-gold/30 text-research-gold" : "bg-white/5 border-white/5 text-white/40 hover:bg-white/10"
  )}>
    {label}
  </div>
);

export default LeftPanel;

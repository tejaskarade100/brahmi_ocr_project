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
    <div className="w-full md:w-96 border-r border-slate-200 flex flex-col custom-scrollbar overflow-y-auto bg-white">
      <div className="p-6 space-y-8">
        {/* Upload Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-research-gold">
            <Upload size={16} />
            <h2 className="text-sm font-bold uppercase tracking-widest">Image Acquisition</h2>
          </div>
          
          <motion.div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            className={cn(
              "relative group cursor-pointer aspect-square rounded-2xl border-2 border-dashed flex flex-col items-center justify-center transition-all duration-300",
              isDragging ? "border-research-cyan bg-research-cyan/5" : "border-slate-200 hover:border-research-gold/50 bg-slate-50"
            )}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            
            <div className="relative">
              <Upload className={cn(
                "w-12 h-12 transition-colors duration-300",
                isDragging ? "text-research-cyan" : "text-slate-300 group-hover:text-research-gold"
              )} strokeWidth={1} />
            </div>
            
            <div className="mt-4 text-center">
              <p className="text-xs font-semibold text-slate-600">Drag & Drop Inscription</p>
              <p className="text-[10px] text-slate-400 mt-1">PNG, JPG up to 10MB</p>
            </div>
          </motion.div>
        </div>

        {/* Preprocessing Controls */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-research-cyan">
            <Sliders size={16} />
            <h2 className="text-sm font-bold uppercase tracking-widest">Image Controls</h2>
          </div>
          
          <div className="bg-slate-50 rounded-xl p-5 border border-slate-100 space-y-5">
            <ControlSlider label="Threshold" defaultValue={65} />
            <ControlSlider label="Contrast" defaultValue={80} />
            <ControlSlider label="Denoise" defaultValue={30} />
            <div className="flex items-center justify-between pt-2">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">Auto-Enhance</span>
              <div className="w-8 h-4 bg-slate-200 rounded-full relative cursor-pointer">
                <div className="absolute right-1 top-1 w-2 h-2 bg-research-cyan rounded-full" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ControlSlider = ({ label, defaultValue }: { label: string, defaultValue: number }) => (
  <div className="space-y-2">
    <div className="flex justify-between text-[10px] uppercase tracking-wider">
      <span className="text-white/60">{label}</span>
      <span className="text-research-cyan">{defaultValue}%</span>
    </div>
    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
      <div className="h-full bg-research-cyan/50" style={{ width: `${defaultValue}%` }} />
    </div>
  </div>
);

const ModelOption = ({ label, active = false }: { label: string, active?: boolean }) => (
  <div className={cn(
    "px-3 py-2 rounded-lg border text-[11px] uppercase tracking-widest cursor-pointer transition-all",
    active ? "bg-research-gold/10 border-research-gold/30 text-research-gold" : "bg-white/5 border-white/5 text-white/40 hover:bg-white/10"
  )}>
    {label}
  </div>
);

export default LeftPanel;

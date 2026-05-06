import React from "react";
import { motion } from "framer-motion";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  return (
    <div className="relative min-h-screen bg-research-bg text-foreground overflow-hidden flex flex-col">
      {/* Header */}
      <header className="relative z-10 border-b border-slate-200 bg-white/80 backdrop-blur-md px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="w-10 h-10 rounded-lg bg-research-gold/10 border border-research-gold/20 flex items-center justify-center"
          >
            <span className="text-research-gold text-xl font-bold">B</span>
          </motion.div>
          <div>
            <h1 className="text-xl font-bold tracking-tighter text-slate-900 uppercase">BRAHMI <span className="text-research-gold">OCR</span></h1>
            <p className="text-[10px] uppercase tracking-[0.3em] text-slate-400 font-medium">Neural Deciphering System v2.0</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="hidden md:flex flex-col items-end">
            <span className="text-[10px] uppercase tracking-widest text-research-cyan font-bold">System Status: Active</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col md:flex-row overflow-hidden bg-slate-50/50">
        {children}
      </main>

      {/* Footer / Status Bar */}
      <footer className="relative z-10 border-t border-slate-200 bg-white px-6 py-2 flex items-center justify-between text-[10px] uppercase tracking-widest text-slate-400 font-medium">
        <div className="flex gap-4">
          <span>AI Research Lab</span>
          <span className="text-slate-200">|</span>
          <span>Inscriptions Processed: 12,402</span>
        </div>
        <div className="flex gap-4">
          <span className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
            Backend: Online
          </span>
          <span>MUMBAI, INDIA</span>
        </div>
      </footer>
    </div>
  );
};

export default DashboardLayout;

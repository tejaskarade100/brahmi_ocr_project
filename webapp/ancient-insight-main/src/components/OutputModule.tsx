import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ScrollArea } from "@/components/ui/scroll-area";
import EtymologyCard from "./EtymologyCard";

interface EtymologyEntry {
  symbol: string;
  modern: string;
  meaning: string;
}

interface OutputModuleProps {
  isProcessing: boolean;
  isComplete: boolean;
  ocrResult?: any;
}



const ETYMOLOGY_DATA: EtymologyEntry[] = [
  { symbol: "𑀅", modern: "A", meaning: "Derived from the Semitic 'aleph' — the head of an ox, symbolizing primordial creation" },
  { symbol: "𑀰", modern: "Sha", meaning: "Represents the sound of breath; connected to 'shanti' meaning inner peace" },
  { symbol: "𑀓", modern: "Ka", meaning: "From proto-Brahmic root for action; ancestor of Devanagari 'क'" },
  { symbol: "𑀤", modern: "Da", meaning: "Symbolizes giving or gift; related to 'dana' — the act of generosity" },
  { symbol: "𑀭", modern: "Ra", meaning: "The solar consonant; appears in words for king, light, and radiance" },
  { symbol: "𑀫", modern: "Ma", meaning: "Universal mother syllable; found across all Indic scripts as creation's sound" },
];

const OutputModule = ({ isProcessing, isComplete, ocrResult }: OutputModuleProps) => {
  const [displayedText, setDisplayedText] = useState("");
  const [typingDone, setTypingDone] = useState(false);
  const [showEtymology, setShowEtymology] = useState(false);

  useEffect(() => {
    if (!isComplete || !ocrResult) {
      setDisplayedText("");
      setTypingDone(false);
      setShowEtymology(false);
      return;
    }

    const targetText = ocrResult.english_translation || "";
    let i = 0;
    setDisplayedText("");
    const interval = setInterval(() => {
      if (i < targetText.length) {
        setDisplayedText(targetText.slice(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
        setTypingDone(true);
        setTimeout(() => setShowEtymology(true), 600);
      }
    }, 18);

    return () => clearInterval(interval);
  }, [isComplete, ocrResult]);

  return (
    <div className="flex flex-col h-full p-8 lg:p-12">
      <div className="mb-6">
        <h2 className="font-display text-2xl font-light tracking-wide text-foreground/80">
          Translation
        </h2>
        <p className="text-xs tracking-widest uppercase text-muted-foreground mt-1 font-body">
          {isProcessing
            ? "Deciphering glyphs…"
            : isComplete
              ? "Brahmi → English"
              : "Awaiting source material"}
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="pr-4">
          <AnimatePresence>
            {!isComplete && !isProcessing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center h-full min-h-[200px]"
              >
                <p className="text-xs tracking-[0.2em] uppercase text-muted-foreground/40 font-body">
                  Upload an inscription to begin
                </p>
              </motion.div>
            )}

            {isProcessing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-3 py-8"
              >
                <div className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-1 h-1 rounded-full bg-primary/40"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1, delay: i * 0.2, repeat: Infinity }}
                    />
                  ))}
                </div>
                <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground/60 font-body">
                  Processing
                </span>
              </motion.div>
            )}

            {isComplete && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                {/* Translated text with typewriter */}
                <div className="font-display text-lg leading-relaxed text-foreground/75 whitespace-pre-line mb-4">
                  <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">English:</span>
                  {displayedText}
                  {!typingDone && (
                    <motion.span
                      className="inline-block w-[2px] h-5 bg-primary/50 ml-0.5 align-text-bottom"
                      animate={{ opacity: [1, 0] }}
                      transition={{ duration: 0.6, repeat: Infinity }}
                    />
                  )}
                </div>

                {typingDone && ocrResult && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                    className="space-y-4 mb-6"
                  >
                    <div className="font-display text-md leading-relaxed text-foreground/70 whitespace-pre-line">
                      <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">Hindi:</span>
                      {ocrResult.hindi_translation}
                    </div>
                    <div className="font-display text-md leading-relaxed text-foreground/70 whitespace-pre-line">
                      <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">Devanagari:</span>
                      {ocrResult.devanagari_text}
                    </div>
                    <div className="font-display text-md leading-relaxed text-foreground/70 whitespace-pre-line">
                      <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">Brahmi text:</span>
                      <span className="font-mono text-lg">{ocrResult.brahmi_text}</span>
                    </div>
                  </motion.div>
                )}

                {/* Etymology section */}
                <AnimatePresence>
                  {showEtymology && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className="mt-10"
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <div className="h-px flex-1 bg-border/60" />
                        <span className="text-[10px] tracking-[0.25em] uppercase text-muted-foreground/50 font-body">
                          Symbol Etymology
                        </span>
                        <div className="h-px flex-1 bg-border/60" />
                      </div>

                      <div className="grid grid-cols-2 lg:grid-cols-3 gap-1">
                        {ETYMOLOGY_DATA.map((entry, i) => (
                          <EtymologyCard
                            key={entry.symbol}
                            symbol={entry.symbol}
                            modern={entry.modern}
                            meaning={entry.meaning}
                            index={i}
                          />
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  );
};

export default OutputModule;

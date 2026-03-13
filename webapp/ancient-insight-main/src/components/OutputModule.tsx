import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type {
  OCRCharacterTraceEntry,
  OCRCharacterTraceGroup,
  OCRLineResult,
  OCRResponse,
  OCRTextBreakdown,
  OCRTokenTraceEntry,
  OCRTokenTraceGroup,
} from "@/types/ocr";

interface OutputModuleProps {
  isProcessing: boolean;
  isComplete: boolean;
  ocrResult: OCRResponse | null;
  errorMessage: string | null;
  activeLineIndex: number | null;
  onActiveLineChange: (lineIndex: number | null) => void;
}

const formatConfidence = (confidence: number | null) => {
  if (confidence === null || Number.isNaN(confidence)) {
    return "n/a";
  }
  return `${Math.max(0, Math.min(confidence * 100, 100)).toFixed(1)}%`;
};

const isTokenTraceGroup = (
  entry: OCRTokenTraceEntry | OCRTokenTraceGroup,
): entry is OCRTokenTraceGroup => "tokens" in entry;

const isCharacterTraceGroup = (
  entry: OCRCharacterTraceEntry | OCRCharacterTraceGroup,
): entry is OCRCharacterTraceGroup => "characters" in entry;

const resolveTokenTrace = (
  ocrResult: OCRResponse | null,
  lineIndex: number | null,
  lineTrace?: OCRTokenTraceEntry[],
) => {
  if (lineTrace && lineTrace.length > 0) {
    return lineTrace;
  }

  const trace = ocrResult?.token_trace || [];
  if (trace.length === 0) {
    return [];
  }

  const first = trace[0];
  if (isTokenTraceGroup(first as OCRTokenTraceEntry | OCRTokenTraceGroup)) {
    const groupedTrace = trace as OCRTokenTraceGroup[];
    if (lineIndex !== null) {
      return groupedTrace.find((entry) => entry.line_index === lineIndex)?.tokens || [];
    }
    return groupedTrace[0]?.tokens || [];
  }

  return trace as OCRTokenTraceEntry[];
};

const resolveCharacterTrace = (
  ocrResult: OCRResponse | null,
  lineIndex: number | null,
  lineTrace?: OCRCharacterTraceEntry[],
) => {
  if (lineTrace && lineTrace.length > 0) {
    return lineTrace;
  }

  const trace = ocrResult?.character_trace || [];
  if (trace.length === 0) {
    return [];
  }

  const first = trace[0];
  if (isCharacterTraceGroup(first as OCRCharacterTraceEntry | OCRCharacterTraceGroup)) {
    const groupedTrace = trace as OCRCharacterTraceGroup[];
    if (lineIndex !== null) {
      return groupedTrace.find((entry) => entry.line_index === lineIndex)?.characters || [];
    }
    return groupedTrace[0]?.characters || [];
  }

  return trace as OCRCharacterTraceEntry[];
};

const renderBreakdownBadges = (breakdown?: OCRTextBreakdown) => {
  if (!breakdown) {
    return null;
  }

  const stats = [
    breakdown.category_guess ? `type: ${breakdown.category_guess}` : null,
    typeof breakdown.character_count === "number"
      ? `${breakdown.character_count} chars`
      : null,
    typeof breakdown.word_count === "number" ? `${breakdown.word_count} words` : null,
    typeof breakdown.line_count === "number" ? `${breakdown.line_count} lines` : null,
  ].filter(Boolean);

  return (
    <div className="flex flex-wrap gap-2">
      {stats.map((value) => (
        <Badge key={value} variant="outline" className="border-border/70 text-[10px] uppercase tracking-[0.16em] text-foreground/70">
          {value}
        </Badge>
      ))}
    </div>
  );
};

const TracePills = ({
  tokens,
  characters,
}: {
  tokens?: OCRTokenTraceEntry[];
  characters?: OCRCharacterTraceEntry[];
}) => (
  <div className="space-y-3">
    {tokens && tokens.length > 0 && (
      <div className="space-y-2">
        <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/60">
          Token Trace
        </p>
        <div className="flex flex-wrap gap-2">
          {tokens.map((token) => (
            <div
              key={`${token.index}-${token.token_id}`}
              className={cn(
                "rounded-sm border px-2 py-1 font-mono text-[11px]",
                token.is_special
                  ? "border-border/50 bg-muted/30 text-muted-foreground"
                  : "border-border/70 bg-background/70 text-foreground/80",
              )}
              title={`Confidence: ${formatConfidence(token.confidence)}`}
            >
              <span>{token.token}</span>
              <span className="ml-2 text-[9px] uppercase tracking-[0.12em] text-muted-foreground/60">
                {formatConfidence(token.confidence)}
              </span>
            </div>
          ))}
        </div>
      </div>
    )}

    {characters && characters.length > 0 && (
      <div className="space-y-2">
        <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/60">
          Character Trace
        </p>
        <div className="flex flex-wrap gap-2">
          {characters.map((char) => (
            <div
              key={`${char.index}-${char.codepoint}`}
              className="rounded-sm border border-border/70 bg-background/70 px-2 py-1 font-mono text-[11px] text-foreground/80"
              title={`${char.codepoint} • ${char.unicode_name}`}
            >
              <span>{char.is_space ? "␠" : char.char}</span>
              <span className="ml-2 text-[9px] uppercase tracking-[0.12em] text-muted-foreground/60">
                {char.codepoint}
              </span>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
);

const OutputModule = ({
  isProcessing,
  isComplete,
  ocrResult,
  errorMessage,
  activeLineIndex,
  onActiveLineChange,
}: OutputModuleProps) => {
  const [displayedText, setDisplayedText] = useState("");
  const [typingDone, setTypingDone] = useState(false);

  useEffect(() => {
    if (!isComplete || !ocrResult || errorMessage) {
      setDisplayedText("");
      setTypingDone(false);
      return;
    }

    const targetText = ocrResult.english_translation || "";
    let i = 0;
    setDisplayedText("");
    setTypingDone(false);

    const interval = setInterval(() => {
      if (i < targetText.length) {
        setDisplayedText(targetText.slice(0, i + 1));
        i += 1;
      } else {
        clearInterval(interval);
        setTypingDone(true);
      }
    }, 18);

    return () => clearInterval(interval);
  }, [isComplete, ocrResult, errorMessage]);

  const lines = ocrResult?.lines || [];
  const selectedLine =
    lines.find((line) => line.line_index === activeLineIndex) ||
    lines[0] ||
    null;
  const breakdown =
    ocrResult?.debug_info?.text_breakdown || ocrResult?.lines?.[0]?.text_breakdown;
  const preprocessInfo = ocrResult?.debug_info?.preprocess;
  const selectedLineTokens = resolveTokenTrace(
    ocrResult,
    selectedLine?.line_index ?? null,
    selectedLine?.token_trace,
  );
  const selectedLineCharacters = resolveCharacterTrace(
    ocrResult,
    selectedLine?.line_index ?? null,
    selectedLine?.character_trace,
  );

  return (
    <div className="flex flex-col flex-1 p-8 lg:p-12 pt-16 overflow-x-hidden min-w-0">
      <div className="mb-6">
        <h2 className="font-display text-2xl font-light tracking-wide text-foreground/95">
          OCR Analysis
        </h2>
        <p className="text-xs tracking-widest uppercase text-foreground/60 mt-1 font-body">
          {isProcessing
            ? "Deciphering glyphs..."
            : isComplete
              ? "Translation, line breakdown, and token diagnostics"
              : "Awaiting source material"}
        </p>
      </div>

      <ScrollArea className="flex-1 overflow-x-hidden min-w-0">
        <div className="pr-4 min-w-0">
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
                <div className="flex gap-1 min-w-0">
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

            {isComplete && errorMessage && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="rounded-sm border border-destructive/40 bg-destructive/5 p-4"
              >
                <p className="text-xs uppercase tracking-[0.16em] text-destructive/80">
                  OCR request failed
                </p>
                <p className="mt-2 text-sm text-foreground/75">{errorMessage}</p>
              </motion.div>
            )}

            {isComplete && !errorMessage && ocrResult && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                <div className="space-y-4 rounded-sm border border-border/60 bg-muted/20 p-4">
                  <div className="font-display text-lg leading-relaxed text-foreground/75 whitespace-pre-line">
                    <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">
                      English:
                    </span>
                    {displayedText}
                    {!typingDone && (
                      <motion.span
                        className="inline-block w-[2px] h-5 bg-primary/50 ml-0.5 align-text-bottom"
                        animate={{ opacity: [1, 0] }}
                        transition={{ duration: 0.6, repeat: Infinity }}
                      />
                    )}
                  </div>

                  {renderBreakdownBadges(breakdown)}

                  <div className="space-y-3">
                    <div className="font-display text-md leading-relaxed text-foreground/70 whitespace-pre-line">
                      <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">
                        Hindi:
                      </span>
                      {ocrResult.hindi_translation || "n/a"}
                    </div>
                    <div className="font-display text-md leading-relaxed text-foreground/70 whitespace-pre-line">
                      <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">
                        Devanagari:
                      </span>
                      {ocrResult.devanagari_text || "n/a"}
                    </div>
                    <div className="font-display text-md leading-relaxed text-foreground/70 whitespace-pre-line">
                      <span className="text-xs tracking-widest uppercase text-muted-foreground mr-2">
                        Brahmi text:
                      </span>
                      <span className="font-mono text-lg">{ocrResult.brahmi_text || "n/a"}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-4">
                    <h3 className="text-xs tracking-[0.2em] uppercase text-muted-foreground/60 font-body">
                      Line Breakdown
                    </h3>
                    <span className="text-[10px] tracking-[0.16em] uppercase text-muted-foreground/45">
                      {lines.length} detected
                    </span>
                  </div>

                  {lines.length > 0 ? (
                    <div className="space-y-3">
                      {lines.map((line: OCRLineResult) => {
                        const isActive = line.line_index === selectedLine?.line_index;
                        return (
                          <button
                            key={line.line_index}
                            type="button"
                            className={cn(
                              "w-full rounded-sm border p-4 text-left transition-colors",
                              isActive
                                ? "border-primary/50 bg-primary/5"
                                : "border-border/60 bg-background/40 hover:border-primary/40",
                            )}
                            onClick={() => onActiveLineChange(line.line_index)}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/55">
                                  Line {line.line_index + 1}
                                </p>
                                <p className="mt-2 font-mono text-sm text-foreground/80 whitespace-pre-line">
                                  {line.text || "No text decoded"}
                                </p>
                              </div>
                              <div className="text-right text-[10px] uppercase tracking-[0.16em] text-muted-foreground/45">
                                <p>
                                  x:{line.bbox.x_min}-{line.bbox.x_max}
                                </p>
                                <p>
                                  y:{line.bbox.y_min}-{line.bbox.y_max}
                                </p>
                              </div>
                            </div>
                            <div className="mt-3">
                              {renderBreakdownBadges(line.text_breakdown)}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="rounded-sm border border-border/60 bg-background/40 p-4 text-sm text-muted-foreground/60">
                      No line-level OCR data returned.
                    </div>
                  )}
                </div>

                {selectedLine && (
                  <div className="space-y-3 rounded-sm border border-border/60 bg-background/40 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <h3 className="text-xs tracking-[0.2em] uppercase text-muted-foreground/60 font-body">
                        Selected Line Diagnostics
                      </h3>
                      <span className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/45">
                        Line {selectedLine.line_index + 1}
                      </span>
                    </div>

                    {selectedLine.text_breakdown && (
                      <div className="space-y-2">
                        {renderBreakdownBadges(selectedLine.text_breakdown)}
                      </div>
                    )}

                    <TracePills
                      tokens={selectedLineTokens}
                      characters={selectedLineCharacters}
                    />
                  </div>
                )}

                <details className="rounded-sm border border-border/60 bg-background/40 p-4" open>
                  <summary className="cursor-pointer text-xs tracking-[0.2em] uppercase text-muted-foreground/60 font-body">
                    Technical Analysis
                  </summary>

                  <div className="mt-4 space-y-4">
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="rounded-sm border border-border/60 bg-muted/20 p-3">
                        <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/55">
                          Preprocess pipeline
                        </p>
                        <p className="mt-2 text-sm text-foreground/75">
                          {preprocessInfo?.pipeline || "n/a"}
                        </p>
                      </div>
                      <div className="rounded-sm border border-border/60 bg-muted/20 p-3">
                        <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/55">
                          Line padding records
                        </p>
                        <p className="mt-2 text-sm text-foreground/75">
                          {preprocessInfo?.line_padding?.length || 0}
                        </p>
                      </div>
                    </div>

                    <div className="rounded-sm border border-border/60 bg-muted/20 p-3">
                      <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/55">
                        Global OCR breakdown
                      </p>
                      <div className="mt-3">
                        {renderBreakdownBadges(breakdown)}
                      </div>
                    </div>

                    {(selectedLineTokens.length > 0 || selectedLineCharacters.length > 0) && (
                      <div className="rounded-sm border border-border/60 bg-muted/20 p-3">
                        <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/55">
                          {selectedLine ? "Active line trace" : "Model trace"}
                        </p>
                        <div className="mt-3">
                          <TracePills
                            tokens={selectedLineTokens}
                            characters={selectedLineCharacters}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </details>
              </motion.div>
            )}

            {isComplete && !errorMessage && !ocrResult && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="rounded-sm border border-border/60 bg-background/40 p-4 text-sm text-muted-foreground/60"
              >
                The request completed, but no OCR payload was returned by the backend.
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  );
};

export default OutputModule;

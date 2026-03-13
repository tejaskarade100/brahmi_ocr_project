import { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import type { OCRBoundingBox, OCRResponse } from "@/types/ocr";

interface InputModuleProps {
  onImageUploaded: (file: File, url: string) => void;
  isProcessing: boolean;
  isComplete: boolean;
  ocrResult: OCRResponse | null;
  errorMessage: string | null;
  activeLineIndex: number | null;
  onActiveLineChange: (lineIndex: number | null) => void;
  onClear: () => void;
}

type ViewMode = "original" | "processed";

interface RenderFrame {
  left: number;
  top: number;
  width: number;
  height: number;
}

const InputModule = ({
  onImageUploaded,
  isProcessing,
  isComplete,
  ocrResult,
  errorMessage,
  activeLineIndex,
  onActiveLineChange,
  onClear,
}: InputModuleProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("original");
  const [showOverlays, setShowOverlays] = useState(true);
  const [renderFrame, setRenderFrame] = useState<RenderFrame | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const processedImageUrl = ocrResult?.base64_image || "";
  const displayImageUrl =
    viewMode === "processed" && processedImageUrl ? processedImageUrl : imageUrl;
  const lineResults = ocrResult?.lines || [];
  const canRenderLineOverlays = Boolean(processedImageUrl) && viewMode === "processed";

  const updateRenderFrame = useCallback(() => {
    const container = stageRef.current;
    const img = imageRef.current;

    if (!container || !img || !img.naturalWidth || !img.naturalHeight) {
      setRenderFrame(null);
      return;
    }

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const scale = Math.min(
      containerWidth / img.naturalWidth,
      containerHeight / img.naturalHeight,
    );

    const width = img.naturalWidth * scale;
    const height = img.naturalHeight * scale;

    setRenderFrame({
      left: (containerWidth - width) / 2,
      top: (containerHeight - height) / 2,
      width,
      height,
    });
  }, []);

  useEffect(() => {
    if (ocrResult?.base64_image) {
      setViewMode("processed");
    } else {
      setViewMode("original");
    }
  }, [ocrResult?.base64_image]);

  useEffect(() => {
    const container = stageRef.current;
    if (!container || typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver(() => updateRenderFrame());
    observer.observe(container);
    return () => observer.disconnect();
  }, [updateRenderFrame]);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  const handleFile = useCallback(
    (file: File) => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }

      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setRenderFrame(null);
      onImageUploaded(file, url);
    },
    [imageUrl, onImageUploaded],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        handleFile(file);
      }
    },
    [handleFile],
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleClick = () => {
    if (!imageUrl) {
      inputRef.current?.click();
    }
  };

  const handleClear = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    setImageUrl(null);
    setViewMode("original");
    setRenderFrame(null);
    onActiveLineChange(null);
    onClear();
  };

  const getOverlayStyle = (bbox: OCRBoundingBox) => {
    if (!renderFrame || !imageRef.current?.naturalWidth || !imageRef.current?.naturalHeight) {
      return {};
    }

    const scaleX = renderFrame.width / imageRef.current.naturalWidth;
    const scaleY = renderFrame.height / imageRef.current.naturalHeight;

    return {
      left: renderFrame.left + bbox.x_min * scaleX,
      top: renderFrame.top + bbox.y_min * scaleY,
      width: Math.max((bbox.x_max - bbox.x_min) * scaleX, 2),
      height: Math.max((bbox.y_max - bbox.y_min) * scaleY, 2),
    };
  };

  const lineCount = lineResults.length;
  const detectedCategory =
    ocrResult?.debug_info?.text_breakdown?.category_guess ||
    ocrResult?.lines?.[0]?.text_breakdown?.category_guess ||
    "unknown";

  return (
    <div className="flex flex-col flex-1 p-8 lg:p-12 pt-16">
      <div className="mb-6">
        <h2 className="font-display text-2xl font-light tracking-wide text-foreground/95">
          Source Inspection
        </h2>
        <p className="font-body text-xs tracking-widest uppercase text-foreground/60 mt-1">
          Upload, inspect preprocessing, and verify detected line regions
        </p>
      </div>

      <div
        className={cn(
          "relative flex-1 flex flex-col rounded-sm transition-colors duration-300",
          isDragging ? "bg-primary/5" : imageUrl ? "bg-transparent" : "bg-muted/30",
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setIsDragging(false)}
        onClick={handleClick}
        style={{ cursor: imageUrl ? 'default' : 'pointer', minHeight: '400px' }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              handleFile(file);
            }
          }}
        />

        <AnimatePresence mode="wait">
          {!imageUrl ? (
            <motion.div
              key="dropzone"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex flex-col items-center justify-center gap-4"
            >
              <motion.div
                animate={{ y: isDragging ? -4 : 0 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <Upload className="w-6 h-6 text-muted-foreground/50" strokeWidth={1} />
              </motion.div>
              <span className="text-xs tracking-widest uppercase text-muted-foreground/50 font-body">
                {isDragging ? "Release to upload" : "Drop image here"}
              </span>
            </motion.div>
          ) : (
            <motion.div
              key="image"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col h-full w-full"
            >
              {/* Top pill toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 p-3 flex-shrink-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-border/70 bg-background/75 px-3 py-1 text-[10px] uppercase tracking-[0.18em] text-foreground/70 backdrop-blur">
                    {detectedCategory}
                  </span>
                  <span className="rounded-full border border-border/70 bg-background/75 px-3 py-1 text-[10px] uppercase tracking-[0.18em] text-foreground/70 backdrop-blur">
                    {lineCount} {lineCount === 1 ? "line" : "lines"}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className={cn(
                      "rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.18em] backdrop-blur transition-colors",
                      viewMode === "original"
                        ? "border-primary/50 bg-primary/10 text-primary"
                        : "border-border/70 bg-background/75 text-foreground/65",
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      setViewMode("original");
                    }}
                  >
                    Original
                  </button>
                  <button
                    type="button"
                    disabled={!processedImageUrl}
                    className={cn(
                      "rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.18em] backdrop-blur transition-colors",
                      viewMode === "processed"
                        ? "border-primary/50 bg-primary/10 text-primary"
                        : "border-border/70 bg-background/75 text-foreground/65",
                      !processedImageUrl && "cursor-not-allowed opacity-50",
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (processedImageUrl) {
                        setViewMode("processed");
                      }
                    }}
                  >
                    Processed
                  </button>
                  <button
                    type="button"
                    className={cn(
                      "rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.18em] backdrop-blur transition-colors",
                      showOverlays
                        ? "border-primary/50 bg-primary/10 text-primary"
                        : "border-border/70 bg-background/75 text-foreground/65",
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowOverlays((prev) => !prev);
                    }}
                  >
                    {showOverlays ? "Hide lines" : "Show lines"}
                  </button>
                </div>
              </div>

              {/* Image stage — takes all remaining height */}
              <div ref={stageRef} className="relative flex-1 overflow-hidden rounded-sm">
                {displayImageUrl && (
                  <img
                    ref={imageRef}
                    src={displayImageUrl}
                    alt="Inscription under inspection"
                    className="absolute inset-0 w-full h-full object-contain"
                    onLoad={updateRenderFrame}
                  />
                )}

                {showOverlays &&
                  canRenderLineOverlays &&
                  renderFrame &&
                  lineResults.map((line) => {
                    const style = getOverlayStyle(line.bbox);
                    const isActive = activeLineIndex === line.line_index;
                    return (
                      <button
                        key={line.line_index}
                        type="button"
                        className={cn(
                          "absolute rounded-sm border-2 transition-all",
                          isActive
                            ? "border-primary bg-primary/10 shadow-[0_0_0_1px_rgba(0,0,0,0.08)]"
                            : "border-foreground/45 bg-background/5 hover:border-primary/70 hover:bg-primary/10",
                        )}
                        style={style}
                        onClick={(e) => {
                          e.stopPropagation();
                          onActiveLineChange(line.line_index);
                        }}
                        title={`Line ${line.line_index + 1}`}
                      >
                        <span
                          className={cn(
                            "absolute -top-6 left-0 rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.18em]",
                            isActive
                              ? "bg-primary text-primary-foreground"
                              : "bg-background/85 text-foreground/75",
                          )}
                        >
                          L{line.line_index + 1}
                        </span>
                      </button>
                    );
                  })}

                <AnimatePresence>
                  {isProcessing && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0"
                    >
                      <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-primary/40 to-transparent scan-line-animation" />
                      <motion.div
                        className="absolute inset-0 bg-primary/[0.03]"
                        animate={{ opacity: [0.03, 0.08, 0.03] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                      <div className="absolute bottom-4 left-4 right-4 flex items-center gap-3">
                        <motion.div
                          className="w-1.5 h-1.5 rounded-full bg-primary/60"
                          animate={{ opacity: [1, 0.3, 1] }}
                          transition={{ duration: 1.2, repeat: Infinity }}
                        />
                        <span className="text-[10px] tracking-[0.2em] uppercase text-primary/60 font-body">
                          Analyzing inscription...
                        </span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {isComplete && errorMessage && (
                  <div className="absolute inset-x-4 bottom-4 rounded-sm border border-destructive/40 bg-background/90 px-4 py-3 text-xs text-destructive backdrop-blur">
                    {errorMessage}
                  </div>
                )}

                {/* Re-upload button — bottom-left of image stage */}
                <button
                  type="button"
                  className="absolute bottom-4 left-4 z-20 flex items-center gap-2 rounded-full border border-border/60 bg-background/80 px-3 py-1.5 text-[10px] uppercase tracking-[0.15em] text-foreground/70 backdrop-blur transition-colors hover:border-primary/50 hover:text-primary/80 hover:bg-background/90"
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current?.click();
                  }}
                >
                  <Upload className="w-3 h-3" strokeWidth={1.5} />
                  Re-upload
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {imageUrl && !isProcessing && (
        <div className="mt-4 flex items-center justify-between gap-4">
          <button
            type="button"
            className="text-[10px] tracking-[0.15em] uppercase text-muted-foreground/60 hover:text-primary/60 transition-colors font-body"
            onClick={handleClear}
          >
            Clear &amp; upload new
          </button>

          {imageUrl && (
            <div className="text-right">
              <span className="block text-[10px] tracking-[0.15em] uppercase text-muted-foreground/45 font-body">
                {viewMode === "processed" && processedImageUrl
                  ? "Showing model-ready preprocessed image"
                  : "Showing original upload"}
              </span>
              {viewMode === "original" && processedImageUrl && lineCount > 0 && (
                <span className="mt-1 block text-[10px] tracking-[0.12em] uppercase text-muted-foreground/40 font-body">
                  Line overlays align to the processed image view
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InputModule;

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload } from "lucide-react";

interface InputModuleProps {
  onImageUploaded: (file: File, url: string) => void;
  isProcessing: boolean;
}

const InputModule = ({ onImageUploaded, isProcessing }: InputModuleProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      onImageUploaded(file, url);
    },
    [onImageUploaded]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleClick = () => {
    if (!imageUrl) inputRef.current?.click();
  };

  return (
    <div className="flex flex-col h-full p-8 lg:p-12">
      <div className="mb-6">
        <h2 className="font-display text-2xl font-light tracking-wide text-foreground/80">
          Source Inscription
        </h2>
        <p className="font-body text-xs tracking-widest uppercase text-muted-foreground mt-1">
          Upload or drop an image of ancient text
        </p>
      </div>

      <div
        className={`relative flex-1 rounded-sm cursor-pointer transition-colors duration-300 ${
          isDragging ? "bg-primary/5" : imageUrl ? "bg-transparent" : "bg-muted/30"
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setIsDragging(false)}
        onClick={handleClick}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
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
              className="relative w-full h-full overflow-hidden rounded-sm"
            >
              <img
                src={imageUrl}
                alt="Uploaded inscription"
                className="w-full h-full object-contain"
              />

              {/* Restoration scanning overlay */}
              <AnimatePresence>
                {isProcessing && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0"
                  >
                    {/* Scan line */}
                    <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-primary/40 to-transparent scan-line-animation" />
                    {/* Soft overlay */}
                    <motion.div
                      className="absolute inset-0 bg-primary/[0.03]"
                      animate={{ opacity: [0.03, 0.08, 0.03] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                    {/* Status text */}
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
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {imageUrl && !isProcessing && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 text-[10px] tracking-[0.15em] uppercase text-muted-foreground/50 hover:text-primary/60 transition-colors font-body self-start"
          onClick={(e) => {
            e.stopPropagation();
            setImageUrl(null);
          }}
        >
          Clear & upload new
        </motion.button>
      )}
    </div>
  );
};

export default InputModule;

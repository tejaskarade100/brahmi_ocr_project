import { motion } from "framer-motion";

interface EtymologyCardProps {
  symbol: string;
  modern: string;
  meaning: string;
  index: number;
}

const EtymologyCard = ({ symbol, modern, meaning, index }: EtymologyCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.12, duration: 0.5, ease: "easeOut" }}
      className="group relative p-4 rounded-sm cursor-default transition-all duration-300 hover:bg-muted/40"
    >
      {/* Symbol */}
      <div className="text-3xl font-display text-primary/70 mb-2 transition-transform duration-300 group-hover:scale-110 origin-bottom-left">
        {symbol}
      </div>

      {/* Modern translation */}
      <div className="text-xs tracking-widest uppercase text-foreground/60 font-body mb-1">
        {modern}
      </div>

      {/* Historical meaning — revealed on hover */}
      <motion.div
        className="overflow-hidden"
        initial={false}
      >
        <p className="text-[11px] leading-relaxed text-muted-foreground/0 group-hover:text-muted-foreground transition-colors duration-500 font-body">
          {meaning}
        </p>
      </motion.div>

      {/* Subtle bottom accent */}
      <div className="absolute bottom-0 left-4 right-4 h-px bg-border/50 group-hover:bg-primary/20 transition-colors duration-300" />
    </motion.div>
  );
};

export default EtymologyCard;

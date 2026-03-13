export interface OCRBoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface OCRTextBreakdown {
  category_guess?: string;
  character_count?: number;
  word_count?: number;
  line_count?: number;
  line_break_count?: number;
  space_count?: number;
  word_char_counts?: number[];
}

export interface OCRTokenTraceEntry {
  index: number;
  token_id: number;
  token: string;
  is_special: boolean;
  confidence: number | null;
}

export interface OCRCharacterTraceEntry {
  index: number;
  char: string;
  codepoint: string;
  unicode_name: string;
  is_space: boolean;
}

export interface OCRLinePadding {
  line_index: number;
  original_width: number;
  original_height: number;
  target_width: number;
  target_height: number;
  resized_width: number;
  resized_height: number;
  x_offset: number;
  y_offset: number;
  scale: number;
  aspect_ratio_preserved: boolean;
}

export interface OCRPreprocessInfo {
  pipeline?: string;
  debug?: boolean;
  target_size?: { width: number; height: number } | null;
  line_padding?: OCRLinePadding[];
  steps?: Array<Record<string, unknown>>;
  auto_mode_resolved?: string;
}

export interface OCRLineResult {
  line_index: number;
  bbox: OCRBoundingBox;
  text: string;
  text_breakdown: OCRTextBreakdown;
  token_trace?: OCRTokenTraceEntry[];
  character_trace?: OCRCharacterTraceEntry[];
}

export interface OCRTokenTraceGroup {
  line_index: number;
  tokens: OCRTokenTraceEntry[];
}

export interface OCRCharacterTraceGroup {
  line_index: number;
  characters: OCRCharacterTraceEntry[];
}

export interface OCRDebugInfo {
  text_breakdown?: OCRTextBreakdown;
  preprocess?: OCRPreprocessInfo;
  [key: string]: unknown;
}

export interface OCRResponse {
  brahmi_text: string;
  devanagari_text: string;
  hindi_translation: string;
  english_translation: string;
  debug_info: OCRDebugInfo;
  lines: OCRLineResult[];
  token_trace: OCRTokenTraceEntry[] | OCRTokenTraceGroup[];
  character_trace: OCRCharacterTraceEntry[] | OCRCharacterTraceGroup[];
  base64_image: string;
}

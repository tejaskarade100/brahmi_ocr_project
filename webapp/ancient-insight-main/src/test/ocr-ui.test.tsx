import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Index from "@/pages/Index";
import OutputModule from "@/components/OutputModule";
import type { OCRResponse } from "@/types/ocr";

const mockOcrResult: OCRResponse = {
  brahmi_text: "𑀅𑀓\n𑀕𑀚",
  devanagari_text: "अक\nगच",
  hindi_translation: "प्राचीन शिलालेख",
  english_translation: "Ancient inscription translated",
  debug_info: {
    text_breakdown: {
      category_guess: "multiline",
      character_count: 4,
      word_count: 2,
      line_count: 2,
      line_break_count: 1,
    },
    preprocess: {
      pipeline: "grayscale -> denoise -> threshold -> normalize",
      line_padding: [],
    },
  },
  lines: [
    {
      line_index: 0,
      bbox: { x_min: 8, y_min: 12, x_max: 156, y_max: 66 },
      text: "𑀅𑀓",
      text_breakdown: {
        category_guess: "word",
        character_count: 2,
        word_count: 1,
        line_count: 1,
      },
      token_trace: [
        {
          index: 0,
          token_id: 1,
          token: "𑀅",
          is_special: false,
          confidence: 0.97,
        },
      ],
      character_trace: [
        {
          index: 0,
          char: "𑀅",
          codepoint: "U+11005",
          unicode_name: "BRAHMI LETTER A",
          is_space: false,
        },
      ],
    },
    {
      line_index: 1,
      bbox: { x_min: 10, y_min: 80, x_max: 160, y_max: 136 },
      text: "𑀕𑀚",
      text_breakdown: {
        category_guess: "word",
        character_count: 2,
        word_count: 1,
        line_count: 1,
      },
      token_trace: [
        {
          index: 0,
          token_id: 2,
          token: "𑀕",
          is_special: false,
          confidence: 0.94,
        },
      ],
      character_trace: [
        {
          index: 0,
          char: "𑀕",
          codepoint: "U+11015",
          unicode_name: "BRAHMI LETTER GA",
          is_space: false,
        },
      ],
    },
  ],
  token_trace: [
    {
      line_index: 0,
      tokens: [
        {
          index: 0,
          token_id: 1,
          token: "𑀅",
          is_special: false,
          confidence: 0.97,
        },
      ],
    },
    {
      line_index: 1,
      tokens: [
        {
          index: 0,
          token_id: 2,
          token: "𑀕",
          is_special: false,
          confidence: 0.94,
        },
      ],
    },
  ],
  character_trace: [
    {
      line_index: 0,
      characters: [
        {
          index: 0,
          char: "𑀅",
          codepoint: "U+11005",
          unicode_name: "BRAHMI LETTER A",
          is_space: false,
        },
      ],
    },
    {
      line_index: 1,
      characters: [
        {
          index: 0,
          char: "𑀕",
          codepoint: "U+11015",
          unicode_name: "BRAHMI LETTER GA",
          is_space: false,
        },
      ],
    },
  ],
  base64_image: "data:image/jpeg;base64,abc123",
};

describe("OCR frontend flow", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders translated output with line diagnostics", () => {
    vi.useFakeTimers();

    render(
      <OutputModule
        isProcessing={false}
        isComplete
        ocrResult={mockOcrResult}
        errorMessage={null}
        activeLineIndex={0}
        onActiveLineChange={() => {}}
      />,
    );

    act(() => {
      vi.advanceTimersByTime(1200);
    });

    expect(screen.getByText(/Ancient inscription translated/i)).toBeInTheDocument();
    expect(screen.getByText(/multiline/i)).toBeInTheDocument();
    expect(screen.getByText(/Line Breakdown/i)).toBeInTheDocument();

    const secondLineButton = screen.getByText("𑀕𑀚").closest("button");
    expect(secondLineButton).not.toBeNull();
    fireEvent.click(secondLineButton!);

    expect(screen.getAllByText(/Line 2/i).length).toBeGreaterThan(0);
    expect(screen.getByText("U+11015")).toBeInTheDocument();
  });

  it("switches from processed image to original image note after upload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockOcrResult,
    }));

    const { container } = render(<Index />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput, {
      target: {
        files: [new File(["image"], "inscription.png", { type: "image/png" })],
      },
    });

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    await screen.findByText(/Showing model-ready preprocessed image/i);
    fireEvent.click(screen.getByRole("button", { name: /Original/i }));

    expect(
      screen.getByText(/Line overlays align to the processed image view/i),
    ).toBeInTheDocument();
  });

  it("shows the backend error state when upload fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("backend offline")));

    const { container } = render(<Index />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(fileInput, {
      target: {
        files: [new File(["image"], "inscription.png", { type: "image/png" })],
      },
    });

    await screen.findByText(/backend offline/i);
    expect(screen.getByText(/OCR request failed/i)).toBeInTheDocument();
  });
});

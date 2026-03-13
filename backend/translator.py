from deep_translator import GoogleTranslator

class Translator:
    def __init__(self):
        self.hi_translator = GoogleTranslator(source='hi', target='hi') 
        self.en_translator = GoogleTranslator(source='hi', target='en')

    def translate(self, text: str):
        """
        Translates Devanagari text (often assumed Hindi/Sanskrit)
        to conversational Hindi and English.
        """
        if not text or not text.strip():
            return {"hindi": "", "english": ""}
            
        try:
            # For Hindi, we just translate Hindi -> Hindi to get a more modern/corrected version if it's archaic
            hindi_text = self.hi_translator.translate(text)
            
            # For English, we translate from Hindi -> English
            english_text = self.en_translator.translate(text)
            
            return {
                "hindi": hindi_text,
                "english": english_text
            }
        except Exception as e:
            print(f"Translation error: {e}")
            return {
                "hindi": text, # Fallback to original
                "english": "Translation failed"
            }

# Simple test
if __name__ == "__main__":
    t = Translator()
    test_text = "मेरा नाम ब्राह्मी है" # "My name is Brahmi"
    print(f"Original: {test_text}")
    print(f"Translations: {t.translate(test_text)}")

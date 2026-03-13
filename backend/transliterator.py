import json
import os
import re

class Transliterator:
    def __init__(self, brahmi_json_path: str):
        self.mapping, self.latin_mapping = self._load_mapping(brahmi_json_path)

    def _load_mapping(self, path: str) -> dict:
        mapping = {}
        latin_mapping = {}
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Transliteration will be empty.")
            return mapping, latin_mapping
            
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # The JSON has some non-json headers like "Extras", "Vowels" etc.
        # We need to extract just the valid JSON parts. We can do this by finding all {"unicode": ...} lines
        
        matches = re.finditer(r'"([^"]+)":\s*({[^}]+})', content)
        for match in matches:
            char = match.group(1)
            try:
                data = json.loads(match.group(2).replace('null', 'null')) 
                if 'devanagari' in data and data['devanagari'] is not None:
                    mapping[char] = data['devanagari']
                if 'latin' in data and data['latin'] is not None:
                    latin_mapping[char] = data['latin']
            except json.JSONDecodeError:
                pass
                
        return mapping, latin_mapping

    def transliterate(self, brahmi_text: str) -> str:
        if not brahmi_text:
            return ""
            
        result = ""
        for char in brahmi_text:
            if char in self.mapping:
                result += self.mapping[char]
            else:
                result += char
                
        return result

    def transliterate_latin(self, brahmi_text: str) -> str:
        if not brahmi_text:
            return ""
            
        result = ""
        for char in brahmi_text:
            if char in self.latin_mapping:
                result += self.latin_mapping[char]
            else:
                result += char
                
        return result

# Simple test
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(os.path.dirname(script_dir), "brahmi.json")
    t = Transliterator(json_path)
    
    # Test a few chars
    test_chars = "𑀅𑀆𑀇𑀓𑀔"
    print(f"Brahmi: {test_chars}")
    print(f"Devanagari: {t.transliterate(test_chars)}")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import os
import sys
import shutil
import uuid
from pydantic import BaseModel, Field

# Add parent directory to path to import inference module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from transliterator import Transliterator
from translator import Translator
from inference.predict import load_trained_model, predict

app = FastAPI(title="Brahmi OCR API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
class ModelConfig:
    processor = None
    model = None
    device = None

# Initialize classes
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
brahmi_json_path = os.path.join(project_dir, "brahmi.json")
model_dir = os.path.join(project_dir, "model", "brahmi_trocr")

transliterator_app = Transliterator(brahmi_json_path)
translator_app = Translator()

@app.on_event("startup")
async def startup_event():
    print(f"Loading model from {model_dir}...")
    try:
        ModelConfig.processor, ModelConfig.model, ModelConfig.device = load_trained_model(model_dir)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Warning: Model failed to load. OCR endpoints will return dummy data.")

class OCRResponse(BaseModel):
    brahmi_text: str
    devanagari_text: str
    hindi_translation: str
    english_translation: str
    debug_info: Dict[str, Any] = Field(default_factory=dict)
    lines: List[Dict[str, Any]] = Field(default_factory=list)
    token_trace: List[Dict[str, Any]] = Field(default_factory=list)
    character_trace: List[Dict[str, Any]] = Field(default_factory=list)
    base64_image: str = ""
    
@app.get("/")
def read_root():
    return {"message": "Brahmi OCR API is running"}

@app.post("/api/upload", response_model=OCRResponse)
async def upload_image(file: UploadFile = File(...)):
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(script_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded file securely
    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        file_ext = ".jpg"
        
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run inference if model is loaded
        if ModelConfig.processor and ModelConfig.model:
            print(f"Running OCR on {temp_path}...")
            result = predict(
                image_path=temp_path,
                processor=ModelConfig.processor,
                model=ModelConfig.model,
                device=ModelConfig.device,
                preprocess=True, # Enable preprocessing so frontend gets cleaned image
                image_size=384,
                debug=True, # Enable trace for character mapping
                multiline=True, # Enable new line segmentation
                return_base64=True # Enable sending image back to UI
            )
            brahmi_text = result.get('predicted_text', '')
            debug_info = {
                "text_breakdown": result.get("text_breakdown", {}),
                "preprocess": result.get("preprocess", {}),
            }
            lines_data = result.get('lines', [])
            token_trace = result.get("token_trace", [])
            character_trace = result.get("character_trace", [])
            base64_data = result.get('base64_image', '')
        else:
            # Fallback for testing when model isn't available
            brahmi_text = "𑀅𑀆𑀇𑀓𑀔"
            debug_info = {}
            lines_data = []
            token_trace = []
            character_trace = []
            base64_data = ""
            print("WARNING: Using dummy text because model is not loaded")
            
        print(f"OCR Result: {brahmi_text}")
        
        # Run transliteration and translation
        devanagari_text = transliterator_app.transliterate(brahmi_text)
        print(f"Transliterated: {devanagari_text}")
        
        translations = translator_app.translate(devanagari_text)
        print(f"Translations: {translations}")
        
        return OCRResponse(
            brahmi_text=brahmi_text,
            devanagari_text=devanagari_text,
            hindi_translation=translations.get("hindi", ""),
            english_translation=translations.get("english", ""),
            debug_info=debug_info,
            lines=lines_data,
            token_trace=token_trace,
            character_trace=character_trace,
            base64_image=base64_data
        )
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import time
import threading
import webbrowser
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = FastAPI(title="VisionSort: 10-Class Contamination Control System")

# Configure CORS cross-origin allowances
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize deep learning zero-shot classification engine
try:
    print("Loading Semantic Vision Model (CLIP)...")
    MODEL_ID = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(MODEL_ID)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("VisionSort engine successfully online.")
except Exception as e:
    print(f"Error loading vision models: {e}")
    clip_model = None
    clip_processor = None

# DEFINITIVE 10-CLASS WASTE SEGREGATION INDEX
WASTE_CLASSES = {
    0: {"category": "Plastic", "display_name": "Polymer Recyclables / PET / HDPE", "prompt": "a photo of plastic waste, plastic bottle, or polymer recycling"},
    1: {"category": "Metal", "display_name": "Ferrous & Non-Ferrous Scrap Metal", "prompt": "a photo of scrap metal, aluminum soda cans, or metallic waste"},
    2: {"category": "Glass", "display_name": "Glass Cullet / Intact Containers", "prompt": "a photo of glass waste, broken glass, or glass bottles"},
    3: {"category": "Paper", "display_name": "High-Grade Paper / Office Waste", "prompt": "a photo of waste paper, shredded documents, or office paper waste"},
    4: {"category": "Cardboard", "display_name": "Corrugated Fibres / Box Packaging", "prompt": "a photo of cardboard boxes, corrugated fiberboard, or packaging waste"},
    5: {"category": "Organic", "display_name": "Biological Food & Garden Waste", "prompt": "a photo of organic waste, food scraps, rotting fruit, or yard waste"},
    6: {"category": "Medical", "display_name": "Clinical Biohazard / Syringes", "prompt": "a photo of medical waste, syringes, biohazard clinical materials, or face masks"},
    7: {"category": "E-Waste", "display_name": "Electronic Components & Circuitry", "prompt": "a photo of e-waste, electronic scrap, circuit boards, or broken devices"},
    8: {"category": "Textile", "display_name": "Fabric / Post-Consumer Garments", "prompt": "a photo of textile waste, old clothing, scrap fabric, or rags"},
    9: {"category": "Mixed Waste", "display_name": "Residual Contaminated Refuse", "prompt": "a photo of unsorted trash, landfill garbage, or mixed contaminated refuse"}
}

@app.post("/api/analyze")
async def analyze_waste(file: UploadFile = File(...)):
    if not clip_model or not clip_processor:
        raise HTTPException(status_code=500, detail="Neural Inference Pipeline Layer Offline.")

    try:
        # Read incoming stream payload
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Compile our candidate text prompts from our target matrix array
        labels = [WASTE_CLASSES[i]["prompt"] for i in sorted(WASTE_CLASSES.keys())]
        
        # Process imagery data and target phrases simultaneously 
        inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            
        # Extract classification probabilities using Softmax normalization
        logits_per_image = outputs.logits_per_image 
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        # Select highest matching matrix target
        best_class_idx = int(probs.argmax())
        confidence_value = float(probs[best_class_idx])
        
        # Pull definitions from metadata architecture
        target_class = WASTE_CLASSES[best_class_idx]
        detected_object = target_class["display_name"]
        category_type = target_class["category"]
        conf_percentage = f"{confidence_value * 100:.1f}%"
        
        # Fault configuration for poor lighting conditions or completely unidentifiable elements
        fault_warning = True if confidence_value < 0.30 else False

        return {
            "status": "Processed Successfully",
            "detected_object": detected_object,
            "category": category_type,
            "confidence": conf_percentage,
            "fault_warning": fault_warning
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Pipeline Crash: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def home_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VisionSort UI Control Center</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #0b0c10; color: #e5e7eb; padding: 40px; margin: 0; }
            .container { max-width: 850px; margin: auto; background: #1f2833; padding: 35px; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.7); border: 1px solid #45f3ff33; }
            h1 { color: #66fcf1; border-bottom: 2px solid #1f2833; padding-bottom: 12px; margin-top: 0; font-size: 26px; text-transform: uppercase; letter-spacing: 1px; }
            .upload-box { border: 2px dashed #66fcf1; padding: 40px; text-align: center; border-radius: 8px; background: #0b0c10; margin-bottom: 25px; }
            input[type="file"] { margin: 0 auto 15px auto; display: block; color: #c5a1ff; font-size: 14px; }
            button { background-color: #66fcf1; color: #0b0c10; border: none; padding: 14px 28px; font-size: 16px; font-weight: bold; border-radius: 6px; cursor: pointer; width: 100%; transition: 0.2s; box-shadow: 0 4px 15px rgba(102,252,241,0.3); }
            button:hover { background-color: #45e3d7; transform: translateY(-1px); }
            .results { margin-top: 25px; background: #0b0c10; padding: 25px; border-radius: 8px; border-left: 5px solid #66fcf1; }
            .metric { font-size: 17px; margin: 12px 0; color: #cbd5e1; }
            .bold { font-weight: bold; color: #66fcf1; font-family: 'Courier New', Courier, monospace; font-size: 19px; }
            .fault-active { border-left: 5px solid #ffbd69 !important; background: #241c15 !important; }
            .warning-text { color: #ffbd69; font-weight: bold; display: none; margin-top: 12px; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VisionSort UI Control Center</h1>
            <p style="color: #c5a1ff; margin-top: -10px; font-weight: 500;">Multi-Stream Industrial Sorting Core — 10 Target Matrix Arrays</p>
            
            <div class="upload-box">
                <input type="file" id="imageInput" accept="image/*">
                <button type="button" onclick="uploadAndAnalyze()">Analyze Waste Sample</button>
            </div>

            <div class="results" id="resultsSection" style="display: none;">
                <h3>Analysis Metrics Report</h3>
                <div class="metric">Pipeline State: <span class="bold" style="color: #66fcf1;">STREAM VERIFIED</span></div>
                <div class="metric">Identified Stream Element: <span id="objLabel" class="bold">-</span></div>
                <div class="metric">Assigned Classification: <span id="catLabel" class="bold">-</span></div>
                <div class="metric">Target Confidence Matrix: <span id="confLabel" class="bold">-</span></div>
                <div id="faultWarning" class="warning-text">⚠️ PROCESS DISPATCH NOTE: Sample context matched with a low-confidence metric score. Inspect conveyor array.</div>
            </div>
        </div>

        <script>
        async function uploadAndAnalyze() {
            const fileInput = document.getElementById('imageInput');
            if(!fileInput.files[0]) {
                alert("Please select a valid image sample frame first!");
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            try {
                const response = await fetch("/api/analyze", {
                    method: "POST",
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('objLabel').innerText = data.detected_object;
                    document.getElementById('catLabel').innerText = data.category;
                    document.getElementById('confLabel').innerText = data.confidence;
                    
                    const resultsSection = document.getElementById('resultsSection');
                    const faultWarning = document.getElementById('faultWarning');
                    
                    if (data.fault_warning) {
                        resultsSection.classList.add('fault-active');
                        faultWarning.style.display = 'block';
                    } else {
                        resultsSection.classList.remove('fault-active');
                        faultWarning.style.display = 'none';
                    }
                    
                    resultsSection.style.display = "block";
                } else {
                    alert("System Fault Response: " + data.detail);
                }
            } catch (err) {
                console.error(err);
                alert("Failed to reach out to the localized engine server.");
            }
        }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:6400/")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=6400, reload=False)
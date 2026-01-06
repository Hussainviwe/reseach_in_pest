# Cashew Pest Detection System

An end-to-end AI-powered pest detection system for cashew plants using YOLOv8 deep learning model, deployed as a REST API with mobile and web interfaces.

## Features

- **YOLOv8 Object Detection**: State-of-the-art deep learning model optimized for small pest detection
- **Grad-CAM Visualization**: Explainable AI showing which image regions influenced predictions
- **Real-time Detection**: Fast inference for immediate pest identification
- **Eco-Friendly Recommendations**: Treatment suggestions using organic and biological control methods
- **Multi-Platform**: Web and mobile applications for accessibility
- **Roboflow Integration**: Streamlined data management and augmentation pipeline
- **Small Object Optimization**: Enhanced preprocessing for detecting tiny insects

## Supported Pests

This system is specialized for detecting three critical cashew pests:

1. **Red Mite** (*Oligonychus coffeae*) - Severity: HIGH
   - Tiny arachnids (0.3-0.5mm) causing leaf bronzing and drop
   - Rapid multiplication in hot, dry conditions

2. **Stem Borer** (*Plocaederus ferrugineus*) - Severity: HIGH
   - Beetle larvae tunneling into stems and roots
   - Causes wilting, branch dieback, and plant death

3. **Thrips** (*Scirtothrips dorsalis*) - Severity: MEDIUM
   - Minute insects (1-2mm) causing leaf curling and fruit scarring
   - Rapid reproduction, 12-15 generations per year

> **Note**: For detailed pest information, treatment recommendations, and field identification guides, see [README_PESTS.md](README_PESTS.md)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Applications                     │
│  ┌────────────────┐              ┌─────────────────┐        │
│  │  Web App       │              │  Mobile App     │        │
│  │  (React)       │              │  (React Native) │        │
│  └────────────────┘              └─────────────────┘        │
└───────────────────┬────────────────────────┬────────────────┘
                    │                        │
                    │    REST API (FastAPI)  │
                    ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend Service                         │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │   YOLOv8     │  │   Grad-CAM    │  │  Recommender    │  │
│  │   Detector   │  │   Visualizer  │  │    System       │  │
│  └──────────────┘  └───────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data & Model Storage                       │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │  Training    │  │   Trained     │  │  Treatment      │  │
│  │  Dataset     │  │   Models      │  │  Database       │  │
│  └──────────────┘  └───────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for web app)
- CUDA-capable GPU (recommended for training)
- Roboflow account with annotated dataset

### 1. Clone Repository

```bash
git clone <repository-url>
cd cashew-pest-detection
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp ../.env.example .env
# Edit .env and add your configuration
```

### 3. Download Dataset from Roboflow

```bash
# Set your Roboflow credentials in .env
# ROBOFLOW_API_KEY=your_api_key
# ROBOFLOW_WORKSPACE=your_workspace
# ROBOFLOW_PROJECT=your_project

# Download dataset
python ../scripts/download_roboflow_data.py \
  --workspace your_workspace \
  --project cashew_pests \
  --version 1
```

### 4. Train Model

```bash
# Train YOLOv8 model
python training/train.py \
  --data data/dataset.yaml \
  --model yolov8m \
  --epochs 100 \
  --batch 16
```

### 5. Start Backend API

```bash
# Run API server
python api/main.py
# API will be available at http://localhost:8000
```

### 6. Start Web Application

```bash
# Navigate to web frontend
cd ../frontend/web

# Install dependencies
npm install

# Create environment file
echo "VITE_API_BASE_URL=http://localhost:8000" > .env

# Start development server
npm run dev
# Web app will be available at http://localhost:5173
```

## Project Structure

```
cashew-pest-detection/
├── backend/
│   ├── api/                     # FastAPI application
│   │   ├── main.py             # Main API file
│   │   ├── models/             # Pydantic schemas
│   │   └── routes/             # API routes
│   ├── core/                    # Core modules
│   │   ├── config.py           # Configuration
│   │   ├── detector.py         # YOLOv8 wrapper
│   │   ├── gradcam.py          # Grad-CAM implementation
│   │   └── preprocessing.py    # Image preprocessing
│   ├── training/                # Training scripts
│   │   ├── train.py            # Main training script
│   │   └── evaluate.py         # Evaluation utilities
│   ├── recommendations/         # Treatment recommendation system
│   │   ├── recommender.py      # Recommendation logic
│   │   └── pest_database.json  # Pest treatment database
│   ├── data/                    # Dataset storage
│   ├── models/                  # Model weights
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── web/                     # React web application
│   │   ├── src/
│   │   │   ├── components/     # React components
│   │   │   ├── services/       # API client
│   │   │   └── App.jsx         # Main app component
│   │   └── package.json
│   └── mobile/                  # React Native app (optional)
├── scripts/                     # Utility scripts
│   └── download_roboflow_data.py
├── docker-compose.yml
├── .env.example
└── README.md
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Model Configuration
MODEL_PATH=backend/models/trained/best.pt
MODEL_CONFIDENCE=0.25
MODEL_IOU_THRESHOLD=0.45
MODEL_DEVICE=cuda

# Roboflow Configuration
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=cashew_pests
ROBOFLOW_VERSION=1

# Grad-CAM
GRADCAM_ENABLED=True

# CORS
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
```

## Training Configuration

The training script is optimized for small object detection with the following settings:

- **Augmentation**: Mosaic, mixup, HSV augmentation
- **Small Object Optimization**: Enhanced preprocessing, CLAHE
- **Learning Rate**: 0.01 with warmup
- **Optimizer**: AdamW
- **Early Stopping**: Patience of 50 epochs
- **Batch Size**: 16 (adjustable based on GPU memory)

### Training Command Options

```bash
python backend/training/train.py \
  --data path/to/dataset.yaml \
  --model yolov8m \              # Model variant: n, s, m, l, x
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device cuda \
  --project cashew_pest
```

## API Documentation

Once the API is running, visit:

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### `POST /detect`

Detect pests in a single image.

**Parameters**:
- `file`: Image file (multipart/form-data)
- `confidence`: Confidence threshold (optional, default: 0.25)
- `enable_gradcam`: Enable Grad-CAM visualization (optional, default: true)
- `enhance_small_objects`: Apply small object enhancement (optional, default: true)

**Response**:
```json
{
  "success": true,
  "num_detections": 2,
  "inference_time": 0.234,
  "detections": [
    {
      "class_id": 0,
      "class_name": "Tea Mosquito Bug",
      "confidence": 0.87,
      "bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 250}
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "gradcam_image": "data:image/jpeg;base64,...",
  "recommendations": [...]
}
```

#### `GET /health`

Check API health status.

#### `GET /model/info`

Get information about the loaded model.

## Model Performance

Expected performance metrics (on validation set):

- **mAP@0.5**: > 85%
- **mAP@0.5:0.95**: > 60%
- **Inference Time**: < 500ms per image (GPU)
- **Small Object IoU**: > 0.5 for objects < 20x20px

## Grad-CAM Visualization

The system includes Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations:

- Highlights image regions that influenced the model's decision
- Builds user trust through transparency
- Helps identify false positives
- Validates model focus on relevant features

## Treatment Recommendations

The system provides eco-friendly treatment options:

- **Organic Pesticides**: Neem oil, botanical extracts
- **Biological Control**: Beneficial insects, microbial pesticides
- **Cultural Practices**: Pruning, sanitation, monitoring
- **Prevention Measures**: Trap crops, habitat management

Each recommendation includes:
- Detailed application instructions
- Dosage and frequency
- Safety precautions
- Effectiveness information

## Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# API: http://localhost:8000
# Web: http://localhost:3000
```

### Production Deployment

1. **Backend**: Deploy to GPU-enabled cloud instance (AWS EC2 g4dn, GCP Compute)
2. **Frontend**: Deploy to Vercel, Netlify, or similar
3. **Model Storage**: Use cloud storage (S3, GCS) for model weights
4. **Database**: Optional MongoDB for detection history

### Performance Optimization

- Enable model quantization for faster inference
- Use TensorRT for GPU optimization
- Implement caching for repeated requests
- Load balance across multiple API instances

## Mobile Application (React Native)

The mobile app provides:

- Camera integration for real-time capture
- Offline detection capability (with model bundled)
- Detection history
- Push notifications for pest alerts

Setup instructions in `frontend/mobile/README.md`

## Data Collection Guidelines

For best results when collecting training data:

1. **Variety**: Include multiple pest stages, angles, and backgrounds
2. **Quality**: Use high-resolution images (min 640x640)
3. **Lighting**: Capture in various lighting conditions
4. **Annotation**: Use Roboflow for precise bounding box annotations
5. **Balance**: Maintain class balance or use weighted sampling

## Troubleshooting

### Model Not Loading

- Check `MODEL_PATH` in `.env`
- Ensure model file exists at specified path
- Verify PyTorch and CUDA compatibility

### Low Detection Accuracy

- Increase training epochs
- Use larger model variant (e.g., yolov8l instead of yolov8m)
- Add more training data
- Verify annotation quality

### API Connection Error

- Ensure backend is running on correct port
- Check CORS settings in `config.py`
- Verify `VITE_API_BASE_URL` in frontend `.env`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- YOLOv8 by Ultralytics
- Roboflow for data management
- PyTorch Grad-CAM library
- React and FastAPI communities

## Contact

For questions or support, please open an issue on GitHub.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{cashew_pest_detection,
  title={Cashew Pest Detection System using YOLOv8},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cashew-pest-detection}
}
```

## Future Enhancements

- [ ] Real-time video detection
- [ ] Multi-language support
- [ ] Pest severity estimation
- [ ] Automated treatment scheduling
- [ ] Integration with IoT sensors
- [ ] Crop health monitoring
- [ ] Disease detection (in addition to pests)
- [ ] Yield prediction based on pest pressure

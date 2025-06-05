# SignLanguageApp

- **frontend/**: Built with React Native (using Expo), allows users to capture hand gestures via camera and display translated results.
- **backend/**: Powered by FastAPI and Python, handles image processing and prediction using a trained machine learning model.

---

## 🚀 Features

- 🎥 Real-time hand gesture capture
- 🔤 Instant translation of sign language to text
- 🤖 Machine learning–based image classification
- 📡 REST API communication between frontend and backend
- 🧏 Focused on accessibility for the DHH community

---

## 🛠️ Tech Stack

| Layer     | Technology                     |
|-----------|--------------------------------|
| Frontend  | React Native, Expo, JavaScript |
| Backend   | FastAPI, Python, Uvicorn       |
| ML/Tools  | OpenCV, NumPy, PIL, custom model |

---

## 🔧 Getting Started

### ✅ Prerequisites

- Node.js & npm
- Python 3.8+
- Git
- Expo CLI (`npm install -g expo-cli`)
- A smartphone or emulator (for testing)

---

### ⚙️ Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload

# Cataract Detection System

This project is a full-stack application for detecting cataracts from eye images using a fine-tuned ResNet18 model. It consists of a FastAPI backend and a React (Vite) frontend.

## Prerequisites

- **Python**: 3.8+
- **Node.js**: 16+
- **npm**: (comes with Node.js)

## Project Structure

- `backend/`: Python FastAPI backend for model inference.
- `frontend/`: React TypeScript frontend for user interaction.

## Setup Instructions

### 1. Backend Setup

Navigate to the project root directory and install the Python dependencies. It is recommended to use a virtual environment.

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Frontend Setup

Navigate to the frontend directory and install the Node.js dependencies.

```bash
cd frontend
npm install
```

## Running the Application

### Start the Backend

Run the backend server from the **project root** directory:

```bash
python backend/main.py
```

The backend API will be available at `http://localhost:8000`.
- API Docs: `http://localhost:8000/docs`

### Start the Frontend

In a new terminal window, navigate to the frontend directory and start the development server:

```bash
cd frontend
npm run dev
```

The frontend application will be available at `http://localhost:3000` (or the port shown in your terminal).

## Usage

1. Open the frontend application in your browser.
2. Use the "Upload Image" button to select an eye image or use the "Camera" button to capture one.
3. The system will analyze the image and display the result (Normal/Cataract) along with a confidence score.

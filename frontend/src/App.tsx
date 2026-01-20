import { useState } from 'react';
import { CameraCapture } from './components/camera-capture';
import { ImageUpload } from './components/image-upload';
import { ResultsDisplay } from './components/results-display';
import { Header } from './components/header';

export interface DetectionResult {
  hasCataract: boolean;
  confidence: number;
  severity?: 'mild' | 'moderate' | 'severe';
  analyzedAt: Date;
}

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleImageSelected = async (imageData: string) => {
    setSelectedImage(imageData);
    setResult(null);
    setIsAnalyzing(true);

    try {
      // Convert base64 to blob
      const response = await fetch(imageData);
      const blob = await response.blob();

      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');

      const apiResponse = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error('Analysis failed');
      }

      const data = await apiResponse.json();

      const detectionResult: DetectionResult = {
        hasCataract: data.has_cataract,
        confidence: data.confidence,
        // Map confidence/probability to severity (simple heuristic)
        severity: data.confidence > 90 ? 'severe' : data.confidence > 75 ? 'moderate' : 'mild',
        analyzedAt: new Date()
      };

      setResult(detectionResult);
    } catch (error) {
      console.error('Error analyzing image:', error);
      // You might want to show an error state here
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResult(null);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 relative">
      {/* Background Image with Overlay */}
      <div
        className="fixed inset-0 z-0 opacity-10"
        style={{
          backgroundImage: 'url(https://images.unsplash.com/photo-1622475441028-8927e3772656?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtZWRpY2FsJTIwY2xpbmljJTIwYmFja2dyb3VuZHxlbnwxfHx8fDE3Njg4OTcwNzZ8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat'
        }}
      />

      <div className="relative z-10">
        <Header />

        <main className="container mx-auto px-4 py-8 max-w-6xl">
          {!selectedImage ? (
            <div className="space-y-8">
              <div className="text-center space-y-4 mb-12">
                <h2 className="text-4xl font-semibold text-slate-800">
                  AI-Powered Cataract Detection
                </h2>
                <p className="text-slate-600 max-w-2xl mx-auto text-lg">
                  Upload an eye image or use your camera to detect potential cataracts.
                  Our system analyzes the image and provides instant results.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
                <ImageUpload onImageSelected={handleImageSelected} />
                <CameraCapture onImageCaptured={handleImageSelected} />
              </div>

              <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 max-w-3xl mx-auto">
                <p className="text-amber-900 text-sm leading-relaxed">
                  <strong className="font-semibold">Medical Disclaimer:</strong> This tool is for educational purposes only
                  and should not be used as a substitute for professional medical advice.
                  Always consult with a qualified healthcare provider for proper diagnosis and treatment.
                </p>
              </div>
            </div>
          ) : (
            <ResultsDisplay
              image={selectedImage}
              result={result}
              isAnalyzing={isAnalyzing}
              onReset={handleReset}
            />
          )}
        </main>
      </div>
    </div>
  );
}
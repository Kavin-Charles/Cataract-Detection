import { Camera } from 'lucide-react';
import { useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';

interface CameraCaptureProps {
  onImageCaptured: (imageData: string) => void;
}

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "environment"
};

export function CameraCapture({ onImageCaptured }: CameraCaptureProps) {
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const webcamRef = useRef<Webcam>(null);

  const startCamera = () => {
    setError(null);
    setIsStreamActive(true);
  };

  const captureImage = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setIsStreamActive(false);
        onImageCaptured(imageSrc);
      }
    }
  }, [webcamRef, onImageCaptured]);

  const stopCamera = () => {
    setIsStreamActive(false);
    setError(null);
  };

  const handleUserMediaError = useCallback((error: string | DOMException) => {
    console.error('Camera error:', error);
    setError('Unable to access camera. Please check permissions and try again.');
    // Keep internal stream active state true so we show the error within the active view or revert?
    // Let's keep it simple: show error in the initial view if it fails immediately, 
    // but react-webcam renders null if error. 
    // Actually, better to catch it and maybe revert to button view with error message.
    // For now, let's just display the error message in the active view if possible, or just revert.
    // Reverting seems safer for UX so they can try again.
    setIsStreamActive(false);
  }, []);

  if (isStreamActive) {
    return (
      <div className="bg-white rounded-2xl p-6 border-2 border-blue-400 shadow-lg shadow-blue-500/10">
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden bg-slate-900 min-h-[300px]">
            {/* Wrapper to ensure aspect ratio or min height */}
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              onUserMediaError={handleUserMediaError}
              className="w-full h-full object-cover"
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={captureImage}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-medium hover:from-blue-600 hover:to-blue-700 transition-all shadow-md"
            >
              Capture Photo
            </button>
            <button
              onClick={stopCamera}
              className="px-6 py-3 bg-slate-100 text-slate-700 rounded-lg font-medium hover:bg-slate-200 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      onClick={startCamera}
      className="bg-white rounded-2xl p-8 border-2 border-dashed border-slate-300 hover:border-blue-400 transition-all cursor-pointer group hover:shadow-lg hover:shadow-blue-500/10"
    >
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="bg-blue-50 group-hover:bg-blue-100 p-5 rounded-2xl transition-colors">
          <Camera className="size-10 text-blue-600" strokeWidth={2} />
        </div>

        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-slate-800">Use Camera</h3>
          <p className="text-slate-500 text-sm leading-relaxed">
            Take a photo using your device camera
          </p>
        </div>

        {error && (
          <p className="text-red-600 text-xs bg-red-50 px-3 py-2 rounded-lg">
            {error}
          </p>
        )}

        <button className="px-6 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-medium hover:from-blue-600 hover:to-blue-700 transition-all shadow-md hover:shadow-lg">
          Open Camera
        </button>
      </div>
    </div>
  );
}

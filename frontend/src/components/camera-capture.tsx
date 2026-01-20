import { Camera } from 'lucide-react';
import { useRef, useState } from 'react';

interface CameraCaptureProps {
  onImageCaptured: (imageData: string) => void;
}

export function CameraCapture({ onImageCaptured }: CameraCaptureProps) {
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreamActive(true);
      }
    } catch (err) {
      setError('Unable to access camera. Please check permissions.');
      console.error('Camera access error:', err);
    }
  };

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Stop camera stream
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
        
        setIsStreamActive(false);
        onImageCaptured(imageData);
      }
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setIsStreamActive(false);
  };

  if (isStreamActive) {
    return (
      <div className="bg-white rounded-2xl p-6 border-2 border-blue-400 shadow-lg shadow-blue-500/10">
        <div className="space-y-4">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline
            className="w-full rounded-xl bg-slate-900"
          />
          
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

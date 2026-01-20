import { Upload } from 'lucide-react';
import { useRef } from 'react';

interface ImageUploadProps {
  onImageSelected: (imageData: string) => void;
}

export function ImageUpload({ onImageSelected }: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onImageSelected(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div 
      onClick={() => fileInputRef.current?.click()}
      className="bg-white rounded-2xl p-8 border-2 border-dashed border-slate-300 hover:border-blue-400 transition-all cursor-pointer group hover:shadow-lg hover:shadow-blue-500/10"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
      
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="bg-blue-50 group-hover:bg-blue-100 p-5 rounded-2xl transition-colors">
          <Upload className="size-10 text-blue-600" strokeWidth={2} />
        </div>
        
        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-slate-800">Upload Image</h3>
          <p className="text-slate-500 text-sm leading-relaxed">
            Select an eye image from your device
          </p>
        </div>
        
        <button className="px-6 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-medium hover:from-blue-600 hover:to-blue-700 transition-all shadow-md hover:shadow-lg">
          Choose File
        </button>
      </div>
    </div>
  );
}

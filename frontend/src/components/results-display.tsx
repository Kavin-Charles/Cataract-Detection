import { AlertCircle, CheckCircle, Loader2, RotateCcw, AlertTriangle } from 'lucide-react';
import type { DetectionResult } from '../App';

interface ResultsDisplayProps {
  image: string;
  result: DetectionResult | null;
  isAnalyzing: boolean;
  onReset: () => void;
}

export function ResultsDisplay({ image, result, isAnalyzing, onReset }: ResultsDisplayProps) {
  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'mild': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'moderate': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'severe': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-slate-600 bg-slate-50 border-slate-200';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <button
        onClick={onReset}
        className="flex items-center gap-2 px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-white rounded-lg transition-colors"
      >
        <RotateCcw className="size-4" />
        <span className="font-medium">Analyze New Image</span>
      </button>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Image Preview */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
          <h3 className="font-semibold text-slate-800 mb-4">Uploaded Image</h3>
          <img 
            src={image} 
            alt="Eye scan" 
            className="w-full rounded-xl border border-slate-200 object-cover"
          />
        </div>

        {/* Results */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
          <h3 className="font-semibold text-slate-800 mb-4">Analysis Results</h3>
          
          {isAnalyzing ? (
            <div className="flex flex-col items-center justify-center py-12 gap-4">
              <Loader2 className="size-12 text-blue-600 animate-spin" />
              <div className="text-center">
                <p className="text-slate-800 font-medium">Analyzing image...</p>
                <p className="text-slate-500 text-sm mt-1">This may take a moment</p>
              </div>
            </div>
          ) : result ? (
            <div className="space-y-4">
              {/* Main Result */}
              <div className={`rounded-xl p-5 border-2 ${
                result.hasCataract 
                  ? 'bg-red-50 border-red-200' 
                  : 'bg-green-50 border-green-200'
              }`}>
                <div className="flex items-start gap-3">
                  {result.hasCataract ? (
                    <AlertCircle className="size-6 text-red-600 flex-shrink-0 mt-0.5" />
                  ) : (
                    <CheckCircle className="size-6 text-green-600 flex-shrink-0 mt-0.5" />
                  )}
                  <div>
                    <h4 className={`font-semibold mb-1 ${
                      result.hasCataract ? 'text-red-900' : 'text-green-900'
                    }`}>
                      {result.hasCataract ? 'Cataract Detected' : 'No Cataract Detected'}
                    </h4>
                    <p className={`text-sm ${
                      result.hasCataract ? 'text-red-700' : 'text-green-700'
                    }`}>
                      {result.hasCataract 
                        ? 'Potential signs of cataract found in the image' 
                        : 'No significant cataract indicators detected'
                      }
                    </p>
                  </div>
                </div>
              </div>

              {/* Confidence & Severity */}
              <div className="space-y-3">
                <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-slate-700">Confidence Level</span>
                    <span className="text-sm font-semibold text-slate-900">
                      {result.confidence.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-500"
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>

                {result.hasCataract && result.severity && (
                  <div className={`rounded-lg p-4 border ${getSeverityColor(result.severity)}`}>
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="size-5" />
                      <div>
                        <span className="text-sm font-medium">Severity: </span>
                        <span className="text-sm font-semibold capitalize">{result.severity}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="text-xs text-slate-500 pt-2">
                  Analyzed on {result.analyzedAt.toLocaleString()}
                </div>
              </div>

              {/* Recommendations */}
              {result.hasCataract && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
                  <h5 className="font-semibold text-blue-900 mb-2 text-sm">Recommended Actions</h5>
                  <ul className="text-sm text-blue-800 space-y-1.5">
                    <li className="flex gap-2">
                      <span className="text-blue-600">•</span>
                      <span>Schedule an appointment with an ophthalmologist</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-blue-600">•</span>
                      <span>Get a comprehensive eye examination</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-blue-600">•</span>
                      <span>Discuss treatment options with your doctor</span>
                    </li>
                  </ul>
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-5">
        <p className="text-amber-900 text-sm leading-relaxed">
          <strong className="font-semibold">Important:</strong> These results are generated by AI and 
          are for informational purposes only. This is not a medical diagnosis. Please consult with a 
          qualified eye care professional for accurate diagnosis and treatment recommendations.
        </p>
      </div>
    </div>
  );
}

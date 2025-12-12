'use client'

import { StateAnalysisResponse } from '@/lib/api/client'
import {
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Package,
  TrendingUp,
  Image as ImageIcon,
} from 'lucide-react'

interface VisualGuidanceProps {
  analysis: StateAnalysisResponse
  onRetry?: () => void
}

export const VisualGuidance: React.FC<VisualGuidanceProps> = ({
  analysis,
  onRetry,
}) => {
  const progressColor =
    analysis.progress_percentage >= 75
      ? 'bg-green-500'
      : analysis.progress_percentage >= 50
      ? 'bg-blue-500'
      : analysis.progress_percentage >= 25
      ? 'bg-yellow-500'
      : 'bg-gray-500'

  return (
    <div className="space-y-6">
      {/* Progress Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-800">Assembly Progress</h2>
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <span className="text-2xl font-bold text-blue-600">
              {analysis.progress_percentage.toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-4 mb-3">
          <div
            className={`${progressColor} h-4 rounded-full transition-all duration-500`}
            style={{ width: `${analysis.progress_percentage}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-sm text-gray-600">
          <span>
            Step {analysis.current_step} of {analysis.total_steps}
          </span>
          <span>
            {analysis.completed_steps.length} steps completed
          </span>
        </div>
      </div>

      {/* Encouragement */}
      {analysis.encouragement && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
          <p className="text-blue-800 font-medium text-center">
            {analysis.encouragement}
          </p>
        </div>
      )}

      {/* Errors and Corrections */}
      {analysis.error_corrections.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-yellow-800 mb-2">
                Attention Needed
              </h3>
              <ul className="space-y-2">
                {analysis.error_corrections.map((correction, index) => (
                  <li key={index} className="text-sm text-yellow-700">
                    {correction}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Next Step Instruction */}
      {analysis.instruction && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center space-x-2 mb-4">
            <ArrowRight className="w-6 h-6 text-green-600" />
            <h2 className="text-xl font-bold text-gray-800">Next Step</h2>
            {analysis.next_step_number && (
              <span className="bg-green-100 text-green-800 text-sm font-medium px-3 py-1 rounded-full">
                Step {analysis.next_step_number}
              </span>
            )}
          </div>

          <div className="prose prose-sm max-w-none">
            <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
              {analysis.instruction}
            </p>
          </div>

          {/* Reference Image */}
          {analysis.reference_image && (
            <div className="mt-4 border border-gray-200 rounded-lg overflow-hidden">
              <div className="bg-gray-50 px-4 py-2 flex items-center space-x-2">
                <ImageIcon className="w-4 h-4 text-gray-600" />
                <span className="text-sm font-medium text-gray-700">
                  Reference Image
                </span>
              </div>
              <img
                src={`http://localhost:8000/api/image?path=${encodeURIComponent(
                  analysis.reference_image
                )}`}
                alt="Step reference"
                className="w-full h-auto"
              />
            </div>
          )}
        </div>
      )}

      {/* Parts Needed */}
      {analysis.parts_needed.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Package className="w-6 h-6 text-purple-600" />
            <h2 className="text-xl font-bold text-gray-800">Parts Needed</h2>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {analysis.parts_needed.map((part, index) => (
              <div
                key={index}
                className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg border border-gray-200"
              >
                <div className="flex-shrink-0 w-12 h-12 bg-white rounded-lg border-2 border-gray-300 flex items-center justify-center">
                  <span className="text-lg font-bold text-gray-600">
                    {part.quantity}×
                  </span>
                </div>
                <div className="flex-1">
                  <div className="font-medium text-gray-800">
                    {part.color} {part.shape || 'piece'}
                  </div>
                  <div className="text-sm text-gray-600">
                    {part.description}
                  </div>
                  {part.part_id && (
                    <div className="text-xs text-gray-500 mt-1">
                      Part ID: {part.part_id}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detected State Summary */}
      {analysis.detected_parts.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-bold text-gray-800 mb-4">
            Detected Assembly State
          </h2>

          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Parts Detected:</span>
              <span className="font-semibold text-gray-800">
                {analysis.detected_parts.length}
              </span>
            </div>

            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Structures Built:</span>
              <span className="font-semibold text-gray-800">
                {analysis.assembled_structures.length}
              </span>
            </div>

            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Detection Confidence:</span>
              <span className="font-semibold text-gray-800">
                {(analysis.detection_confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Guidance Confidence:</span>
              <span className="font-semibold text-gray-800">
                {(analysis.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Completed Steps */}
          {analysis.completed_steps.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="flex items-center space-x-2 mb-2">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-gray-700">
                  Completed Steps
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {analysis.completed_steps.map((step) => (
                  <span
                    key={step}
                    className="bg-green-100 text-green-800 text-xs font-medium px-2 py-1 rounded"
                  >
                    {step}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Retry Button */}
      {onRetry && (
        <button
          onClick={onRetry}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
        >
          Analyze Again
        </button>
      )}
    </div>
  )
}



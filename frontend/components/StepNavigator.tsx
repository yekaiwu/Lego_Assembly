'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { useManualStore } from '@/lib/store/manualStore'
import { ChevronLeft, ChevronRight, Loader2, Image as ImageIcon } from 'lucide-react'
import { useState, useEffect } from 'react'

export default function StepNavigator() {
  const { selectedManual, currentStep, nextStep, previousStep, totalSteps } =
    useManualStore()

  const [imageError, setImageError] = useState(false)

  const { data: stepData, isLoading } = useQuery({
    queryKey: ['step', selectedManual, currentStep],
    queryFn: () => api.getStep(selectedManual!, currentStep!),
    enabled: !!selectedManual && !!currentStep,
  })

  const { data: stepsData } = useQuery({
    queryKey: ['manual-steps', selectedManual],
    queryFn: () => api.getManualSteps(selectedManual!),
    enabled: !!selectedManual,
  })

  // Reset image error when step changes
  useEffect(() => {
    setImageError(false)
  }, [currentStep])

  if (!selectedManual || !currentStep) {
    return (
      <div className="text-center py-12 text-gray-400">
        <ImageIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p>Select a manual to view steps</p>
      </div>
    )
  }

  const currentStepDetails = stepsData?.steps.find(
    (s) => s.step_number === currentStep
  )

  const imagePath = currentStepDetails?.image_path
  const imageUrl = imagePath ? api.getImageUrl(imagePath) : null

  return (
    <div className="space-y-4">
      {/* Step Image */}
      <div className="relative bg-gray-100 rounded-lg overflow-hidden aspect-[4/3]">
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="w-8 h-8 animate-spin text-lego-blue" />
          </div>
        ) : imageUrl && !imageError ? (
          <img
            src={imageUrl}
            alt={`Step ${currentStep}`}
            className="w-full h-full object-contain"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <ImageIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">Image not available</p>
            </div>
          </div>
        )}

        {/* Step Counter Overlay */}
        <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded-full text-sm font-semibold">
          Step {currentStep} of {totalSteps}
        </div>
      </div>

      {/* Step Information */}
      {stepData && (
        <div className="bg-gray-50 rounded-lg p-4 space-y-3">
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Instructions:</h3>
            <p className="text-gray-700 text-sm leading-relaxed">
              {stepData.answer}
            </p>
          </div>

          {stepData.parts_needed && stepData.parts_needed.length > 0 && (
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Parts Needed:</h3>
              <ul className="space-y-1">
                {stepData.parts_needed.map((part, idx) => (
                  <li key={idx} className="text-sm text-gray-700">
                    • {part.quantity}x {part.color} {part.description}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {stepData.guidance && (
            <div className="bg-blue-50 border border-blue-200 rounded p-3">
              <p className="text-sm text-blue-800">💡 {stepData.guidance}</p>
            </div>
          )}
        </div>
      )}

      {/* Navigation Buttons */}
      <div className="flex items-center justify-between space-x-4">
        <button
          onClick={previousStep}
          disabled={currentStep === 1}
          className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-lg font-medium transition-colors ${
            currentStep === 1
              ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
              : 'bg-lego-blue text-white hover:bg-blue-700'
          }`}
        >
          <ChevronLeft className="w-5 h-5" />
          <span>Previous</span>
        </button>

        <button
          onClick={nextStep}
          disabled={currentStep === totalSteps}
          className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-lg font-medium transition-colors ${
            currentStep === totalSteps
              ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
              : 'bg-lego-green text-white hover:bg-green-700'
          }`}
        >
          <span>Next</span>
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>

      {/* Progress Bar */}
      <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className="bg-lego-yellow h-full transition-all duration-300"
          style={{ width: `${(currentStep / totalSteps) * 100}%` }}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => {
            // Jump to step logic - could open a modal
          }}
          className="text-sm px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-gray-700 font-medium transition-colors"
        >
          Jump to Step
        </button>
        <button
          className="text-sm px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-gray-700 font-medium transition-colors"
        >
          Show Dependencies
        </button>
      </div>
    </div>
  )
}



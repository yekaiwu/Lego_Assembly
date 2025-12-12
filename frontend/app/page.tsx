'use client'

import { useState } from 'react'
import ManualSelector from '@/components/ManualSelector'
import StepNavigator from '@/components/StepNavigator'
import ChatInterface from '@/components/ChatInterface'
import { ImageUpload } from '@/components/ImageUpload'
import { VisualGuidance } from '@/components/VisualGuidance'
import { useManualStore } from '@/lib/store/manualStore'
import { api, StateAnalysisResponse } from '@/lib/api/client'
import { MessageSquare, Camera, Loader2 } from 'lucide-react'

type TabType = 'chat' | 'vision'

export default function Home() {
  const { selectedManual, currentStep } = useManualStore()
  const [activeTab, setActiveTab] = useState<TabType>('chat')
  const [selectedImages, setSelectedImages] = useState<File[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<StateAnalysisResponse | null>(null)
  const [error, setError] = useState<string>('')

  const handleAnalyze = async () => {
    if (!selectedManual) {
      setError('Please select a manual first')
      return
    }

    if (selectedImages.length < 2) {
      setError('Please upload at least 2 images')
      return
    }

    setIsAnalyzing(true)
    setError('')

    try {
      // Step 1: Upload images
      const uploadResponse = await api.uploadImages(selectedImages)
      
      // Step 2: Analyze assembly state
      const analysisResponse = await api.analyzeAssemblyState(
        selectedManual,
        uploadResponse.session_id
      )

      setAnalysisResult(analysisResponse)

      // Step 3: Cleanup session (optional, can be done later)
      // await api.cleanupSession(uploadResponse.session_id)
    } catch (err: any) {
      console.error('Analysis error:', err)
      setError(err.response?.data?.detail || 'Failed to analyze assembly. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleRetry = () => {
    setAnalysisResult(null)
    setSelectedImages([])
    setError('')
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md border-b-4 border-lego-red">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-lego-red rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">🧱</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-900">
                LEGO Builder Assistant
              </h1>
            </div>
            
            {selectedManual && (
              <div className="text-sm text-gray-600">
                Manual: <span className="font-semibold">{selectedManual}</span>
                {currentStep && (
                  <span className="ml-3">
                    Step: <span className="font-semibold">{currentStep}</span>
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Manual Selection */}
        <div className="mb-6">
          <ManualSelector />
        </div>

        {selectedManual ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Step Navigator */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">
                📸 Current Step View
              </h2>
              <StepNavigator />
            </div>

            {/* Right: Tabbed Interface */}
            <div className="bg-white rounded-lg shadow-lg overflow-hidden">
              {/* Tab Headers */}
              <div className="flex border-b border-gray-200">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`flex-1 px-6 py-4 font-medium transition-colors ${
                    activeTab === 'chat'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-center space-x-2">
                    <MessageSquare className="w-5 h-5" />
                    <span>Text Chat</span>
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('vision')}
                  className={`flex-1 px-6 py-4 font-medium transition-colors ${
                    activeTab === 'vision'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-center space-x-2">
                    <Camera className="w-5 h-5" />
                    <span>Photo Analysis</span>
                  </div>
                </button>
              </div>

              {/* Tab Content */}
              <div className="p-6">
                {activeTab === 'chat' ? (
                  <div>
                    <h2 className="text-xl font-semibold mb-4 text-gray-800">
                      💬 Chat Assistant
                    </h2>
                    <ChatInterface />
                  </div>
                ) : (
                  <div>
                    <h2 className="text-xl font-semibold mb-4 text-gray-800">
                      📷 Assembly State Analysis
                    </h2>
                    
                    {!analysisResult ? (
                      <div className="space-y-4">
                        <ImageUpload
                          onImagesSelected={setSelectedImages}
                          maxImages={4}
                          minImages={2}
                        />

                        {error && (
                          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                            {error}
                          </div>
                        )}

                        {selectedImages.length >= 2 && (
                          <button
                            onClick={handleAnalyze}
                            disabled={isAnalyzing}
                            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
                          >
                            {isAnalyzing ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                <span>Analyzing Assembly...</span>
                              </>
                            ) : (
                              <>
                                <Camera className="w-5 h-5" />
                                <span>Analyze My Assembly</span>
                              </>
                            )}
                          </button>
                        )}
                      </div>
                    ) : (
                      <VisualGuidance
                        analysis={analysisResult}
                        onRetry={handleRetry}
                      />
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-lg p-12 text-center">
            <div className="text-6xl mb-4">🧱</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              Welcome to LEGO Builder Assistant
            </h2>
            <p className="text-gray-600 max-w-md mx-auto mb-4">
              Select a manual above to get started with AI-powered assembly guidance.
              I'll help you build your LEGO set step by step!
            </p>
            <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto mt-8">
              <div className="p-4 bg-blue-50 rounded-lg">
                <MessageSquare className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                <h3 className="font-semibold text-gray-800 mb-1">Text Chat</h3>
                <p className="text-sm text-gray-600">
                  Ask questions about any step in the manual
                </p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <Camera className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                <h3 className="font-semibold text-gray-800 mb-1">Photo Analysis</h3>
                <p className="text-sm text-gray-600">
                  Upload photos to track your progress automatically
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-12 py-6 bg-white border-t">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-600 text-sm">
          <p>LEGO Builder Assistant • Powered by Vision AI & RAG</p>
        </div>
      </footer>
    </main>
  )
}


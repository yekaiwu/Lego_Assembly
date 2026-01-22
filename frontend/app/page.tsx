'use client'

import { useState } from 'react'
import ManualSelector from '@/components/ManualSelector'
import StepNavigator from '@/components/StepNavigator'
import ChatInterface from '@/components/ChatInterface'
import VideoUpload from '@/components/VideoUpload'
import VideoStepPlayer from '@/components/VideoStepPlayer'
import { useManualStore } from '@/lib/store/manualStore'
import { AnalysisResults } from '@/lib/api/client'
import { MessageSquare, Network, Video } from 'lucide-react'
import Link from 'next/link'
import ApiDebug from '@/components/ApiDebug'

type TabType = 'chat' | 'video'

export default function Home() {
  const { selectedManual, currentStep } = useManualStore()
  const [activeTab, setActiveTab] = useState<TabType>('chat')
  const [videoAnalysisResults, setVideoAnalysisResults] = useState<AnalysisResults | null>(null)

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
              <div className="flex items-center space-x-4">
                <div className="text-sm text-gray-600">
                  Manual: <span className="font-semibold">{selectedManual}</span>
                  {currentStep && (
                    <span className="ml-3">
                      Step: <span className="font-semibold">{currentStep}</span>
                    </span>
                  )}
                </div>
                <Link
                  href="/graph"
                  className="flex items-center space-x-2 px-3 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg transition-colors text-sm font-medium"
                >
                  <Network className="w-4 h-4" />
                  <span>View Graph</span>
                </Link>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* API Debug (only in development) */}
        {process.env.NODE_ENV === 'development' && (
          <div className="mb-4">
            <ApiDebug />
          </div>
        )}
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
                  onClick={() => setActiveTab('video')}
                  className={`flex-1 px-6 py-4 font-medium transition-colors ${
                    activeTab === 'video'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-center space-x-2">
                    <Video className="w-5 h-5" />
                    <span>Video Analysis</span>
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
                      🎥 Video Step Detection
                    </h2>

                    {!videoAnalysisResults ? (
                      <VideoUpload
                        manualId={selectedManual!}
                        onAnalysisComplete={(results) => setVideoAnalysisResults(results)}
                      />
                    ) : (
                      <VideoStepPlayer
                        videoUrl={`/api/video/stream/${videoAnalysisResults.results.video_id}`}
                        analysisResults={videoAnalysisResults}
                        manualId={selectedManual!}
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
            <div className="grid grid-cols-2 gap-4 max-w-2xl mx-auto mt-8">
              <div className="p-4 bg-blue-50 rounded-lg">
                <MessageSquare className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                <h3 className="font-semibold text-gray-800 mb-1">Text Chat</h3>
                <p className="text-sm text-gray-600">
                  Ask questions about any step in the manual
                </p>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <Video className="w-8 h-8 text-green-600 mx-auto mb-2" />
                <h3 className="font-semibold text-gray-800 mb-1">Video Analysis</h3>
                <p className="text-sm text-gray-600">
                  Detect assembly steps from your build video
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


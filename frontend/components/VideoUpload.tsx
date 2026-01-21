'use client'

import { useState, useEffect } from 'react'
import { api, VideoUploadResponse, AnalysisResults } from '@/lib/api/client'

interface VideoUploadProps {
  manualId: string
  onAnalysisComplete: (results: AnalysisResults) => void
}

export default function VideoUpload({ manualId, onAnalysisComplete }: VideoUploadProps) {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'uploaded'>('idle')
  const [analysisId, setAnalysisId] = useState<string | null>(null)
  const [analysisStatus, setAnalysisStatus] = useState<'processing' | 'completed' | 'error'>('processing')
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState('Initializing...')
  const [error, setError] = useState<string | null>(null)
  const [uploadData, setUploadData] = useState<VideoUploadResponse | null>(null)

  // Poll analysis status
  useEffect(() => {
    if (!analysisId || analysisStatus !== 'processing') return

    const pollInterval = setInterval(async () => {
      try {
        const status = await api.getAnalysisStatus(analysisId)

        if (status.status === 'completed') {
          clearInterval(pollInterval)
          setAnalysisStatus('completed')
          setProgress(100)
          setCurrentStep('Analysis complete!')
          onAnalysisComplete(status)
        } else if (status.status === 'error') {
          clearInterval(pollInterval)
          setAnalysisStatus('error')
          setError('Analysis failed. Please try again.')
        } else {
          // Update progress
          const progressPercent = status.results.progress_percentage || 0
          const step = status.results.current_step || 'Processing...'
          setProgress(progressPercent)
          setCurrentStep(step)
        }
      } catch (err) {
        console.error('Error polling analysis status:', err)
      }
    }, 5000) // Poll every 5 seconds

    return () => clearInterval(pollInterval)
  }, [analysisId, analysisStatus, onAnalysisComplete])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setVideoFile(files[0])
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!videoFile) return

    try {
      setUploadStatus('uploading')
      setError(null)

      // Upload video
      const uploadResponse = await api.uploadVideo(manualId, videoFile)
      setUploadData(uploadResponse)
      setUploadStatus('uploaded')

      // Start analysis
      const analysisResponse = await api.analyzeVideo(manualId, uploadResponse.video_id)
      setAnalysisId(analysisResponse.analysis_id)
      setAnalysisStatus('processing')
      setProgress(0)
      setCurrentStep('Starting analysis...')
    } catch (err: any) {
      console.error('Error uploading video:', err)
      setError(err.response?.data?.detail || 'Failed to upload video')
      setUploadStatus('idle')
    }
  }

  const handleReset = () => {
    setVideoFile(null)
    setUploadStatus('idle')
    setAnalysisId(null)
    setAnalysisStatus('processing')
    setProgress(0)
    setCurrentStep('Initializing...')
    setError(null)
    setUploadData(null)
  }

  return (
    <div className="video-upload-container bg-white rounded-lg shadow p-6">
      <h3 className="text-xl font-bold mb-4">Upload Assembly Video</h3>

      {uploadStatus === 'idle' && (
        <div>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <input
              type="file"
              accept="video/mp4,video/mov,video/avi"
              onChange={handleFileSelect}
              className="hidden"
              id="video-upload"
            />
            <label
              htmlFor="video-upload"
              className="cursor-pointer flex flex-col items-center"
            >
              <svg
                className="w-12 h-12 text-gray-400 mb-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="text-lg text-gray-700">
                {videoFile ? videoFile.name : 'Click to select video or drag and drop'}
              </p>
              <p className="text-sm text-gray-500 mt-2">Supported: MP4, MOV, AVI (max 500MB)</p>
            </label>
          </div>

          {videoFile && (
            <div className="mt-4 flex gap-2">
              <button
                onClick={handleUpload}
                className="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
              >
                Upload and Analyze
              </button>
              <button
                onClick={() => setVideoFile(null)}
                className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-100 transition"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}

      {uploadStatus === 'uploading' && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-lg text-gray-700">Uploading video...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a moment</p>
        </div>
      )}

      {uploadStatus === 'uploaded' && analysisStatus === 'processing' && (
        <div className="py-6">
          <div className="mb-4">
            <div className="flex justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Analyzing video...</span>
              <span className="text-sm font-medium text-gray-700">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>

          <p className="text-sm text-gray-600 text-center">{currentStep}</p>
          <p className="text-xs text-gray-500 text-center mt-2">
            This process takes 2-5 minutes. We're detecting assembly steps in your video using AI.
          </p>

          {uploadData && (
            <div className="mt-4 p-3 bg-gray-50 rounded text-sm">
              <p className="text-gray-600">
                <strong>Duration:</strong> {uploadData.duration_sec.toFixed(1)}s
              </p>
              <p className="text-gray-600">
                <strong>Resolution:</strong> {uploadData.resolution[0]}x{uploadData.resolution[1]}
              </p>
              <p className="text-gray-600">
                <strong>FPS:</strong> {uploadData.fps}
              </p>
            </div>
          )}
        </div>
      )}

      {analysisStatus === 'completed' && (
        <div className="text-center py-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
            <svg
              className="w-8 h-8 text-green-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
          <p className="text-lg font-semibold text-gray-900">Analysis Complete!</p>
          <p className="text-sm text-gray-600 mt-2">
            Your video has been analyzed. View results below.
          </p>
          <button
            onClick={handleReset}
            className="mt-4 px-4 py-2 border border-gray-300 rounded hover:bg-gray-100 transition"
          >
            Upload Another Video
          </button>
        </div>
      )}

      {analysisStatus === 'error' && error && (
        <div className="text-center py-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
            <svg
              className="w-8 h-8 text-red-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </div>
          <p className="text-lg font-semibold text-gray-900">Analysis Failed</p>
          <p className="text-sm text-red-600 mt-2">{error}</p>
          <button
            onClick={handleReset}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  )
}

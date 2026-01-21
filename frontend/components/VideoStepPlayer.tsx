'use client'

import { useState, useRef, useEffect } from 'react'
import { api, AnalysisResults, AssemblyEvent } from '@/lib/api/client'

interface VideoStepPlayerProps {
  videoUrl: string
  analysisResults: AnalysisResults
  manualId: string
  onOverlayGenerated?: (overlayId: string) => void
}

export default function VideoStepPlayer({
  videoUrl,
  analysisResults,
  manualId,
  onOverlayGenerated,
}: VideoStepPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [activeStep, setActiveStep] = useState<AssemblyEvent | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [showOverlay, setShowOverlay] = useState(false)
  const [overlayId, setOverlayId] = useState<string | null>(null)
  const [overlayGenerating, setOverlayGenerating] = useState(false)

  const events = analysisResults.results.detected_events || []
  const totalDuration = analysisResults.results.total_duration_sec || 0
  const totalSteps = analysisResults.results.total_steps_detected || 0

  // Update active step based on current time
  useEffect(() => {
    const event = events.find(
      (e) => e.start_seconds <= currentTime && currentTime <= e.end_seconds
    )
    setActiveStep(event || null)
  }, [currentTime, events])

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const seekToStep = (event: AssemblyEvent) => {
    if (videoRef.current) {
      videoRef.current.currentTime = event.anchor_timestamp
    }
  }

  const handleGenerateOverlay = async () => {
    if (overlayGenerating) return

    try {
      setOverlayGenerating(true)

      const response = await api.generateOverlay(analysisResults.analysis_id, {
        show_target_marker: true,
        show_hud_panel: true,
        show_instruction_card: true,
        show_debug_grid: false,
      })

      setOverlayId(response.overlay_id)

      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const status = await api.getAnalysisStatus(response.overlay_id)

          if (status.status === 'completed') {
            clearInterval(pollInterval)
            setOverlayGenerating(false)
            setShowOverlay(true)
            if (onOverlayGenerated) {
              onOverlayGenerated(response.overlay_id)
            }
          } else if (status.status === 'error') {
            clearInterval(pollInterval)
            setOverlayGenerating(false)
            alert('Overlay generation failed. Please try again.')
          }
        } catch (err) {
          console.error('Error polling overlay status:', err)
        }
      }, 5000)
    } catch (err: any) {
      console.error('Error generating overlay:', err)
      alert(err.response?.data?.detail || 'Failed to generate overlay')
      setOverlayGenerating(false)
    }
  }

  const downloadOverlay = () => {
    if (overlayId) {
      const url = api.getOverlayVideoUrl(overlayId)
      window.open(url, '_blank')
    }
  }

  return (
    <div className="video-player-container bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold">Video Analysis Results</h3>
        <div className="flex gap-2">
          {!showOverlay && !overlayId && (
            <button
              onClick={handleGenerateOverlay}
              disabled={overlayGenerating}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition disabled:bg-gray-400"
            >
              {overlayGenerating ? 'Generating Overlay...' : 'Generate Visual Overlay'}
            </button>
          )}
          {overlayId && !showOverlay && (
            <button
              onClick={() => setShowOverlay(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              View Overlay Video
            </button>
          )}
          {showOverlay && (
            <button
              onClick={() => setShowOverlay(false)}
              className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-100 transition"
            >
              View Original Video
            </button>
          )}
          {overlayId && (
            <button
              onClick={downloadOverlay}
              className="px-4 py-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 transition"
            >
              Download Overlay
            </button>
          )}
        </div>
      </div>

      {/* Video Player */}
      <div className="relative bg-black rounded-lg overflow-hidden mb-4">
        <video
          ref={videoRef}
          src={showOverlay && overlayId ? api.getOverlayVideoUrl(overlayId) : videoUrl}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          className="w-full"
          controls
        />
      </div>

      {/* Timeline with Step Markers */}
      <div className="step-timeline mb-4">
        <div className="relative w-full h-12 bg-gray-200 rounded overflow-hidden">
          {/* Step segments */}
          {events.map((event, idx) => {
            const startPercent = (event.start_seconds / totalDuration) * 100
            const widthPercent =
              ((event.end_seconds - event.start_seconds) / totalDuration) * 100
            const isActive = activeStep?.step_id === event.step_id

            const colors = ['bg-blue-400', 'bg-green-400', 'bg-yellow-400', 'bg-purple-400']
            const color = colors[idx % colors.length]

            return (
              <div
                key={event.step_id}
                className={`absolute top-0 h-full ${
                  isActive ? 'opacity-100 ring-2 ring-blue-600' : 'opacity-70'
                } ${color} cursor-pointer hover:opacity-100 transition-opacity flex items-center justify-center`}
                style={{
                  left: `${startPercent}%`,
                  width: `${widthPercent}%`,
                }}
                onClick={() => seekToStep(event)}
                title={`Step ${event.step_id}: ${event.instruction}`}
              >
                <span className="text-white text-xs font-bold">{event.step_id}</span>
              </div>
            )
          })}

          {/* Current time indicator */}
          <div
            className="absolute top-0 w-1 h-full bg-red-600 pointer-events-none"
            style={{
              left: `${(currentTime / totalDuration) * 100}%`,
            }}
          />
        </div>

        <div className="flex justify-between mt-2 text-xs text-gray-600">
          <span>0:00</span>
          <span>{Math.floor(totalDuration / 60)}:{String(Math.floor(totalDuration % 60)).padStart(2, '0')}</span>
        </div>
      </div>

      {/* Active Step Info Card */}
      {activeStep && (
        <div className="active-step-card p-4 bg-blue-50 rounded-lg border border-blue-200 mb-4">
          <h4 className="font-bold text-lg mb-2">Current Step: {activeStep.step_id}</h4>
          <p className="text-gray-700 mb-3">{activeStep.instruction}</p>

          <div className="flex flex-wrap gap-2 mb-3">
            <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
              Confidence: {(activeStep.confidence * 100).toFixed(0)}%
            </span>
            <span className="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded">
              {activeStep.start_seconds.toFixed(1)}s - {activeStep.end_seconds.toFixed(1)}s
            </span>
          </div>

          {activeStep.parts_required && activeStep.parts_required.length > 0 && (
            <div className="mt-3">
              <p className="text-sm font-semibold text-gray-700 mb-1">Parts Required:</p>
              <ul className="text-sm text-gray-600">
                {activeStep.parts_required.map((part, idx) => (
                  <li key={idx}>
                    {part.quantity}x {part.color} {part.description}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {activeStep.reference_image && (
            <div className="mt-3">
              <p className="text-xs text-gray-600 mb-1">Reference from manual:</p>
              <img
                src={`/api/image?path=${activeStep.reference_image}`}
                alt={`Step ${activeStep.step_id} reference`}
                className="w-full max-w-md rounded border"
              />
            </div>
          )}
        </div>
      )}

      {/* Step List Summary */}
      <div className="steps-summary">
        <h4 className="font-semibold mb-3">
          Detected Steps ({totalSteps})
        </h4>

        <div className="grid gap-2">
          {events.map((event) => (
            <div
              key={event.step_id}
              className={`step-item p-3 rounded cursor-pointer transition ${
                activeStep?.step_id === event.step_id
                  ? 'bg-blue-100 border-2 border-blue-400'
                  : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
              }`}
              onClick={() => seekToStep(event)}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <span className="font-medium text-gray-900">Step {event.step_id}</span>
                  <p className="text-sm text-gray-700 mt-1">{event.instruction}</p>
                </div>
                <span className="text-xs text-gray-500 ml-2">
                  {event.start_seconds.toFixed(1)}s
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Analysis Summary */}
      <div className="mt-6 p-4 bg-gray-50 rounded border">
        <h4 className="font-semibold mb-2">Analysis Summary</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-600">Total Steps Detected:</span>
            <span className="ml-2 font-semibold">{totalSteps}</span>
          </div>
          <div>
            <span className="text-gray-600">Coverage:</span>
            <span className="ml-2 font-semibold">
              {analysisResults.results.coverage_percentage}%
            </span>
          </div>
          <div>
            <span className="text-gray-600">Avg Confidence:</span>
            <span className="ml-2 font-semibold">
              {((analysisResults.results.average_confidence || 0) * 100).toFixed(0)}%
            </span>
          </div>
          <div>
            <span className="text-gray-600">Processing Time:</span>
            <span className="ml-2 font-semibold">
              {analysisResults.processing_time_sec?.toFixed(1)}s
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

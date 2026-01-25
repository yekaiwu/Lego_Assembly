import axios from 'axios'

// Get API URL and ensure no trailing slash
const getApiUrl = () => {
  const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  // Remove trailing slash if present
  return url.replace(/\/$/, '')
}

const API_URL = getApiUrl()

// Log API URL in development for debugging
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  console.log('API URL:', API_URL)
}

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000, // 300 second timeout (VLM analysis can be slow)
})

// Add request interceptor for debugging
apiClient.interceptors.request.use(
  (config) => {
    if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
      console.log('API Request:', config.method?.toUpperCase(), config.url)
    }
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error Response:', {
        status: error.response.status,
        statusText: error.response.statusText,
        url: error.config?.url,
        baseURL: error.config?.baseURL,
        data: error.response.data,
      })
    } else if (error.request) {
      // Request was made but no response received
      console.error('API No Response:', {
        url: error.config?.url,
        baseURL: error.config?.baseURL,
        message: error.message,
      })
    } else {
      // Something else happened
      console.error('API Error:', error.message)
    }
    return Promise.reject(error)
  }
)

// API Types
export interface Manual {
  manual_id: string
  total_steps: number
  generated_at: string
  status: string
}

export interface ImageAnalysisResult {
  detected_parts: DetectedPart[]
  confidence: number
  matched_node_ids: string[]
  unmatched_parts: DetectedPart[]
}

export interface QueryResponse {
  answer: string
  sources: RetrievalResult[]
  current_step?: number
  next_step?: number
  guidance?: string
  parts_needed?: PartInfo[]
  image_analysis?: ImageAnalysisResult
}

export interface RetrievalResult {
  step_number: number
  content: string
  similarity_score: number
  metadata: Record<string, any>
  image_path?: string
}

export interface PartInfo {
  part_id?: string
  description: string
  color: string
  shape?: string
  quantity: number
  position?: { x: number; y: number; z: number }
}

export interface StepDetails {
  step_number: number
  has_parts: boolean
  parts_count: number
  has_dependencies: boolean
  image_path: string
  preview: string
}

// Vision Analysis Types
export interface DetectedPart {
  description: string
  color: string
  shape?: string
  part_id?: string
  quantity: number
  location?: string
  confidence: number
}

export interface AssembledStructure {
  description: string
  size?: string
  completeness?: string
}

export interface PartConnection {
  part_a: string
  part_b: string
  connection_type: string
  orientation?: string
}

export interface SpatialLayout {
  overall_shape?: string
  front_view?: string
  top_view?: string
  complexity?: string
}

export interface AssemblyError {
  error_type: string
  severity: string
  message: string
  suggested_fix?: string
}

export interface StateAnalysisResponse {
  detected_parts: DetectedPart[]
  assembled_structures: AssembledStructure[]
  connections: PartConnection[]
  spatial_layout: SpatialLayout
  detection_confidence: number
  completed_steps: number[]
  current_step: number
  progress_percentage: number
  total_steps: number
  instruction: string
  next_step_number?: number
  parts_needed: PartInfo[]
  reference_image?: string
  errors: AssemblyError[]
  error_corrections: string[]
  missing_parts: any[]
  encouragement: string
  confidence: number
  status: string
}

export interface ImageUploadResponse {
  uploaded_files: string[]
  session_id: string
  message: string
  status: string
}

// Video Analysis Types
export interface VideoUploadResponse {
  video_id: string
  filename: string
  size_mb: number
  duration_sec: number
  fps: number
  resolution: number[]
  status: string
}

export interface VideoAnalysisResponse {
  analysis_id: string
  status: string
  estimated_time_sec: number
  message: string
}

export interface AssemblyEvent {
  step_id: number
  step_number?: number
  start_seconds: number
  end_seconds: number
  anchor_timestamp: number
  instruction: string
  action: string
  target_box_2d?: number[]
  assembly_box_2d?: number[]
  confidence: number
  reasoning?: string
  parts_required?: PartInfo[]
  reference_image?: string
}

export interface AnalysisResults {
  analysis_id: string
  status: string
  results: {
    manual_id?: string
    video_id?: string
    total_duration_sec?: number
    fps?: number
    resolution?: number[]
    detected_events?: AssemblyEvent[]
    total_steps_detected?: number
    expected_steps?: number
    coverage_percentage?: number
    average_confidence?: number
    analysis_timestamp?: string
    model_used?: string
    progress_percentage?: number
    current_step?: string
  }
  processing_time_sec?: number
}

export interface OverlayOptions {
  show_target_marker?: boolean
  show_hud_panel?: boolean
  show_instruction_card?: boolean
  show_debug_grid?: boolean
}

export interface OverlayGenerationResponse {
  overlay_id: string
  status: string
  estimated_time_sec: number
}

export interface ActiveStepResponse {
  timestamp_sec: number
  active_step?: AssemblyEvent
  message?: string
}

// API Functions
export const api = {
  // Health check
  async health() {
    const { data } = await apiClient.get('/health')
    return data
  },

  // List all manuals
  async listManuals() {
    const { data } = await apiClient.get<{ manuals: Manual[]; total: number }>(
      '/api/manuals'
    )
    return data
  },

  // Get steps for a manual
  async getManualSteps(manualId: string) {
    const { data } = await apiClient.get<{
      manual_id: string
      total_steps: number
      steps: StepDetails[]
    }>(`/api/manual/${manualId}/steps`)
    return data
  },

  // Get extracted step details (raw data from Phase 1)
  async getExtractedSteps(
    manualId: string,
    limit?: number,
    stepNumber?: number
  ) {
    const params: any = {}
    if (limit) params.limit = limit
    if (stepNumber) params.step_number = stepNumber

    const { data } = await apiClient.get<{
      manual_id: string
      total_steps: number
      returned_steps: number
      steps: any[]
    }>(`/api/manual/${manualId}/extracted-steps`, { params })
    return data
  },

  // Query text
  async queryText(
    manualId: string,
    question: string,
    includeImages: boolean = true,
    maxResults: number = 5,
    sessionId?: string
  ) {
    const { data } = await apiClient.post<QueryResponse>('/api/query/text', {
      manual_id: manualId,
      question,
      include_images: includeImages,
      max_results: maxResults,
      session_id: sessionId,
    })
    return data
  },

  // Query with images (multimodal)
  async queryMultimodal(
    manualId: string,
    question: string,
    sessionId: string,
    includeImages: boolean = true,
    maxResults: number = 5
  ) {
    const { data } = await apiClient.post<QueryResponse>('/api/query/multimodal', {
      manual_id: manualId,
      question,
      session_id: sessionId,
      include_images: includeImages,
      max_results: maxResults,
    })
    return data
  },

  // Get specific step
  async getStep(manualId: string, stepNumber: number) {
    const { data } = await apiClient.get<QueryResponse>(
      `/api/manual/${manualId}/step/${stepNumber}`
    )
    return data
  },

  // Get hierarchical graph
  async getGraph(manualId: string) {
    const { data } = await apiClient.get(`/api/manual/${manualId}/graph`)
    return data
  },

  // Get graph summary
  async getGraphSummary(manualId: string) {
    const { data } = await apiClient.get(`/api/manual/${manualId}/graph/summary`)
    return data
  },

  // Get image URL
  getImageUrl(imagePath: string, manualId: string): string {
    return `${API_URL}/api/image?path=${encodeURIComponent(imagePath)}&manual_id=${manualId}`
  },

  // Ingest manual
  async ingestManual(manualId: string) {
    const { data } = await apiClient.post(`/api/ingest/manual/${manualId}`)
    return data
  },

  // Vision Analysis: Upload images
  async uploadImages(images: File[]) {
    const formData = new FormData()
    images.forEach((image) => {
      formData.append('images', image)
    })

    const { data } = await apiClient.post<ImageUploadResponse>(
      '/api/vision/upload-images',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return data
  },

  // Vision Analysis: Analyze assembly state
  async analyzeAssemblyState(
    manualId: string,
    sessionId: string,
    outputDir: string = './output'
  ) {
    const { data } = await apiClient.post<StateAnalysisResponse>(
      '/api/vision/analyze',
      null,
      {
        params: {
          manual_id: manualId,
          session_id: sessionId,
          output_dir: outputDir,
        },
      }
    )
    return data
  },

  // Vision Analysis: Cleanup session
  async cleanupSession(sessionId: string) {
    const { data } = await apiClient.delete(`/api/vision/session/${sessionId}`)
    return data
  },

  // Video Analysis: Upload video
  async uploadVideo(manualId: string, videoFile: File) {
    const formData = new FormData()
    formData.append('manual_id', manualId)
    formData.append('video', videoFile)

    const { data } = await apiClient.post<VideoUploadResponse>(
      '/api/video/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return data
  },

  // Video Analysis: Start analysis
  async analyzeVideo(manualId: string, videoId: string) {
    const { data } = await apiClient.post<VideoAnalysisResponse>(
      '/api/video/analyze',
      null,
      {
        params: {
          manual_id: manualId,
          video_id: videoId,
        },
      }
    )
    return data
  },

  // Video Analysis: Get analysis status
  async getAnalysisStatus(analysisId: string) {
    const { data } = await apiClient.get<AnalysisResults>(
      `/api/video/analysis/${analysisId}`
    )
    return data
  },

  // Video Analysis: Generate overlay
  async generateOverlay(analysisId: string, options?: OverlayOptions) {
    const { data } = await apiClient.post<OverlayGenerationResponse>(
      '/api/video/overlay',
      options || {},
      {
        params: {
          analysis_id: analysisId,
        },
      }
    )
    return data
  },

  // Video Analysis: Get step at timestamp
  async getStepAtTimestamp(analysisId: string, timestampSec: number) {
    const { data } = await apiClient.get<ActiveStepResponse>(
      '/api/video/step-at-time',
      {
        params: {
          analysis_id: analysisId,
          timestamp_sec: timestampSec,
        },
      }
    )
    return data
  },

  // Video Analysis: Download overlay video
  getOverlayVideoUrl(overlayId: string) {
    return `${API_URL}/api/video/download/${overlayId}`
  },
}


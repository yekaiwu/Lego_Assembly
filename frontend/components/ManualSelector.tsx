'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { useManualStore } from '@/lib/store/manualStore'
import { ChevronDown, Book, Loader2 } from 'lucide-react'

export default function ManualSelector() {
  const { selectedManual, setSelectedManual, setTotalSteps } = useManualStore()

  const { data, isLoading, error } = useQuery({
    queryKey: ['manuals'],
    queryFn: api.listManuals,
  })

  const handleManualSelect = async (manualId: string) => {
    setSelectedManual(manualId)

    // Fetch steps to get total count
    try {
      const stepsData = await api.getManualSteps(manualId)
      setTotalSteps(stepsData.total_steps)
    } catch (err) {
      console.error('Error fetching manual steps:', err)
    }
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-center space-x-2">
          <Loader2 className="w-5 h-5 animate-spin text-lego-blue" />
          <span className="text-gray-600">Loading manuals...</span>
        </div>
      </div>
    )
  }

  if (error) {
    // Check if it's a network/connection error vs API error
    const isNetworkError = error instanceof Error && (
      error.message.includes('fetch') || 
      error.message.includes('network') ||
      error.message.includes('Failed to fetch')
    )
    
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0">
            <span className="text-2xl">⚠️</span>
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-red-800 mb-2">
              Failed to connect to backend
            </h3>
            <p className="text-red-700 mb-3">
              {isNetworkError 
                ? "Cannot reach the backend server. Please check:"
                : "Error loading manuals. Please check:"}
            </p>
            <ul className="list-disc list-inside text-sm text-red-600 space-y-1 mb-3">
              <li>Backend server is running and accessible</li>
              <li>API URL is correctly configured</li>
              <li>Network connection is working</li>
            </ul>
            <details className="mt-3">
              <summary className="text-sm text-red-600 cursor-pointer hover:text-red-800">
                Technical details
              </summary>
              <pre className="mt-2 p-2 bg-red-100 rounded text-xs text-red-800 overflow-auto">
                {error instanceof Error ? error.message : String(error)}
              </pre>
            </details>
          </div>
        </div>
      </div>
    )
  }

  const manuals = data?.manuals || []

  if (manuals.length === 0) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0">
            <span className="text-2xl">📦</span>
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-blue-800 mb-2">
              No manuals available yet
            </h3>
            <p className="text-blue-700 mb-4">
              You need to upload and ingest a manual first. Here's how:
            </p>
            
            <div className="bg-white rounded-lg p-4 mb-4 border border-blue-200">
              <h4 className="font-medium text-blue-900 mb-2">Step 1: Process Manual Locally</h4>
              <p className="text-sm text-blue-700 mb-2">
                Run Phase 1 processing on your local machine:
              </p>
              <code className="block bg-blue-50 px-3 py-2 rounded text-sm text-blue-900 mb-2">
                uv run python main.py &lt;pdf_url&gt;
              </code>
              <p className="text-xs text-blue-600">
                This creates files in <code>./output/</code> directory
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 mb-4 border border-blue-200">
              <h4 className="font-medium text-blue-900 mb-2">Step 2: Upload to Railway</h4>
              <p className="text-sm text-blue-700 mb-2">
                Upload the processed files to Railway backend:
              </p>
              <code className="block bg-blue-50 px-3 py-2 rounded text-sm text-blue-900 mb-2">
                python scripts/upload_manual_to_railway.py &lt;manual_id&gt; &lt;railway_url&gt;
              </code>
              <p className="text-xs text-blue-600">
                Example: <code>python scripts/upload_manual_to_railway.py 6262059 https://your-app.railway.app</code>
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 border border-blue-200">
              <h4 className="font-medium text-blue-900 mb-2">Step 3: Ingest into Vector Store</h4>
              <p className="text-sm text-blue-700 mb-2">
                After uploading, trigger ingestion via API:
              </p>
              <code className="block bg-blue-50 px-3 py-2 rounded text-sm text-blue-900">
                curl -X POST {process.env.NEXT_PUBLIC_API_URL || 'https://your-backend.railway.app'}/api/ingest/manual/&lt;manual_id&gt;
              </code>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            📦 Select Manual
          </label>
          <div className="relative">
            <select
              value={selectedManual || ''}
              onChange={(e) => handleManualSelect(e.target.value)}
              className="block w-full px-4 py-3 pr-10 text-base border border-gray-300 focus:outline-none focus:ring-2 focus:ring-lego-blue focus:border-lego-blue rounded-lg appearance-none bg-white"
            >
              <option value="">Select a manual...</option>
              {manuals.map((manual) => (
                <option key={manual.manual_id} value={manual.manual_id}>
                  Manual {manual.manual_id} ({manual.total_steps} steps)
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-400">
              <ChevronDown className="w-5 h-5" />
            </div>
          </div>
        </div>

        {selectedManual && (
          <div className="ml-6">
            <div className="bg-lego-blue text-white px-4 py-3 rounded-lg flex items-center space-x-2">
              <Book className="w-5 h-5" />
              <div className="text-sm">
                <div className="font-semibold">{selectedManual}</div>
                <div className="text-blue-100">Ready to build</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Recently Used - could be enhanced with localStorage */}
      {manuals.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500 mb-2">Available Manuals:</p>
          <div className="flex flex-wrap gap-2">
            {manuals.slice(0, 5).map((manual) => (
              <button
                key={manual.manual_id}
                onClick={() => handleManualSelect(manual.manual_id)}
                className={`text-sm px-3 py-1 rounded-full transition-colors ${
                  selectedManual === manual.manual_id
                    ? 'bg-lego-blue text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {manual.manual_id}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}





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
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <p className="text-red-600">
          ⚠️ Failed to load manuals. Make sure the backend server is running.
        </p>
      </div>
    )
  }

  const manuals = data?.manuals || []

  if (manuals.length === 0) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <p className="text-yellow-800">
          📦 No manuals available. Please ingest manuals using the backend first.
        </p>
        <p className="text-sm text-yellow-600 mt-2">
          Run: <code className="bg-yellow-100 px-2 py-1 rounded">python -m app.scripts.ingest_manual &lt;manual_id&gt;</code>
        </p>
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





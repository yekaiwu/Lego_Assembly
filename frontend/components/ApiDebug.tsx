'use client'

import { useEffect, useState } from 'react'
import { api } from '@/lib/api/client'

export default function ApiDebug() {
  const [apiUrl, setApiUrl] = useState<string>('')
  const [healthStatus, setHealthStatus] = useState<string>('checking...')
  const [error, setError] = useState<string>('')

  useEffect(() => {
    // Get the API URL from environment
    const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    setApiUrl(url)

    // Test health endpoint
    api.health()
      .then((data) => {
        setHealthStatus(`✅ Connected: ${JSON.stringify(data)}`)
        setError('')
      })
      .catch((err: any) => {
        setHealthStatus('❌ Failed')
        setError(err.message || JSON.stringify(err))
      })
  }, [])

  return (
    <div className="bg-gray-100 border border-gray-300 rounded-lg p-4 text-sm font-mono">
      <div className="space-y-2">
        <div>
          <strong>API URL:</strong> {apiUrl || 'NOT SET'}
        </div>
        <div>
          <strong>Health Check:</strong> {healthStatus}
        </div>
        {error && (
          <div className="text-red-600">
            <strong>Error:</strong> {error}
          </div>
        )}
        <div className="text-xs text-gray-500 mt-2">
          Environment: {process.env.NODE_ENV}
        </div>
      </div>
    </div>
  )
}

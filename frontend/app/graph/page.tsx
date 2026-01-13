import GraphVisualization from '@/components/GraphVisualization'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function GraphPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md border-b-4 border-lego-red">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-lego-red rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">🧱</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-900">
                Assembly Graph Visualization
              </h1>
            </div>

            <Link
              href="/"
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors font-medium"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Main</span>
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        <GraphVisualization />
      </div>

      {/* Footer */}
      <footer className="mt-12 py-6 bg-white border-t">
        <div className="max-w-7xl mx-auto px-4 text-center text-gray-600 text-sm">
          <p>LEGO Builder Assistant • Hierarchical Graph Visualization</p>
        </div>
      </footer>
    </div>
  )
}

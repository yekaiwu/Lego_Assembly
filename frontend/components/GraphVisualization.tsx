'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { useManualStore } from '@/lib/store/manualStore'
import { Loader2, Layers, Box, Component } from 'lucide-react'
import { useState } from 'react'

interface GraphNode {
  node_id: string
  type: 'model' | 'subassembly' | 'part'
  name: string
  description: string
  color?: string
  shape?: string
  role?: string
  children: string[]
  parents: string[]
  step_created: number
  layer: number
}

interface GraphData {
  manual_id: string
  metadata: {
    total_parts: number
    total_subassemblies: number
    total_steps: number
    max_depth: number
    generated_at: string
  }
  nodes: GraphNode[]
  edges: Array<{
    source: string
    target: string
    relationship: string
  }>
}

export default function GraphVisualization() {
  const { selectedManual } = useManualStore()
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)

  const { data: graphData, isLoading } = useQuery<GraphData>({
    queryKey: ['graph', selectedManual],
    queryFn: () => api.getGraph(selectedManual!),
    enabled: !!selectedManual,
  })

  if (!selectedManual) {
    return (
      <div className="text-center py-12 text-gray-400">
        <Layers className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p>Select a manual to view its hierarchical graph</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-lego-blue" />
      </div>
    )
  }

  if (!graphData) {
    return (
      <div className="text-center py-12 text-gray-400">
        <p>No graph data available</p>
      </div>
    )
  }

  // Group nodes by layer
  const nodesByLayer = graphData.nodes.reduce((acc, node) => {
    if (!acc[node.layer]) acc[node.layer] = []
    acc[node.layer].push(node)
    return acc
  }, {} as Record<number, GraphNode[]>)

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'model':
        return <Component className="w-4 h-4" />
      case 'subassembly':
        return <Layers className="w-4 h-4" />
      case 'part':
        return <Box className="w-4 h-4" />
      default:
        return <Box className="w-4 h-4" />
    }
  }

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'model':
        return 'bg-purple-100 border-purple-400 text-purple-900'
      case 'subassembly':
        return 'bg-blue-100 border-blue-400 text-blue-900'
      case 'part':
        return 'bg-green-100 border-green-400 text-green-900'
      default:
        return 'bg-gray-100 border-gray-400 text-gray-900'
    }
  }

  return (
    <div className="space-y-6">
      {/* Graph Metadata */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Graph Overview</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-500">Total Parts</p>
            <p className="text-2xl font-bold text-lego-blue">
              {graphData.metadata.total_parts}
            </p>
          </div>
          <div>
            <p className="text-gray-500">Subassemblies</p>
            <p className="text-2xl font-bold text-lego-green">
              {graphData.metadata.total_subassemblies}
            </p>
          </div>
          <div>
            <p className="text-gray-500">Total Steps</p>
            <p className="text-2xl font-bold text-lego-yellow">
              {graphData.metadata.total_steps}
            </p>
          </div>
          <div>
            <p className="text-gray-500">Max Depth</p>
            <p className="text-2xl font-bold text-lego-red">
              {graphData.metadata.max_depth}
            </p>
          </div>
        </div>
      </div>

      {/* Hierarchical View */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">Hierarchical Structure</h2>

        <div className="space-y-6 overflow-x-auto">
          {Object.keys(nodesByLayer)
            .sort((a, b) => Number(a) - Number(b))
            .map((layer) => (
              <div key={layer}>
                <div className="text-xs font-semibold text-gray-500 mb-2">
                  Layer {layer}
                </div>
                <div className="flex flex-wrap gap-2">
                  {nodesByLayer[Number(layer)].map((node) => (
                    <button
                      key={node.node_id}
                      onClick={() => setSelectedNode(node)}
                      className={`flex items-center space-x-2 px-3 py-2 rounded-lg border-2 transition-all hover:shadow-md ${getNodeColor(
                        node.type
                      )} ${
                        selectedNode?.node_id === node.node_id
                          ? 'ring-2 ring-offset-2 ring-blue-500'
                          : ''
                      }`}
                    >
                      {getNodeIcon(node.type)}
                      <div className="text-left">
                        <div className="text-xs font-semibold">{node.name}</div>
                        <div className="text-xs opacity-75">
                          Step {node.step_created}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-4">Node Details</h2>
          <div className="space-y-3">
            <div>
              <p className="text-sm text-gray-500">Node ID</p>
              <p className="font-mono text-sm">{selectedNode.node_id}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Type</p>
              <p className="capitalize">{selectedNode.type}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Name</p>
              <p>{selectedNode.name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Description</p>
              <p>{selectedNode.description}</p>
            </div>
            {selectedNode.color && (
              <div>
                <p className="text-sm text-gray-500">Color</p>
                <p className="capitalize">{selectedNode.color}</p>
              </div>
            )}
            {selectedNode.shape && (
              <div>
                <p className="text-sm text-gray-500">Shape</p>
                <p>{selectedNode.shape}</p>
              </div>
            )}
            <div>
              <p className="text-sm text-gray-500">Step Created</p>
              <p>Step {selectedNode.step_created}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Layer</p>
              <p>Layer {selectedNode.layer}</p>
            </div>
            {selectedNode.parents.length > 0 && (
              <div>
                <p className="text-sm text-gray-500">Parents</p>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.parents.map((parent) => (
                    <span
                      key={parent}
                      className="text-xs bg-gray-100 px-2 py-1 rounded"
                    >
                      {parent}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {selectedNode.children.length > 0 && (
              <div>
                <p className="text-sm text-gray-500">
                  Children ({selectedNode.children.length})
                </p>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.children.map((child) => (
                    <span
                      key={child}
                      className="text-xs bg-gray-100 px-2 py-1 rounded"
                    >
                      {child}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-800">
          💡 <strong>Tip:</strong> For advanced graph visualization with
          interactive layouts, install a library like{' '}
          <code className="bg-blue-100 px-1 rounded">reactflow</code> or{' '}
          <code className="bg-blue-100 px-1 rounded">cytoscape</code>. The graph
          data is available via <code className="bg-blue-100 px-1 rounded">
            GET /api/manual/{'{manual_id}'}/graph
          </code>
        </p>
      </div>
    </div>
  )
}

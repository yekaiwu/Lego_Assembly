'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api/client'
import { useManualStore } from '@/lib/store/manualStore'
import { Loader2, Layers, Box, Component } from 'lucide-react'
import { useState, useMemo, useCallback, useEffect } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
} from 'reactflow'
import 'reactflow/dist/style.css'

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
  image_path?: string
  mask_path?: string
  bounding_box?: number[]
}

interface GraphEdge {
  from: string
  to: string
  type: string
  created_step: number
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
  edges: GraphEdge[]
}

// Transform graph data to React Flow format
function transformToReactFlow(graphData: GraphData) {
  // Group nodes by layer for layout
  const nodesByLayer: Record<number, GraphNode[]> = {}
  graphData.nodes.forEach((node) => {
    if (!nodesByLayer[node.layer]) nodesByLayer[node.layer] = []
    nodesByLayer[node.layer].push(node)
  })

  const layers = Object.keys(nodesByLayer)
    .map(Number)
    .sort((a, b) => a - b)

  // Calculate positions for nodes (hierarchical layout)
  const LAYER_HEIGHT = 200
  const NODE_SPACING = 250
  const nodes: Node[] = []

  layers.forEach((layer) => {
    const layerNodes = nodesByLayer[layer]
    const layerWidth = layerNodes.length * NODE_SPACING
    const startX = -layerWidth / 2

    layerNodes.forEach((node, index) => {
      const x = startX + index * NODE_SPACING
      const y = layer * LAYER_HEIGHT

      // Get node styling based on type
      let bgColor = '#f3f4f6'
      let borderColor = '#9ca3af'
      if (node.type === 'model') {
        bgColor = '#e9d5ff'
        borderColor = '#a855f7'
      } else if (node.type === 'subassembly') {
        bgColor = '#dbeafe'
        borderColor = '#3b82f6'
      } else if (node.type === 'part') {
        bgColor = '#d1fae5'
        borderColor = '#10b981'
      }

      nodes.push({
        id: node.node_id,
        type: 'default',
        position: { x, y },
        data: {
          label: (
            <div className="text-center">
              <div className="font-semibold text-sm">{node.name}</div>
              <div className="text-xs text-gray-600">Step {node.step_created}</div>
            </div>
          ),
          originalNode: node,
        },
        style: {
          background: bgColor,
          border: `2px solid ${borderColor}`,
          borderRadius: '8px',
          padding: '10px',
          minWidth: '180px',
        },
      })
    })
  })

  // Transform edges
  const edges: Edge[] = graphData.edges.map((edge, index) => {
    // Different styles for different edge types
    const isAttachment = edge.type === 'attachment'
    return {
      id: `edge-${index}`,
      source: edge.from,
      target: edge.to,
      type: 'smoothstep',
      animated: false,
      style: {
        stroke: isAttachment ? '#3b82f6' : '#10b981',
        strokeWidth: 2,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: isAttachment ? '#3b82f6' : '#10b981',
      },
      label: edge.type,
      labelStyle: {
        fontSize: 10,
        fill: '#6b7280',
      },
      labelBgStyle: {
        fill: '#ffffff',
      },
    }
  })

  return { nodes, edges }
}

export default function GraphVisualization() {
  const { selectedManual } = useManualStore()
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)

  const { data: graphData, isLoading } = useQuery<GraphData>({
    queryKey: ['graph', selectedManual],
    queryFn: () => api.getGraph(selectedManual!),
    enabled: !!selectedManual,
  })

  // Transform graph data to React Flow format
  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!graphData) return { nodes: [], edges: [] }
    return transformToReactFlow(graphData)
  }, [graphData])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  // Update nodes and edges when data changes
  useEffect(() => {
    setNodes(initialNodes)
    setEdges(initialEdges)
  }, [initialNodes, initialEdges, setNodes, setEdges])

  // Handle node click
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      const originalNode = node.data.originalNode as GraphNode
      setSelectedNode(originalNode)
    },
    []
  )

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

      {/* Interactive Graph View with Edges */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-4">
          Interactive Graph Visualization
        </h2>

        {/* Legend */}
        <div className="mb-4 flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-purple-200 border-2 border-purple-400 rounded"></div>
            <span>Model</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-200 border-2 border-blue-400 rounded"></div>
            <span>Subassembly</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-200 border-2 border-green-400 rounded"></div>
            <span>Part</span>
          </div>
          <div className="flex items-center gap-2 ml-4">
            <div className="w-8 h-0.5 bg-blue-500"></div>
            <span>Attachment</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-green-500"></div>
            <span>Component</span>
          </div>
        </div>

        <div className="h-[600px] border-2 border-gray-200 rounded-lg">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            nodesDraggable={true}
            nodesConnectable={false}
            elementsSelectable={true}
            fitView
            attributionPosition="bottom-left"
          >
            <Controls />
            <Background />
            <MiniMap
              nodeColor={(node) => {
                const originalNode = node.data.originalNode as GraphNode
                if (originalNode.type === 'model') return '#a855f7'
                if (originalNode.type === 'subassembly') return '#3b82f6'
                return '#10b981'
              }}
            />
          </ReactFlow>
        </div>

        <div className="mt-2 text-sm text-gray-600">
          💡 <strong>Tip:</strong> Use mouse wheel to zoom, drag to pan, click nodes to see details
        </div>
      </div>

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-4">Node Details</h2>

          {/* Image Section */}
          {selectedNode.image_path && (
            <div className="mb-4 pb-4 border-b border-gray-200">
              <p className="text-sm text-gray-500 mb-2">Image</p>
              <div className="flex justify-center">
                <img
                  src={api.getImageUrl(selectedNode.image_path, selectedManual!)}
                  alt={selectedNode.name}
                  className="max-w-full h-auto max-h-64 object-contain border-2 border-gray-200 rounded-lg p-2 bg-gray-50"
                  onError={(e) => {
                    // Hide image on load error
                    const img = e.target as HTMLImageElement
                    img.style.display = 'none'
                    // Optionally show a fallback message
                    const parent = img.parentElement
                    if (parent && !parent.querySelector('.error-message')) {
                      const errorMsg = document.createElement('p')
                      errorMsg.className = 'error-message text-sm text-gray-400 text-center'
                      errorMsg.textContent = 'Image not available'
                      parent.appendChild(errorMsg)
                    }
                  }}
                />
              </div>
            </div>
          )}

          <div className="space-y-3">
            <div>
              <p className="text-sm text-gray-500">Node ID</p>
              <p className="font-mono text-sm text-gray-900">{selectedNode.node_id}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Type</p>
              <p className="capitalize text-gray-900">{selectedNode.type}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Name</p>
              <p className="text-gray-900 font-medium">{selectedNode.name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Description</p>
              <p className="text-gray-900">{selectedNode.description}</p>
            </div>
            {selectedNode.color && (
              <div>
                <p className="text-sm text-gray-500">Color</p>
                <p className="capitalize text-gray-900">{selectedNode.color}</p>
              </div>
            )}
            {selectedNode.shape && (
              <div>
                <p className="text-sm text-gray-500">Shape</p>
                <p className="text-gray-900">{selectedNode.shape}</p>
              </div>
            )}
            <div>
              <p className="text-sm text-gray-500">Step Created</p>
              <p className="text-gray-900">Step {selectedNode.step_created}</p>
            </div>
            {selectedNode.parents.length > 0 && (
              <div>
                <p className="text-sm text-gray-500">Parents</p>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.parents.map((parent) => (
                    <span
                      key={parent}
                      className="text-xs bg-gray-100 text-gray-900 px-2 py-1 rounded"
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
                      className="text-xs bg-gray-100 text-gray-900 px-2 py-1 rounded"
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

      {/* Graph Statistics */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-800">
          📊 <strong>Graph Stats:</strong> Displaying {graphData.nodes.length} nodes
          and {graphData.edges.length} edges across {graphData.metadata.max_depth + 1} layers
        </p>
      </div>
    </div>
  )
}

import GraphVisualization from '@/components/GraphVisualization'

export default function GraphPage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Assembly Graph</h1>
      <GraphVisualization />
    </div>
  )
}

'use client'

import { useState, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api, ImageAnalysisResult } from '@/lib/api/client'
import { useManualStore } from '@/lib/store/manualStore'
import { Send, Loader2, Bot, User, HelpCircle, Image as ImageIcon, X, CheckCircle2, Package } from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  images?: string[] // Preview URLs for uploaded images
  imageAnalysis?: ImageAnalysisResult // VLM analysis results
}

export default function ChatInterface() {
  const { selectedManual } = useManualStore()
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      role: 'assistant',
      content: "Hi! I'm your LEGO assembly assistant. Ask me anything about your build! You can also upload photos of your assembly for context-aware help.",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [uploadedImages, setUploadedImages] = useState<File[]>([])
  const [imagePreviews, setImagePreviews] = useState<string[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  // Convert File to base64 data URL for permanent storage in messages
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Handle clipboard paste for images
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return

      const imageFiles: File[] = []
      
      // Check each clipboard item
      for (let i = 0; i < items.length; i++) {
        const item = items[i]
        
        // Check if item is an image
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) {
            imageFiles.push(file)
          }
        }
      }

      // Add pasted images if any found
      if (imageFiles.length > 0 && uploadedImages.length < 4) {
        const availableSlots = 4 - uploadedImages.length
        const newImages = imageFiles.slice(0, availableSlots)
        
        // Create previews
        const newPreviews = newImages.map(file => URL.createObjectURL(file))
        
        setUploadedImages(prev => [...prev, ...newImages])
        setImagePreviews(prev => [...prev, ...newPreviews])
        setSessionId(null)

        // Show feedback
        console.log(`Pasted ${newImages.length} image(s) from clipboard`)

        // Prevent default paste behavior
        e.preventDefault()
      }
    }

    // Add event listener
    window.addEventListener('paste', handlePaste)

    // Cleanup
    return () => {
      window.removeEventListener('paste', handlePaste)
    }
  }, [uploadedImages])

  // Upload images mutation
  const uploadImagesMutation = useMutation({
    mutationFn: (images: File[]) => api.uploadImages(images),
    onSuccess: (data) => {
      setSessionId(data.session_id)
      console.log('Images uploaded successfully:', data.session_id)
    },
    onError: (error) => {
      console.error('Image upload failed:', error)
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: "Failed to upload images. Please try again.",
          timestamp: new Date(),
        },
      ])
    },
  })

  const queryMutation = useMutation({
    mutationFn: (question: string) => {
      // If images were uploaded, use multimodal query
      if (sessionId) {
        return api.queryMultimodal(selectedManual!, question, sessionId, true, 5)
      }
      // Otherwise use regular text query
      return api.queryText(selectedManual!, question, true, 5)
    },
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: data.answer,
          timestamp: new Date(),
          imageAnalysis: data.image_analysis,  // Include VLM analysis if present
        },
      ])
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: "Sorry, I encountered an error. Please try again.",
          timestamp: new Date(),
        },
      ])
    },
  })

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    addImages(files)

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const addImages = (files: File[]) => {
    // Filter for images only
    const imageFiles = files.filter(file => file.type.startsWith('image/'))
    if (imageFiles.length === 0) return

    // Limit to 4 images total
    const availableSlots = 4 - uploadedImages.length
    const newImages = imageFiles.slice(0, availableSlots)
    
    // Create previews
    const newPreviews = newImages.map(file => URL.createObjectURL(file))
    
    setUploadedImages(prev => [...prev, ...newImages])
    setImagePreviews(prev => [...prev, ...newPreviews])

    // Reset session ID when new images are added
    setSessionId(null)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (uploadedImages.length < 4) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    addImages(files)
  }

  const handleRemoveImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index))
    setImagePreviews(prev => {
      // Revoke old URL to free memory
      URL.revokeObjectURL(prev[index])
      return prev.filter((_, i) => i !== index)
    })
    // Reset session ID when images change
    setSessionId(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim() || !selectedManual) return

    // Upload images first if present and not already uploaded
    if (uploadedImages.length > 0 && !sessionId) {
      try {
        await uploadImagesMutation.mutateAsync(uploadedImages)
      } catch (error) {
        console.error('Failed to upload images:', error)
        return
      }
    }

    // Convert images to base64 for permanent storage in messages
    let messageImages: string[] | undefined = undefined
    if (uploadedImages.length > 0) {
      try {
        messageImages = await Promise.all(uploadedImages.map(file => fileToBase64(file)))
      } catch (error) {
        console.error('Failed to convert images to base64:', error)
        // Fallback to blob URLs if conversion fails
        messageImages = imagePreviews
      }
    }

    // Add user message with permanent base64 image URLs
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      images: messageImages,
    }
    setMessages((prev) => [...prev, userMessage])
    setInput('')

    // Clear images after sending (keep session for follow-up questions)
    setUploadedImages([])
    imagePreviews.forEach(url => URL.revokeObjectURL(url))
    setImagePreviews([])

    // Query the API
    queryMutation.mutate(userMessage.content)
  }

  const quickQuestions = uploadedImages.length > 0 ? [
    "What's next?",
    "Am I doing this right?",
    "What should I do?",
    "Is this correct?",
  ] : [
    "What's the next step?",
    "What parts do I need?",
    "How do I attach this piece?",
    "Show me step dependencies",
  ]

  const handleQuickQuestion = (question: string) => {
    setInput(question)
  }

  if (!selectedManual) {
    return (
      <div className="h-[600px] flex items-center justify-center text-gray-400">
        <div className="text-center">
          <HelpCircle className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p>Select a manual to start chatting</p>
        </div>
      </div>
    )
  }

  return (
    <div 
      className="flex flex-col h-[600px] relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && uploadedImages.length < 4 && (
        <div className="absolute inset-0 z-50 bg-lego-blue bg-opacity-10 border-4 border-dashed border-lego-blue rounded-lg flex items-center justify-center">
          <div className="text-center">
            <ImageIcon className="w-16 h-16 mx-auto mb-4 text-lego-blue" />
            <p className="text-lg font-semibold text-lego-blue">Drop images here</p>
            <p className="text-sm text-gray-600">Up to {4 - uploadedImages.length} more image(s)</p>
          </div>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start space-x-3 message-fade-in ${
              message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            }`}
          >
            {/* Avatar */}
            <div
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                message.role === 'user'
                  ? 'bg-lego-blue'
                  : 'bg-lego-red'
              }`}
            >
              {message.role === 'user' ? (
                <User className="w-5 h-5 text-white" />
              ) : (
                <Bot className="w-5 h-5 text-white" />
              )}
            </div>

            {/* Message Content */}
            <div
              className={`flex-1 ${
                message.role === 'user' ? 'text-right' : 'text-left'
              }`}
            >
              <div
                className={`inline-block max-w-[85%] ${
                  message.role === 'user' ? 'text-left' : ''
                }`}
              >
                {/* Images if present */}
                {message.images && message.images.length > 0 && (
                  <div className="flex gap-2 mb-2 flex-wrap">
                    {message.images.map((imgUrl, idx) => (
                      <img
                        key={idx}
                        src={imgUrl}
                        alt={`Uploaded ${idx + 1}`}
                        className="w-20 h-20 object-cover rounded border-2 border-white"
                      />
                    ))}
                  </div>
                )}
                {/* Message bubble */}
                <div
                  className={`px-4 py-2 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-lego-blue text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </p>
                </div>

                {/* Image Analysis Results (only for assistant messages) */}
                {message.role === 'assistant' && message.imageAnalysis && message.imageAnalysis.detected_parts.length > 0 && (
                  <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Package className="w-4 h-4 text-blue-600" />
                      <h4 className="text-sm font-semibold text-blue-800">Detected Parts</h4>
                      <span className="text-xs text-blue-600 bg-blue-100 px-2 py-0.5 rounded">
                        {message.imageAnalysis.detected_parts.length} parts found
                      </span>
                    </div>
                    <div className="space-y-1">
                      {message.imageAnalysis.detected_parts.map((part, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-xs">
                          <CheckCircle2 className="w-3 h-3 text-green-600 mt-0.5 flex-shrink-0" />
                          <span className="text-gray-700">
                            <span className="font-medium capitalize">{part.color}</span> {part.shape || part.description}
                            {part.quantity > 1 && ` (×${part.quantity})`}
                          </span>
                        </div>
                      ))}
                    </div>
                    {message.imageAnalysis.matched_node_ids.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-blue-200">
                        <p className="text-xs text-blue-700">
                          ✓ Matched {message.imageAnalysis.matched_node_ids.length} part{message.imageAnalysis.matched_node_ids.length !== 1 ? 's' : ''} to assembly graph
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {message.timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </div>
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {queryMutation.isPending && (
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-lego-red flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="bg-gray-100 px-4 py-2 rounded-lg">
              <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Questions */}
      {messages.length === 1 && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Quick questions:</p>
          <div className="flex flex-wrap gap-2">
            {quickQuestions.map((q) => (
              <button
                key={q}
                onClick={() => handleQuickQuestion(q)}
                className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-gray-700 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Image Upload Hint */}
      {uploadedImages.length === 0 && (
        <div className="mb-2 text-center">
          <p className="text-xs text-gray-400">
            💡 Tip: Click <ImageIcon className="inline w-3 h-3" /> to upload, <strong>paste (Ctrl/Cmd+V)</strong>, or <strong>drag & drop</strong> images (1-4 photos)
          </p>
        </div>
      )}

      {/* Image Preview Area */}
      {uploadedImages.length > 0 && (
        <div className="mb-3 p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-gray-600 font-medium">
              {uploadedImages.length} image{uploadedImages.length > 1 ? 's' : ''} attached
              {sessionId && ' (uploaded ✓)'}
            </p>
            {!sessionId && uploadImagesMutation.isPending && (
              <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
            )}
          </div>
          <div className="flex gap-2 flex-wrap">
            {imagePreviews.map((preview, idx) => (
              <div key={idx} className="relative group">
                <img
                  src={preview}
                  alt={`Preview ${idx + 1}`}
                  className="w-16 h-16 object-cover rounded border-2 border-gray-200"
                />
                <button
                  type="button"
                  onClick={() => handleRemoveImage(idx)}
                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleImageSelect}
          accept="image/*"
          multiple
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploadedImages.length >= 4 || queryMutation.isPending}
          className={`px-4 py-3 rounded-lg transition-colors ${
            uploadedImages.length >= 4 || queryMutation.isPending
              ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
              : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
          }`}
          title="Upload images (max 4)"
        >
          <ImageIcon className="w-5 h-5" />
        </button>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={uploadedImages.length > 0 ? "Ask about your assembly photos..." : "Ask about your LEGO build..."}
          className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-lego-blue focus:border-lego-blue text-gray-900 bg-white"
          disabled={queryMutation.isPending || uploadImagesMutation.isPending}
        />
        <button
          type="submit"
          disabled={!input.trim() || queryMutation.isPending || uploadImagesMutation.isPending}
          className={`px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2 ${
            !input.trim() || queryMutation.isPending || uploadImagesMutation.isPending
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-lego-green text-white hover:bg-green-700'
          }`}
        >
          {queryMutation.isPending || uploadImagesMutation.isPending ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </form>
    </div>
  )
}



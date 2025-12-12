'use client'

import { useState, useCallback } from 'react'
import { Upload, X, Camera, AlertCircle } from 'lucide-react'

interface ImageUploadProps {
  onImagesSelected: (images: File[]) => void
  maxImages?: number
  minImages?: number
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onImagesSelected,
  maxImages = 4,
  minImages = 2,
}) => {
  const [selectedImages, setSelectedImages] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [error, setError] = useState<string>('')

  const handleFileSelect = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files || [])
      
      // Validate file types
      const validFiles = files.filter((file) => file.type.startsWith('image/'))
      
      if (validFiles.length !== files.length) {
        setError('Some files were not images and were skipped')
      }

      // Check total count
      const totalImages = selectedImages.length + validFiles.length
      if (totalImages > maxImages) {
        setError(`Maximum ${maxImages} images allowed`)
        return
      }

      // Add new images
      const newImages = [...selectedImages, ...validFiles]
      setSelectedImages(newImages)

      // Generate previews
      const newPreviews = [...previews]
      validFiles.forEach((file) => {
        const reader = new FileReader()
        reader.onloadend = () => {
          newPreviews.push(reader.result as string)
          setPreviews([...newPreviews])
        }
        reader.readAsDataURL(file)
      })

      // Clear error if valid
      if (newImages.length >= minImages && newImages.length <= maxImages) {
        setError('')
        onImagesSelected(newImages)
      }
    },
    [selectedImages, previews, maxImages, minImages, onImagesSelected]
  )

  const removeImage = useCallback(
    (index: number) => {
      const newImages = selectedImages.filter((_, i) => i !== index)
      const newPreviews = previews.filter((_, i) => i !== index)
      
      setSelectedImages(newImages)
      setPreviews(newPreviews)
      
      if (newImages.length < minImages) {
        setError(`Please select at least ${minImages} images`)
      } else {
        setError('')
        onImagesSelected(newImages)
      }
    },
    [selectedImages, previews, minImages, onImagesSelected]
  )

  const clearAll = useCallback(() => {
    setSelectedImages([])
    setPreviews([])
    setError('')
    onImagesSelected([])
  }, [onImagesSelected])

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
        <input
          type="file"
          id="image-upload"
          className="hidden"
          accept="image/*"
          multiple
          onChange={handleFileSelect}
        />
        <label
          htmlFor="image-upload"
          className="cursor-pointer flex flex-col items-center space-y-2"
        >
          <Upload className="w-12 h-12 text-gray-400" />
          <div className="text-sm text-gray-600">
            <span className="text-blue-600 font-medium">Click to upload</span> or
            drag and drop
          </div>
          <div className="text-xs text-gray-500">
            Upload {minImages}-{maxImages} photos of your assembly from different
            angles
          </div>
        </label>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Image Previews */}
      {selectedImages.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-700">
              Selected Images ({selectedImages.length}/{maxImages})
            </h3>
            <button
              onClick={clearAll}
              className="text-sm text-red-600 hover:text-red-700 font-medium"
            >
              Clear All
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {previews.map((preview, index) => (
              <div
                key={index}
                className="relative group rounded-lg overflow-hidden border border-gray-200"
              >
                <img
                  src={preview}
                  alt={`Preview ${index + 1}`}
                  className="w-full h-48 object-cover"
                />
                <button
                  onClick={() => removeImage(index)}
                  className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  aria-label="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-2">
                  {selectedImages[index].name}
                </div>
              </div>
            ))}
          </div>

          {/* Tips */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <Camera className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-medium mb-1">Tips for best results:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li>Take photos from multiple angles (front, back, sides, top)</li>
                  <li>Ensure good lighting and clear focus</li>
                  <li>Show the entire assembly in each photo</li>
                  <li>Avoid shadows and reflections</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}



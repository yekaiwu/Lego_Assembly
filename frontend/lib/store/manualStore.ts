import { create } from 'zustand'

interface ManualState {
  selectedManual: string | null
  currentStep: number | null
  totalSteps: number
  setSelectedManual: (manualId: string) => void
  setCurrentStep: (step: number) => void
  setTotalSteps: (total: number) => void
  clearManual: () => void
  nextStep: () => void
  previousStep: () => void
}

export const useManualStore = create<ManualState>((set) => ({
  selectedManual: null,
  currentStep: null,
  totalSteps: 0,

  setSelectedManual: (manualId) =>
    set({ selectedManual: manualId, currentStep: 1 }),

  setCurrentStep: (step) =>
    set((state) => ({
      currentStep: Math.max(1, Math.min(step, state.totalSteps || step)),
    })),

  setTotalSteps: (total) => set({ totalSteps: total }),

  clearManual: () =>
    set({ selectedManual: null, currentStep: null, totalSteps: 0 }),

  nextStep: () =>
    set((state) => ({
      currentStep: state.currentStep
        ? Math.min(state.currentStep + 1, state.totalSteps)
        : 1,
    })),

  previousStep: () =>
    set((state) => ({
      currentStep: state.currentStep ? Math.max(state.currentStep - 1, 1) : 1,
    })),
}))





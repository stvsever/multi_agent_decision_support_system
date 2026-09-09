/**
 * Client state.
 *
 * Server state lives in React Query. This store holds the things the server
 * does not own: the run being composed, transient interface state, and a local
 * mirror of appearance settings so the theme applies before the first fetch
 * resolves.
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  Accent,
  Density,
  Participant,
  RunOverrides,
  TaskSpecInput,
  Theme,
} from './types'

export interface Toast {
  id: string
  tone: 'info' | 'positive' | 'caution' | 'critical'
  title: string
  body?: string
}

export const emptyTask = (): TaskSpecInput => ({
  prediction_type: 'binary',
  target_label: '',
  control_label: '',
  class_labels: [],
  regression_outputs: [],
  root: null,
})

interface AppearanceMirror {
  theme: Theme
  accent: Accent
  density: Density
  fontScale: number
  reducedMotion: boolean
  numericLocale: string
}

interface AppState {
  /* Appearance mirror, applied to the document root immediately. */
  appearance: AppearanceMirror
  setAppearance: (patch: Partial<AppearanceMirror>) => void

  /* The run being composed in the studio. */
  selected: Participant[]
  setSelected: (participants: Participant[]) => void
  toggleSelected: (participant: Participant) => void
  task: TaskSpecInput
  setTask: (patch: Partial<TaskSpecInput>) => void
  resetTask: () => void
  overrides: RunOverrides
  setOverride: (section: keyof RunOverrides, patch: Record<string, unknown>) => void
  clearOverrides: () => void
  generateDeepReport: boolean
  setGenerateDeepReport: (value: boolean) => void

  /* Interface state. */
  navCollapsed: boolean
  toggleNav: () => void
  settingsOpen: false | string
  openSettings: (section?: string) => void
  closeSettings: () => void
  advancedOpen: Record<string, boolean>
  toggleAdvanced: (key: string) => void
  paletteOpen: boolean
  setPaletteOpen: (open: boolean) => void

  /* Guided tour. */
  tourActive: boolean
  tourStep: number
  startTour: () => void
  endTour: () => void
  setTourStep: (step: number) => void

  /* Run focus. */
  activeRunId: string | null
  setActiveRunId: (id: string | null) => void

  /* Toasts. */
  toasts: Toast[]
  notify: (toast: Omit<Toast, 'id'>) => void
  dismissToast: (id: string) => void
}

let toastSeq = 0

export const useApp = create<AppState>()(
  persist(
    (set, get) => ({
      appearance: {
        theme: 'system',
        accent: 'indigo',
        density: 'comfortable',
        fontScale: 1,
        reducedMotion: false,
        numericLocale: 'en-US',
      },
      setAppearance: (patch) => set((s) => ({ appearance: { ...s.appearance, ...patch } })),

      selected: [],
      setSelected: (participants) => set({ selected: participants }),
      toggleSelected: (participant) =>
        set((s) => {
          const exists = s.selected.some((p) => p.directory === participant.directory)
          return {
            selected: exists
              ? s.selected.filter((p) => p.directory !== participant.directory)
              : [...s.selected, participant],
          }
        }),

      task: emptyTask(),
      setTask: (patch) => set((s) => ({ task: { ...s.task, ...patch } })),
      resetTask: () => set({ task: emptyTask() }),

      overrides: {},
      setOverride: (section, patch) =>
        set((s) => ({
          overrides: {
            ...s.overrides,
            [section]: { ...(s.overrides[section] ?? {}), ...patch },
          },
        })),
      clearOverrides: () => set({ overrides: {} }),

      generateDeepReport: true,
      setGenerateDeepReport: (value) => set({ generateDeepReport: value }),

      navCollapsed: false,
      toggleNav: () => set((s) => ({ navCollapsed: !s.navCollapsed })),

      settingsOpen: false,
      openSettings: (section = 'appearance') => set({ settingsOpen: section }),
      closeSettings: () => set({ settingsOpen: false }),

      advancedOpen: {},
      toggleAdvanced: (key) =>
        set((s) => ({ advancedOpen: { ...s.advancedOpen, [key]: !s.advancedOpen[key] } })),

      paletteOpen: false,
      setPaletteOpen: (open) => set({ paletteOpen: open }),

      tourActive: false,
      tourStep: 0,
      startTour: () => set({ tourActive: true, tourStep: 0 }),
      endTour: () => set({ tourActive: false, tourStep: 0 }),
      setTourStep: (step) => set({ tourStep: step }),

      activeRunId: null,
      setActiveRunId: (id) => set({ activeRunId: id }),

      toasts: [],
      notify: (toast) => {
        const id = `t${++toastSeq}`
        set((s) => ({ toasts: [...s.toasts, { ...toast, id }] }))
        window.setTimeout(() => get().dismissToast(id), toast.tone === 'critical' ? 9000 : 5000)
      },
      dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
    }),
    {
      name: 'compass.ui',
      partialize: (s) => ({
        appearance: s.appearance,
        navCollapsed: s.navCollapsed,
        task: s.task,
        generateDeepReport: s.generateDeepReport,
        advancedOpen: s.advancedOpen,
      }),
    },
  ),
)

/** Reflect appearance onto the document root so CSS tokens resolve. */
export function applyAppearance(appearance: AppearanceMirror): void {
  const root = document.documentElement
  root.dataset.theme = appearance.theme
  root.dataset.accent = appearance.accent
  root.dataset.density = appearance.density
  root.dataset.motion = appearance.reducedMotion ? 'reduced' : 'full'
  root.style.setProperty('--font-scale', String(appearance.fontScale))
}

/**
 * Settings persistence.
 *
 * Every control writes through one controller. A touched value is applied to a
 * local overlay straight away so the interface never lags the pointer, then
 * coalesced into a single PATCH for its section. A rejected save drops the
 * overlay entry, which puts the stored value back on screen, and raises a
 * toast that names what went wrong.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiError } from '@/lib/api'
import { useConfigPatch } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { titleCase } from '@/lib/format'
import type { DashboardConfig } from '@/lib/types'

export type ConfigSection = Extract<
  keyof DashboardConfig,
  | 'connection'
  | 'models'
  | 'engine'
  | 'token_budget'
  | 'local'
  | 'batch'
  | 'cost'
  | 'workspace'
  | 'appearance'
  | 'instructions'
>

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error'

export interface SettingsController {
  config: DashboardConfig
  /** Merge values into a section, then persist. Sub-objects are sent whole. */
  update: <K extends ConfigSection>(section: K, values: Partial<DashboardConfig[K]>) => void
  /** Persist a top-level flag that is not part of a section. */
  updateRoot: (values: Partial<Pick<DashboardConfig, 'onboarding_complete' | 'tour_completed'>>) => Promise<void>
  status: SaveStatus
}

const Ctx = createContext<SettingsController | null>(null)

export const useSettingsController = (): SettingsController => {
  const value = useContext(Ctx)
  if (!value) throw new Error('Settings controls must render inside SettingsProvider.')
  return value
}

type Bucket = Record<string, unknown>

export function SettingsProvider({ config, children }: { config: DashboardConfig; children: ReactNode }) {
  const notify = useApp((s) => s.notify)

  // `useConfigPatch` hands back a fresh function on every render, so it is held
  // in a ref: an effect or callback that depended on it directly would be torn
  // down and rebuilt on each pass.
  const patchConfig = useConfigPatch()
  const patchRef = useRef(patchConfig)
  patchRef.current = patchConfig

  const overlay = useRef<Partial<Record<ConfigSection, Bucket>>>({})
  const pending = useRef<Partial<Record<ConfigSection, Bucket>>>({})
  const timers = useRef<Partial<Record<ConfigSection, number>>>({})
  const [tick, setTick] = useState(0)
  const [status, setStatus] = useState<SaveStatus>('idle')

  const bump = useCallback(() => setTick((n) => n + 1), [])

  const drop = useCallback((section: ConfigSection, bucket: Bucket, onlyIfUnchanged: boolean) => {
    const current = overlay.current[section]
    if (!current) return
    for (const key of Object.keys(bucket)) {
      // A newer edit replaced the reference; that one is still in flight.
      if (!onlyIfUnchanged || Object.is(current[key], bucket[key])) delete current[key]
    }
    if (Object.keys(current).length === 0) delete overlay.current[section]
  }, [])

  const flush = useCallback(
    async (section: ConfigSection) => {
      const bucket = pending.current[section]
      if (!bucket || Object.keys(bucket).length === 0) return
      delete pending.current[section]
      setStatus('saving')
      try {
        await patchRef.current({ [section]: bucket })
        drop(section, bucket, true)
        setStatus('saved')
      } catch (error) {
        drop(section, bucket, false)
        setStatus('error')
        notify({
          tone: 'critical',
          title: `${titleCase(section)} settings were not saved`,
          body:
            error instanceof ApiError
              ? error.message
              : 'The dashboard service rejected the change. The previous value has been restored.',
        })
      }
      bump()
    },
    [drop, notify, bump],
  )

  const update = useCallback(
    <K extends ConfigSection>(section: K, values: Partial<DashboardConfig[K]>) => {
      const nextOverlay = (overlay.current[section] ??= {})
      const nextPending = (pending.current[section] ??= {})
      Object.assign(nextOverlay, values)
      Object.assign(nextPending, values)
      bump()
      window.clearTimeout(timers.current[section])
      // Zero delay still coalesces everything changed in one interaction.
      timers.current[section] = window.setTimeout(() => void flush(section), 0)
    },
    [bump, flush],
  )

  const updateRoot = useCallback(
    async (values: Partial<Pick<DashboardConfig, 'onboarding_complete' | 'tour_completed'>>) => {
      setStatus('saving')
      try {
        await patchRef.current(values)
        setStatus('saved')
      } catch (error) {
        setStatus('error')
        notify({
          tone: 'critical',
          title: 'That setting was not saved',
          body: error instanceof ApiError ? error.message : 'The dashboard service rejected the change.',
        })
      }
    },
    [notify],
  )

  /* Flush anything still waiting when the sheet closes. */
  useEffect(() => {
    const buckets = pending.current
    const pendingTimers = timers.current
    return () => {
      for (const key of Object.keys(buckets) as ConfigSection[]) {
        window.clearTimeout(pendingTimers[key])
        const bucket = buckets[key]
        if (bucket && Object.keys(bucket).length > 0) {
          patchRef.current({ [key]: bucket }).catch(() => undefined)
        }
      }
    }
  }, [])

  useEffect(() => {
    if (status !== 'saved') return
    const id = window.setTimeout(() => setStatus('idle'), 2200)
    return () => window.clearTimeout(id)
  }, [status, tick])

  const merged = useMemo(() => {
    let out = config
    for (const [section, values] of Object.entries(overlay.current)) {
      const key = section as ConfigSection
      out = { ...out, [key]: { ...(out[key] as object), ...values } }
    }
    return out
    // `tick` is what signals that the overlay changed.
  }, [config, tick])

  const value = useMemo<SettingsController>(
    () => ({ config: merged, update, updateRoot, status }),
    [merged, update, updateRoot, status],
  )

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

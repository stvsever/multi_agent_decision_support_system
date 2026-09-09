/** Data hooks shared across screens. */

import { useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import type {
  AccountStatus,
  Capabilities,
  CatalogResponse,
  DashboardConfig,
  ParticipantsResponse,
  RunDetail,
  SettingsResponse,
} from './types'

export const queryKeys = {
  capabilities: ['capabilities'] as const,
  settings: ['settings'] as const,
  account: ['account'] as const,
  catalog: (params: Record<string, unknown>) => ['catalog', params] as const,
  participants: ['participants'] as const,
  ontology: (dir: string) => ['ontology', dir] as const,
  runs: ['runs'] as const,
  run: (id: string) => ['run', id] as const,
  batches: ['batches'] as const,
  report: (params: Record<string, unknown>) => ['report', params] as const,
  prompts: ['prompts'] as const,
}

export const useCapabilities = (): UseQueryResult<Capabilities> =>
  useQuery({ queryKey: queryKeys.capabilities, queryFn: api.capabilities, staleTime: Infinity })

export const useSettings = (): UseQueryResult<SettingsResponse> =>
  useQuery({ queryKey: queryKeys.settings, queryFn: api.settings.get, staleTime: 30_000 })

export const useAccount = (enabled = true): UseQueryResult<AccountStatus> =>
  useQuery({
    queryKey: queryKeys.account,
    queryFn: api.settings.verifyCredential,
    enabled,
    staleTime: 60_000,
    retry: false,
  })

export const useParticipants = (): UseQueryResult<ParticipantsResponse> =>
  useQuery({ queryKey: queryKeys.participants, queryFn: () => api.datasets.participants(), staleTime: 20_000 })

export const useCatalog = (
  params: Parameters<typeof api.catalog.models>[0] = {},
  enabled = true,
): UseQueryResult<CatalogResponse> =>
  useQuery({
    queryKey: queryKeys.catalog(params as Record<string, unknown>),
    queryFn: () => api.catalog.models(params),
    enabled,
    staleTime: 5 * 60_000,
    retry: 1,
  })

export const useOntology = (directory: string | null) =>
  useQuery({
    queryKey: queryKeys.ontology(directory ?? ''),
    queryFn: () => api.datasets.ontology(directory!),
    enabled: Boolean(directory),
    staleTime: 5 * 60_000,
  })

export const useRuns = (poll = false) =>
  useQuery({
    queryKey: queryKeys.runs,
    queryFn: () => api.runs.list(),
    refetchInterval: poll ? 4000 : false,
    staleTime: 2000,
  })

export const useBatches = (poll = false) =>
  useQuery({
    queryKey: queryKeys.batches,
    queryFn: api.batches.list,
    refetchInterval: poll ? 3000 : false,
  })

export const usePrompts = () =>
  useQuery({ queryKey: queryKeys.prompts, queryFn: api.prompts.list, staleTime: 60_000 })

/** Patch a configuration section and refresh every dependent query. */
export function useConfigPatch() {
  const client = useQueryClient()
  return async (patch: Partial<Record<keyof DashboardConfig, unknown>>) => {
    const result = await api.settings.patch(patch)
    client.setQueryData<SettingsResponse>(queryKeys.settings, (prev) =>
      prev ? { ...prev, config: result.config, effective: result.effective } : prev,
    )
    return result.config
  }
}

/**
 * Live run state over server-sent events, with a polling fallback so a proxy
 * that buffers the stream still produces a usable screen.
 */
export function useRunStream(runId: string | null) {
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [connected, setConnected] = useState(false)
  const client = useQueryClient()
  const detailRef = useRef<RunDetail | null>(null)
  detailRef.current = detail

  useEffect(() => {
    if (!runId) {
      setDetail(null)
      return
    }
    let cancelled = false
    let source: EventSource | null = null
    let pollTimer: number | undefined

    const finish = () => {
      client.invalidateQueries({ queryKey: queryKeys.runs })
      client.invalidateQueries({ queryKey: queryKeys.batches })
    }

    const poll = async () => {
      try {
        const next = await api.runs.get(runId)
        if (!cancelled) setDetail(next)
        if (!cancelled && next.status !== 'succeeded' && next.status !== 'failed' && next.status !== 'cancelled') {
          pollTimer = window.setTimeout(poll, 2000)
        } else {
          finish()
        }
      } catch {
        if (!cancelled) pollTimer = window.setTimeout(poll, 4000)
      }
    }

    api.runs
      .get(runId)
      .then((initial) => {
        if (!cancelled) setDetail(initial)
      })
      .catch(() => undefined)

    try {
      source = new EventSource(api.runs.streamUrl(runId))
      source.onopen = () => setConnected(true)
      source.onmessage = (event) => {
        if (cancelled) return
        let message: Record<string, any>
        try {
          message = JSON.parse(event.data)
        } catch {
          return
        }
        setDetail((prev) => reduceRun(prev, message))
        if (message.type === 'done') {
          finish()
          source?.close()
          setConnected(false)
        }
      }
      source.onerror = () => {
        setConnected(false)
        source?.close()
        source = null
        if (!cancelled && !pollTimer) poll()
      }
    } catch {
      poll()
    }

    return () => {
      cancelled = true
      source?.close()
      if (pollTimer) window.clearTimeout(pollTimer)
    }
  }, [runId, client])

  return { detail, connected }
}

function reduceRun(prev: RunDetail | null, message: Record<string, any>): RunDetail | null {
  if (message.type === 'snapshot') {
    const { type, ...rest } = message
    return rest as RunDetail
  }
  if (!prev) return prev
  switch (message.type) {
    case 'event': {
      const events = [...prev.events, message.event].slice(-4000)
      return {
        ...prev,
        events,
        state: message.state ?? prev.state,
        graph: message.graph ?? prev.graph,
      }
    }
    case 'usage':
      return { ...prev, usage: message.usage ?? prev.usage, cost: message.cost ?? prev.cost }
    case 'result':
      return { ...prev, result: message.result ?? prev.result }
    case 'log':
      return { ...prev, logs: [...prev.logs, message.line].slice(-400) }
    case 'status':
    case 'done':
      return { ...prev, ...(message.run ?? {}) }
    default:
      return prev
  }
}

/** Ticking clock for live elapsed timers, shared so components stay in step. */
export function useNow(active: boolean, intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(id)
  }, [active, intervalMs])
  return now
}

export function useDebounced<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delay)
    return () => window.clearTimeout(id)
  }, [value, delay])
  return debounced
}

/** Media query hook used for the responsive layout breakpoints. */
export function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const list = window.matchMedia(query)
    const handler = (event: MediaQueryListEvent) => setMatches(event.matches)
    list.addEventListener('change', handler)
    setMatches(list.matches)
    return () => list.removeEventListener('change', handler)
  }, [query])
  return matches
}

export function useMemoisedIds<T extends { id: string }>(rows: T[] | undefined): Record<string, T> {
  return useMemo(() => Object.fromEntries((rows ?? []).map((row) => [row.id, row])), [rows])
}

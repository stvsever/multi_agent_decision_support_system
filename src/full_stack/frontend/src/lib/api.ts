/** Typed client for the dashboard service. */

import type {
  AccountStatus,
  BatchStatus,
  BrowseResponse,
  Capabilities,
  CatalogResponse,
  Connectivity,
  CredentialProvider,
  DashboardConfig,
  DeployPlan,
  DeployProbe,
  DistributionResponse,
  EstimateResponse,
  HfModelDetail,
  HfModelRow,
  LocalBackendConfig,
  Ontology,
  Participant,
  ParticipantsResponse,
  PromptRow,
  ReportBundle,
  RunDetail,
  RunOverrides,
  RunSummary,
  SettingsResponse,
  TaskSpecInput,
} from './types'

export const API_BASE = '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    })
  } catch (cause) {
    throw new ApiError(
      'The dashboard service is not reachable. Confirm it is running on port 5005.',
      0,
      cause,
    )
  }

  if (!response.ok) {
    let detail: unknown
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      detail = body
      if (typeof body?.detail === 'string') message = body.detail
      else if (Array.isArray(body?.detail)) {
        message = body.detail
          .map((d: { loc?: string[]; msg?: string }) => `${d.loc?.slice(1).join('.') ?? ''} ${d.msg ?? ''}`.trim())
          .join('; ')
      }
    } catch {
      /* a non-JSON error body keeps the status text */
    }
    throw new ApiError(message, response.status, detail)
  }

  if (response.status === 204) return undefined as T
  const contentType = response.headers.get('content-type') ?? ''
  if (contentType.includes('application/json')) return (await response.json()) as T
  return (await response.text()) as unknown as T
}

const query = (params: Record<string, string | number | boolean | undefined | null>) => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),
  capabilities: () => request<Capabilities>('/capabilities'),
  /* Grouped under /system so it shares a namespace with the storage controls. */
  connectivity: () => request<Connectivity>('/system/connectivity'),

  settings: {
    get: () => request<SettingsResponse>('/settings'),
    patch: (patch: Partial<Record<keyof DashboardConfig, unknown>>) =>
      request<{ config: DashboardConfig; effective: Record<string, unknown> }>('/settings', {
        method: 'PATCH',
        body: JSON.stringify(patch),
      }),
    reset: () =>
      request<{ config: DashboardConfig; effective: Record<string, unknown> }>('/settings/reset', {
        method: 'POST',
      }),
    setCredential: (provider: CredentialProvider, apiKey: string) =>
      request<{ credential: unknown; verification: AccountStatus }>('/settings/credentials', {
        method: 'PUT',
        body: JSON.stringify({ provider, api_key: apiKey }),
      }),
    verifyCredential: () => request<AccountStatus>('/settings/credentials/verify'),
  },

  catalog: {
    models: (params: {
      refresh?: boolean
      search?: string
      provider?: string
      embedding?: boolean
      free_only?: boolean
      min_context?: number
      limit?: number
    } = {}) => request<CatalogResponse>(`/catalog/models${query(params)}`),
    account: () => request<AccountStatus>('/catalog/account'),
  },

  datasets: {
    participants: (refresh = false) =>
      request<ParticipantsResponse>(`/datasets/participants${query({ refresh })}`),
    participant: (directory: string) =>
      request<Participant>(`/datasets/participant${query({ directory })}`),
    file: (directory: string, key: string) =>
      request<{ key: string; file: string; format: string; content: unknown; error?: string }>(
        `/datasets/participant/file${query({ directory, key })}`,
      ),
    ontology: (directory: string) => request<Ontology>(`/datasets/ontology${query({ directory })}`),
    ontologyAggregate: (directories: string[]) =>
      request<Ontology>('/datasets/ontology/aggregate', {
        method: 'POST',
        body: JSON.stringify({ directories }),
      }),
    distribution: (directories: string[], path: string[]) =>
      request<DistributionResponse>('/datasets/ontology/distribution', {
        method: 'POST',
        body: JSON.stringify({ directories, path }),
      }),
    browse: (path?: string) => request<BrowseResponse>(`/datasets/browse${query({ path })}`),
    addRoot: (path: string) =>
      request<ParticipantsResponse>('/datasets/roots', { method: 'POST', body: JSON.stringify({ path }) }),
    removeRoot: (path: string) =>
      request<ParticipantsResponse>(`/datasets/roots${query({ path })}`, { method: 'DELETE' }),
  },

  runs: {
    estimate: (body: {
      participant_dirs: string[]
      task: TaskSpecInput
      overrides?: RunOverrides
      generate_deep_phenotype?: boolean
    }) => request<EstimateResponse>('/runs/estimate', { method: 'POST', body: JSON.stringify(body) }),
    create: (body: {
      participant_dir: string
      task: TaskSpecInput
      overrides?: RunOverrides
      generate_deep_phenotype?: boolean
      label?: string
    }) => request<RunSummary>('/runs', { method: 'POST', body: JSON.stringify(body) }),
    audit: (body: { participant_dir: string; task: TaskSpecInput }) =>
      request<RunSummary>('/runs/audit', { method: 'POST', body: JSON.stringify(body) }),
    list: (limit = 100) => request<{ runs: RunSummary[] }>(`/runs${query({ limit })}`),
    get: (runId: string, sinceEvent = 0) =>
      request<RunDetail>(`/runs/${runId}${query({ since_event: sinceEvent })}`),
    cancel: (runId: string) => request<RunSummary>(`/runs/${runId}/cancel`, { method: 'POST' }),
    remove: (runId: string) => request<{ deleted: string }>(`/runs/${runId}`, { method: 'DELETE' }),
    streamUrl: (runId: string) => `${API_BASE}/runs/${runId}/stream`,
  },

  batches: {
    create: (body: {
      participant_dirs: string[]
      task: TaskSpecInput
      overrides?: RunOverrides
      generate_deep_phenotype?: boolean
      concurrency?: number
      continue_on_error?: boolean
      label?: string
    }) => request<BatchStatus>('/batches', { method: 'POST', body: JSON.stringify(body) }),
    list: () => request<{ batches: BatchStatus[] }>('/batches'),
    get: (batchId: string) => request<BatchStatus>(`/batches/${batchId}`),
    cancel: (batchId: string) => request<BatchStatus>(`/batches/${batchId}/cancel`, { method: 'POST' }),
  },

  reports: {
    bundle: (params: { run_id?: string; participant_dir?: string }) =>
      request<ReportBundle>(`/reports${query(params)}`),
    file: (params: { key: string; run_id?: string; participant_dir?: string }) =>
      request<string>(`/reports/file${query(params)}`),
    pdfUrl: (params: { run_id?: string; participant_dir?: string }) =>
      `${API_BASE}/reports/pdf${query(params)}`,
  },

  hf: {
    search: (params: { q?: string; task?: string; limit?: number } = {}) =>
      request<{ models: HfModelRow[]; error?: string }>(`/hf/models${query(params)}`),
    detail: (modelId: string) => request<HfModelDetail>(`/hf/model/${modelId}`),
  },

  deploy: {
    plan: () => request<DeployPlan>('/deploy/plan'),
    preview: (local: Partial<LocalBackendConfig>) =>
      request<DeployPlan>('/deploy/plan', { method: 'POST', body: JSON.stringify(local) }),
    probe: () => request<DeployProbe>('/deploy/probe'),
  },

  prompts: {
    list: () => request<{ prompts: PromptRow[] }>('/prompts'),
    get: (scope: string, name: string) =>
      request<{ scope: string; name: string; content: string; original: string; modified: boolean }>(
        `/prompts/${scope}/${name}`,
      ),
    save: (scope: string, name: string, content: string) =>
      request<{ modified: boolean }>(`/prompts/${scope}/${name}`, {
        method: 'PUT',
        body: JSON.stringify({ content }),
      }),
    restore: (scope: string, name: string) =>
      request<{ content: string; modified: boolean }>(`/prompts/${scope}/${name}/restore`, {
        method: 'POST',
      }),
  },
}

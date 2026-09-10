/**
 * Local storage inventory.
 *
 * What COMPASS has written on this machine, and the means to delete it. The
 * two endpoints are newer than the rest of the service contract, so a 404 is
 * read as "this build cannot report storage" and the panel says so instead of
 * failing.
 *
 * Contract assumed by this client:
 *
 *   GET /api/system/storage
 *     -> { entries: StorageEntry[], total_bytes: number }
 *
 *   DELETE /api/system/storage/{key}
 *     -> { key: string, deleted: boolean, freed_bytes: number }
 *
 * `key` is one of the STORAGE_KEYS below. `bytes` is the size on disk of the
 * whole entry, `item_count` the number of files or records it holds (null when
 * the entry is a single file), and `removable` is false for an entry the
 * service will not delete, which renders the row without an action.
 */

export const STORAGE_KEYS = ['model_catalog', 'runs', 'prompts', 'embeddings', 'outputs', 'config'] as const
export type StorageKey = (typeof STORAGE_KEYS)[number]

export interface StorageEntry {
  key: string
  label: string
  description: string
  path: string
  bytes: number
  item_count: number | null
  exists: boolean
  removable: boolean
}

export interface StorageResponse {
  entries: StorageEntry[]
  total_bytes: number
}

export interface StorageDeleteResult {
  key: string
  deleted: boolean
  freed_bytes: number
}

/** Raised when the service has no storage endpoint at all. */
export class StorageUnavailable extends Error {
  constructor() {
    super('The dashboard service does not report local storage.')
    this.name = 'StorageUnavailable'
  }
}

const BASE = '/api/system/storage'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    throw new Error('The dashboard service is not reachable.')
  }

  if (response.status === 404 || response.status === 501) throw new StorageUnavailable()

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') message = body.detail
    } catch {
      /* a non-JSON error body keeps the status text */
    }
    throw new Error(message)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const readStorage = () => call<StorageResponse>(BASE)

export const removeStorage = (key: string) =>
  call<StorageDeleteResult>(`${BASE}/${encodeURIComponent(key)}`, { method: 'DELETE' })

/**
 * What a delete actually costs, in the reader's terms. The service sends a
 * description of each entry; this is the sentence shown when the delete is
 * armed, which has to be about the consequence rather than the contents.
 *
 * The `config` entry is deliberately absent: that row runs the settings reset,
 * whose sentence lives with the reset itself in `reset.tsx` so the two places
 * that offer it cannot drift apart.
 */
export const LOSS: Record<string, string> = {
  model_catalog:
    'The cached model list and prices are removed. The next catalog request fetches them again from the provider, which needs a network connection.',
  runs: 'Every stored run record and its event log are removed, so the run list will be empty. Reports already written to the output directory are not touched.',
  prompts:
    'Every prompt override is removed and each agent goes back to its shipped system prompt. Edits that were never saved elsewhere cannot be recovered.',
  embeddings:
    'The embedding cache is removed. Nothing is lost from a finished run, but the next run that needs an embedding pays for it again.',
  outputs:
    'Every generated report, execution log, and audit artifact under the output directory is deleted. This is usually the only copy.',
}

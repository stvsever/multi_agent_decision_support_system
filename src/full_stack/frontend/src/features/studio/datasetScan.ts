/**
 * Reading a scan result out loud.
 *
 * Adding a folder that turns out to hold nothing looked exactly like adding a
 * folder that worked, which is the whole complaint. The service now reports how
 * much it looked at and which folders came close, so every add can end in a
 * sentence that says what actually happened.
 */

import type { NearMiss, ParticipantsResponse, ScannedRoot } from '@/lib/types'

export const REQUIRED_FILES: { file: string; why: string }[] = [
  { file: 'data_overview.json', why: 'the per-modality summary the plan is built from' },
  { file: 'hierarchical_deviation_map.json', why: 'the evidence hierarchy with a deviation per feature' },
  { file: 'multimodal_data.json', why: 'the measured values themselves' },
  { file: 'non_numerical_data.txt', why: 'the free-text record' },
]

export const REQUIRED_FILE_NAMES = REQUIRED_FILES.map((entry) => entry.file)

export interface AddOutcome {
  tone: 'positive' | 'caution' | 'info' | 'critical'
  title: string
  body: string
  nearMisses: NearMiss[]
  root?: ScannedRoot
}

const normalise = (path: string): string => path.replace(/\/+$/, '')

const samePath = (a: string, b: string): boolean => {
  const left = normalise(a)
  const right = normalise(b)
  return left === right || left.endsWith(right) || right.endsWith(left)
}

const plural = (count: number, word: string): string => `${count} ${word}${count === 1 ? '' : 's'}`

/** What to say after POST /datasets/roots comes back. */
export function summariseAdd(
  before: ParticipantsResponse | undefined,
  after: ParticipantsResponse,
  requestedPath: string,
): AddOutcome {
  const known = new Set((before?.roots ?? []).map((root) => root.root))
  const fresh = after.roots.find((root) => !known.has(root.root))
  const matched = fresh ?? after.roots.find((root) => samePath(root.root, requestedPath))
  const delta = after.count - (before?.count ?? 0)
  const nearMisses = matched?.near_misses ?? []

  const alreadyPresent = after.already_present === true || (!fresh && Boolean(matched))
  if (alreadyPresent && delta <= 0) {
    return {
      tone: 'info',
      title: 'That folder is already being scanned',
      body: matched
        ? `It contributes ${plural(matched.found, 'participant folder')} to the list below.`
        : 'It is already one of the scanned folders below.',
      nearMisses,
      root: matched,
    }
  }

  const found = fresh ? fresh.found : delta
  if (found > 0) {
    return {
      tone: 'positive',
      title: `Added, ${plural(found, 'participant folder')} found`,
      body: matched?.scanned_dir_count
        ? `Scanned ${plural(matched.scanned_dir_count, 'folder')} under this path.`
        : 'The new folders are selectable in the list above.',
      nearMisses,
      root: matched,
    }
  }

  return {
    tone: 'caution',
    title: 'Added, but nothing in it looks like a participant input',
    body: matched?.scanned_dir_count
      ? `Scanned ${plural(matched.scanned_dir_count, 'folder')} up to four levels deep and none held all four required files.`
      : 'Nothing under this path held all four required files.',
    nearMisses,
    root: matched,
  }
}

/** "missing multimodal_data.json and non_numerical_data.txt" */
export function describeMissing(missing: string[]): string {
  if (missing.length === 0) return 'nothing missing'
  if (missing.length === 1) return `missing ${missing[0]}`
  return `missing ${missing.slice(0, -1).join(', ')} and ${missing[missing.length - 1]}`
}

/**
 * Reading and writing a task tree as .json.
 *
 * A dropped file is checked field by field before it is applied, so a bad file
 * is answered with the node and the field that caused it rather than with a
 * parser message. Export writes the same shape the import accepts, which makes
 * the format discoverable by round trip.
 */

import type { NodeMode, TaskNodeInput } from '@/lib/types'
import { MODE_ORDER } from './taskValidation'

export interface ImportIssue {
  /** Where in the file: `root.children[1].mode`. */
  path: string
  message: string
}

export type ImportResult =
  | { ok: true; root: TaskNodeInput; nodeCount: number }
  | { ok: false; issues: ImportIssue[] }

const MAX_NODES = 200
const MAX_DEPTH = 8

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/**
 * Say where the file went wrong, in place of the engine's own wording.
 *
 * The parsers disagree on the phrasing but agree on the location: V8 gives
 * "(line 3 column 5)" and, on older builds, only "at position 18"; Firefox
 * gives "line 3 column 5" without the brackets. Any of those is enough to
 * point at a spot in the text, which is the only part worth repeating.
 */
function describeParseError(text: string, error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)

  let line = 0
  let column = 0
  const stated = /line (\d+) column (\d+)/.exec(message)
  const position = /position (\d+)/.exec(message)
  if (stated) {
    line = Number(stated[1])
    column = Number(stated[2])
  } else if (position) {
    const before = text.slice(0, Number(position[1]))
    line = before.split('\n').length
    column = Number(position[1]) - before.lastIndexOf('\n')
  }

  if (!line) {
    return 'the text could not be read at all. Check that every name is in double quotes and every bracket is closed.'
  }
  return `something is wrong at line ${line}, column ${column}. Check for a missing quote, comma or brace there.`
}

function readStringList(
  value: unknown,
  path: string,
  field: string,
  issues: ImportIssue[],
): string[] {
  if (value === undefined || value === null) return []
  if (typeof value === 'string') return value.split(',').map((part) => part.trim()).filter(Boolean)
  if (!Array.isArray(value)) {
    issues.push({ path: `${path}.${field}`, message: 'Expected a list of names.' })
    return []
  }
  const out: string[] = []
  value.forEach((entry, index) => {
    if (typeof entry !== 'string' && typeof entry !== 'number') {
      issues.push({ path: `${path}.${field}[${index}]`, message: 'Expected a name, not an object.' })
      return
    }
    const text = String(entry).trim()
    if (text && !out.includes(text)) out.push(text)
  })
  return out
}

function readNode(
  raw: unknown,
  path: string,
  depth: number,
  issues: ImportIssue[],
  counter: { total: number },
): TaskNodeInput | null {
  if (!isRecord(raw)) {
    issues.push({ path, message: 'Expected an object describing one node.' })
    return null
  }
  counter.total += 1
  if (counter.total > MAX_NODES) {
    issues.push({ path, message: `This tree has more than ${MAX_NODES} nodes, which is past what the designer can show.` })
    return null
  }
  if (depth > MAX_DEPTH) {
    issues.push({ path, message: `Nested deeper than ${MAX_DEPTH} levels.` })
    return null
  }

  const label = typeof raw.display_name === 'string' ? raw.display_name.trim() : ''
  const where = label ? `${path} ("${label}")` : path

  const nodeId = typeof raw.node_id === 'string' ? raw.node_id.trim() : ''
  if (!nodeId) {
    issues.push({
      path: `${where}.node_id`,
      message: raw.node_id === undefined ? 'Missing. Every node needs an id.' : 'Must be a non-empty string.',
    })
  }
  if (!label) {
    issues.push({
      path: `${where}.display_name`,
      message: raw.display_name === undefined ? 'Missing. Every node needs a name.' : 'Must be a non-empty string.',
    })
  }

  const mode = raw.mode
  if (typeof mode !== 'string' || !MODE_ORDER.includes(mode as NodeMode)) {
    issues.push({
      path: `${where}.mode`,
      message: `${mode === undefined ? 'Missing' : `"${String(mode)}" is not a mode`}. Use one of ${MODE_ORDER.join(', ')}.`,
    })
  }

  const classLabels = readStringList(raw.class_labels, where, 'class_labels', issues)
  const regressionOutputs = readStringList(raw.regression_outputs, where, 'regression_outputs', issues)

  const units: Record<string, string> = {}
  if (raw.unit_by_output !== undefined && raw.unit_by_output !== null) {
    if (!isRecord(raw.unit_by_output)) {
      issues.push({ path: `${where}.unit_by_output`, message: 'Expected an object mapping an output name to its unit.' })
    } else {
      for (const [key, value] of Object.entries(raw.unit_by_output)) units[key] = String(value ?? '')
    }
  }

  if (raw.required !== undefined && typeof raw.required !== 'boolean') {
    issues.push({ path: `${where}.required`, message: 'Expected true or false.' })
  }

  const children: TaskNodeInput[] = []
  if (raw.children !== undefined && raw.children !== null) {
    if (!Array.isArray(raw.children)) {
      issues.push({ path: `${where}.children`, message: 'Expected a list of nodes.' })
    } else {
      raw.children.forEach((child, index) => {
        const parsed = readNode(child, `${path}.children[${index}]`, depth + 1, issues, counter)
        if (parsed) children.push(parsed)
      })
    }
  }

  if (issues.length > 0) return null

  return {
    node_id: nodeId,
    display_name: label,
    mode: mode as NodeMode,
    class_labels: classLabels,
    regression_outputs: regressionOutputs,
    unit_by_output: units,
    required: raw.required === undefined ? true : Boolean(raw.required),
    children,
  }
}

/**
 * Accepts a bare node, `{ root: ... }`, or a whole task specification, since
 * all three are shapes a person plausibly has on disk.
 */
export function parseTaskTree(text: string): ImportResult {
  const trimmed = text.trim()
  if (!trimmed) return { ok: false, issues: [{ path: 'file', message: 'Nothing to read: the file or box is empty.' }] }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch (error) {
    return {
      ok: false,
      issues: [{ path: 'file', message: `This file is not valid JSON: ${describeParseError(trimmed, error)}` }],
    }
  }

  if (!isRecord(parsed)) {
    return {
      ok: false,
      issues: [{ path: 'file', message: 'Expected a JSON object, either a node or a task with a root.' }],
    }
  }

  let rootRaw: unknown = parsed
  if ('root' in parsed) {
    if (parsed.root === null || parsed.root === undefined) {
      return { ok: false, issues: [{ path: 'root', message: 'The file has a root key with nothing in it.' }] }
    }
    rootRaw = parsed.root
  } else if (!('node_id' in parsed) && !('mode' in parsed)) {
    return {
      ok: false,
      issues: [
        {
          path: 'file',
          message: 'No task tree here. Expected a "root" object, or a node with node_id, display_name and mode.',
        },
      ],
    }
  }

  const issues: ImportIssue[] = []
  const counter = { total: 0 }
  const root = readNode(rootRaw, 'root', 0, issues, counter)
  if (!root) return { ok: false, issues: issues.length ? issues : [{ path: 'root', message: 'Could not read the root node.' }] }
  return { ok: true, root, nodeCount: counter.total }
}

/** The exported shape is exactly what the import accepts and what /runs takes. */
export function serialiseTaskTree(root: TaskNodeInput): string {
  const node = (input: TaskNodeInput): Record<string, unknown> => {
    const out: Record<string, unknown> = {
      node_id: input.node_id,
      display_name: input.display_name,
      mode: input.mode,
    }
    if (input.mode.endsWith('classification')) out.class_labels = input.class_labels
    else out.regression_outputs = input.regression_outputs
    if (Object.keys(input.unit_by_output ?? {}).length > 0) out.unit_by_output = input.unit_by_output
    out.required = input.required
    out.children = (input.children ?? []).map(node)
    return out
  }
  return `${JSON.stringify({ prediction_type: 'hierarchical', root: node(root) }, null, 2)}\n`
}

export function downloadJson(filename: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // Revoking immediately can cancel the download in some browsers.
  window.setTimeout(() => URL.revokeObjectURL(url), 2000)
}

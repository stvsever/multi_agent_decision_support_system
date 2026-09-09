/**
 * Pure structure over the ontology payload.
 *
 * Every view reads the same flat index, so the tree, the two partition charts
 * and the table agree on keys, colours and filtering without recomputing.
 */

import type { Band, Ontology, OntologyFeature, OntologyNode } from '@/lib/types'

export const BAND_ORDER: Band[] = [
  'very_high',
  'high',
  'high_normal',
  'normal',
  'low_normal',
  'low',
  'very_low',
  'missing',
]

export const BAND_LABEL: Record<Band, string> = {
  very_high: 'Very high',
  high: 'High',
  high_normal: 'High normal',
  normal: 'Normal',
  low_normal: 'Low normal',
  low: 'Low',
  very_low: 'Very low',
  missing: 'Missing',
}

/** Deviation colour always resolves through a token so dark mode follows. */
export const bandColor = (band: Band): string => `var(--dev-${band.replace(/_/g, '-')})`

/** Mirrors the thresholds the service uses, for values we band client side. */
export function bandFor(z: number | null | undefined): Band {
  if (z === null || z === undefined || Number.isNaN(z)) return 'missing'
  if (z >= 2) return 'very_high'
  if (z >= 1) return 'high'
  if (z >= 0.5) return 'high_normal'
  if (z >= -0.5) return 'normal'
  if (z >= -1) return 'low_normal'
  if (z >= -2) return 'low'
  return 'very_low'
}

/** Node ids carry punctuation, so paths join on a character they never hold. */
export const SEP = '\u001f'
export const ROOT_KEY = '\u0000root'

export const keyOf = (path: readonly string[]): string => path.join(SEP)

/** Long all-caps tokens that are still acronyms, past the length heuristic. */
const ACRONYMS = new Set(['PANSS', 'MADRS', 'WAIS', 'WISC', 'BPRS', 'YMRS', 'HAMD', 'FMRI'])
const MINOR = new Set(['and', 'or', 'of', 'the', 'a', 'an', 'to', 'in', 'at', 'for', 'with', 'per'])

/**
 * Labels arrive part humanised. Shouty ids stay shouty unless we soften them,
 * so all-caps text is title cased while mixed-case labels are left untouched.
 * Short all-caps tokens are read as acronyms: BDNF and SES survive, BRAIN does
 * not.
 */
export function prettyLabel(raw: string): string {
  const text = String(raw ?? '').replace(/_/g, ' ').trim()
  if (!text) return ''
  if (/[a-z]/.test(text)) return text.charAt(0).toUpperCase() + text.slice(1)
  return text
    .split(/\s+/)
    .map((word, index) => {
      const lower = word.toLowerCase()
      if (index > 0 && MINOR.has(lower)) return lower
      if (ACRONYMS.has(word) || (word.length <= 4 && /^[A-Z]+$/.test(word))) return word
      return lower.charAt(0).toUpperCase() + lower.slice(1)
    })
    .join(' ')
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '-'
    if (Number.isInteger(value)) return String(value)
    return String(Number(value.toFixed(4)))
  }
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (Array.isArray(value)) return value.map(formatValue).join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

/** A measured value a node carries directly: its own score, or a feature. */
export interface Atom {
  band: Band
  z: number | null
}

export interface FlatNode {
  key: string
  parent: string | null
  children: string[]
  node: OntologyNode
  depth: number
  domain: string
  haystack: string
  /** Magnitude used for ranking: the subtree mean, or the leaf's own |z|. */
  magnitude: number | null
  /**
   * Mean signed z over every measurement in the subtree. The service only
   * averages a node's direct leaf children, which leaves most branches with a
   * null score, so direction is computed here instead.
   */
  signedMean: number | null
  atoms: Atom[]
  /** The parent feature this leaf restates, holding the value and reference. */
  attached: OntologyFeature | null
  /** Features that are not already represented by a child leaf. */
  ownFeatures: OntologyFeature[]
}

export interface TableRow {
  id: string
  nodeKey: string
  feature: string | null
  label: string
  value: string
  z: number | null
  band: Band
  domain: string
  pathText: string
  haystack: string
}

export interface OntologyIndex {
  byKey: Map<string, FlatNode>
  /** Depth first preorder, so a parent always precedes its descendants. */
  order: string[]
  roots: string[]
  rows: TableRow[]
  bandCounts: Record<Band, number>
  domains: { key: string; id: string; label: string }[]
  /** `parentKey + feature name` to the leaf that already represents it. */
  attachedBy: Map<string, string>
  maxDepth: number
  /** Visual ceiling for bars and colour ramps, robust to stray raw values. */
  clamp: number
  atomCount: number
}

function emptyCounts(): Record<Band, number> {
  return {
    very_high: 0,
    high: 0,
    high_normal: 0,
    normal: 0,
    low_normal: 0,
    low: 0,
    very_low: 0,
    missing: 0,
  }
}

const normalise = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]+/g, '')

/**
 * The two source files describe the same measurement twice: once as a leaf of
 * the deviation map, once as a feature on that leaf's parent. Pairing them by
 * name gives the leaf its value and reference range, and stops every count,
 * table and histogram from reading double.
 */
function attachFeatures(
  index: Map<string, FlatNode>,
  flat: FlatNode,
  attachedBy: Map<string, string>,
): void {
  const features = flat.node.features
  if (features.length === 0 || flat.children.length === 0) return
  const pool = features.map((feature) => ({ feature, key: normalise(feature.feature), used: false }))
  const consumed = new Set<OntologyFeature>()

  for (const childKey of flat.children) {
    const child = index.get(childKey)
    if (!child || child.children.length > 0) continue
    const label = normalise(child.node.label)
    const id = normalise(child.node.id)
    if (label.length < 3 && id.length < 3) continue
    const hit = pool.find(
      (entry) =>
        !entry.used &&
        ((label.length >= 3 && entry.key.endsWith(label)) ||
          (id.length >= 3 && entry.key.endsWith(id))),
    )
    if (!hit) continue
    hit.used = true
    consumed.add(hit.feature)
    child.attached = hit.feature
    attachedBy.set(`${flat.key}${SEP}${SEP}${hit.feature.feature}`, child.key)
  }

  if (consumed.size > 0) flat.ownFeatures = features.filter((f) => !consumed.has(f))
}

export function buildIndex(ontology: Ontology): OntologyIndex {
  const byKey = new Map<string, FlatNode>()
  const order: string[] = []
  const roots: string[] = []
  const rows: TableRow[] = []
  const bandCounts = emptyCounts()
  const magnitudes: number[] = []
  const attachedBy = new Map<string, string>()
  let maxDepth = 0
  let atomCount = 0

  // Pass one: structure only, since pairing a leaf with its parent's feature
  // needs the children to exist first.
  const visit = (node: OntologyNode, parent: string | null): string => {
    const key = keyOf(node.path)
    const flat: FlatNode = {
      key,
      parent,
      children: [],
      node,
      depth: node.depth,
      domain: node.path[0] ?? node.id,
      haystack: '',
      magnitude: node.mean_abs_score ?? (node.score === null ? null : Math.abs(node.score)),
      signedMean: null,
      atoms: [],
      attached: null,
      ownFeatures: node.features,
    }
    byKey.set(key, flat)
    order.push(key)
    maxDepth = Math.max(maxDepth, node.depth)
    flat.children = node.children.map((child) => visit(child, key))
    return key
  }
  for (const domain of ontology.domains) roots.push(visit(domain, null))

  for (const key of order) {
    const flat = byKey.get(key)
    if (flat) attachFeatures(byKey, flat, attachedBy)
  }

  // Pass two: every measurement now appears exactly once.
  for (const key of order) {
    const flat = byKey.get(key)
    if (!flat) continue
    const node = flat.node
    const pathText = node.path.join(' / ')
    const isLeaf = flat.children.length === 0

    flat.haystack = `${node.label} ${node.id} ${flat.attached?.feature ?? ''} ${flat.ownFeatures
      .map((f) => f.feature)
      .join(' ')}`.toLowerCase()

    if (isLeaf) {
      flat.atoms.push({ band: node.band, z: node.score })
      rows.push({
        id: key,
        nodeKey: key,
        feature: null,
        label: prettyLabel(node.label),
        value: flat.attached ? formatValue(flat.attached.value) : '-',
        z: node.score,
        band: node.band,
        domain: flat.domain,
        pathText,
        haystack: `${node.label} ${node.id} ${flat.attached?.feature ?? ''} ${pathText}`.toLowerCase(),
      })
    }
    for (const feature of flat.ownFeatures) {
      flat.atoms.push({ band: feature.band, z: feature.z_score })
      rows.push({
        id: `${key}${SEP}${SEP}${feature.feature}`,
        nodeKey: key,
        feature: feature.feature,
        label: prettyLabel(feature.feature),
        value: formatValue(feature.value),
        z: feature.z_score,
        band: feature.band,
        domain: flat.domain,
        pathText,
        haystack: `${feature.feature} ${pathText}`.toLowerCase(),
      })
    }
    for (const atom of flat.atoms) {
      bandCounts[atom.band] += 1
      atomCount += 1
      if (atom.z !== null && Number.isFinite(atom.z)) magnitudes.push(Math.abs(atom.z))
    }
  }

  // Reverse preorder rolls each subtree's signed mean up into its parent.
  const sums = new Map<string, { sum: number; count: number }>()
  for (let i = order.length - 1; i >= 0; i -= 1) {
    const flat = byKey.get(order[i])
    if (!flat) continue
    let sum = 0
    let count = 0
    for (const atom of flat.atoms) {
      if (atom.z !== null && Number.isFinite(atom.z)) {
        sum += atom.z
        count += 1
      }
    }
    for (const child of flat.children) {
      const rolled = sums.get(child)
      if (rolled) {
        sum += rolled.sum
        count += rolled.count
      }
    }
    sums.set(flat.key, { sum, count })
    flat.signedMean = count > 0 ? sum / count : null
  }

  magnitudes.sort((a, b) => a - b)
  const p98 = magnitudes.length
    ? magnitudes[Math.min(magnitudes.length - 1, Math.floor(magnitudes.length * 0.98))]
    : 0
  const clamp = Math.max(3, Math.min(12, Math.ceil(p98 * 2) / 2))

  return {
    byKey,
    order,
    roots,
    rows,
    bandCounts,
    domains: ontology.domains.map((d) => ({
      key: keyOf(d.path),
      id: d.id,
      label: prettyLabel(d.label),
    })),
    attachedBy,
    maxDepth,
    clamp,
    atomCount,
  }
}

/* --- Filtering ------------------------------------------------------------ */

export interface Filters {
  search: string
  bands: Band[]
  minAbs: number
  domain: string
  onlyMeasured: boolean
}

export const emptyFilters = (): Filters => ({
  search: '',
  bands: [],
  minAbs: 0,
  domain: '',
  onlyMeasured: false,
})

export const filtersActive = (filters: Filters): boolean =>
  filters.bands.length > 0 || filters.minAbs > 0 || filters.domain !== '' || filters.onlyMeasured

/**
 * Filters judge the measured atoms a node carries, then internal nodes survive
 * if anything below them survives. That keeps the path to a hit intact instead
 * of leaving orphaned leaves with no context.
 */
export function computeKeep(index: OntologyIndex, filters: Filters): Set<string> | null {
  if (!filtersActive(filters)) return null
  const bands = filters.bands.length ? new Set<Band>(filters.bands) : null
  const keep = new Set<string>()

  const selfPasses = (flat: FlatNode): boolean => {
    if (filters.domain && flat.domain !== filters.domain) return false
    for (const atom of flat.atoms) {
      if (bands && !bands.has(atom.band)) continue
      if (filters.onlyMeasured && atom.z === null) continue
      if (atom.z === null) {
        if (filters.minAbs > 0) continue
      } else if (Math.abs(atom.z) < filters.minAbs) continue
      return true
    }
    return false
  }

  // Reverse preorder visits every descendant before its parent.
  for (let i = index.order.length - 1; i >= 0; i -= 1) {
    const key = index.order[i]
    const flat = index.byKey.get(key)
    if (!flat) continue
    if (selfPasses(flat) || flat.children.some((child) => keep.has(child))) keep.add(key)
  }
  return keep
}

export function computeMatches(index: OntologyIndex, query: string): Set<string> | null {
  const needle = query.trim().toLowerCase()
  if (!needle) return null
  const matches = new Set<string>()
  for (const [key, flat] of index.byKey) {
    if (flat.haystack.includes(needle)) matches.add(key)
  }
  return matches
}

export function ancestorsOf(index: OntologyIndex, keys: Iterable<string>): Set<string> {
  const out = new Set<string>()
  for (const key of keys) {
    let cursor = index.byKey.get(key)?.parent ?? null
    while (cursor && !out.has(cursor)) {
      out.add(cursor)
      cursor = index.byKey.get(cursor)?.parent ?? null
    }
  }
  return out
}

export function chainOf(index: OntologyIndex, key: string): FlatNode[] {
  const chain: FlatNode[] = []
  let cursor: string | null = key
  while (cursor) {
    const flat: FlatNode | undefined = index.byKey.get(cursor)
    if (!flat) break
    chain.unshift(flat)
    cursor = flat.parent
  }
  return chain
}

/** Marks nodes whose subtree contains a search hit, for chart dimming. */
export function subtreeMatches(index: OntologyIndex, matches: Set<string>): Set<string> {
  const out = new Set<string>(matches)
  for (let i = index.order.length - 1; i >= 0; i -= 1) {
    const key = index.order[i]
    const flat = index.byKey.get(key)
    if (!flat || out.has(key)) continue
    if (flat.children.some((child) => out.has(child))) out.add(key)
  }
  return out
}

/* --- Hierarchy for the partition views ------------------------------------ */

export interface HierarchyDatum {
  key: string
  label: string
  band: Band
  score: number | null
  magnitude: number | null
  leaves: number
  present: number
  kids: HierarchyDatum[]
}

export function buildHierarchy(
  index: OntologyIndex,
  keep: Set<string> | null,
  rootLabel: string,
): HierarchyDatum {
  const convert = (key: string): HierarchyDatum | null => {
    if (keep && !keep.has(key)) return null
    const flat = index.byKey.get(key)
    if (!flat) return null
    const kids: HierarchyDatum[] = []
    for (const child of flat.children) {
      const converted = convert(child)
      if (converted) kids.push(converted)
    }
    return {
      key,
      label: prettyLabel(flat.node.label),
      band: flat.node.band,
      score: flat.signedMean,
      magnitude: flat.magnitude,
      leaves: Math.max(1, flat.node.leaf_count),
      present: flat.node.present_leaves,
      kids,
    }
  }

  const kids: HierarchyDatum[] = []
  for (const root of index.roots) {
    const converted = convert(root)
    if (converted) kids.push(converted)
  }
  const leaves = kids.reduce((total, kid) => total + kid.leaves, 0)

  // The synthetic root carries the participant-wide figures the sunburst
  // centre shows before any zoom.
  let absolute = 0
  let signed = 0
  let count = 0
  for (const key of keep ?? index.order) {
    for (const atom of index.byKey.get(key)?.atoms ?? []) {
      if (atom.z === null || !Number.isFinite(atom.z)) continue
      absolute += Math.abs(atom.z)
      signed += atom.z
      count += 1
    }
  }

  return {
    key: ROOT_KEY,
    label: rootLabel,
    band: count > 0 ? bandFor(absolute / count) : 'normal',
    score: count > 0 ? signed / count : null,
    magnitude: count > 0 ? absolute / count : null,
    leaves: Math.max(1, leaves),
    present: kids.reduce((total, kid) => total + kid.present, 0),
    kids,
  }
}

/* --- Subtree statistics --------------------------------------------------- */

export function collectAtoms(index: OntologyIndex, key: string): number[] {
  const out: number[] = []
  const stack = [key]
  while (stack.length) {
    const current = stack.pop()
    if (!current) continue
    const flat = index.byKey.get(current)
    if (!flat) continue
    for (const atom of flat.atoms) {
      if (atom.z !== null && Number.isFinite(atom.z)) out.push(atom.z)
    }
    for (const child of flat.children) stack.push(child)
  }
  return out
}

export interface Histogram {
  bins: { from: number; to: number; count: number; band: Band }[]
  max: number
  total: number
}

export function histogram(values: number[], clamp: number, binCount = 25): Histogram {
  const bins = Array.from({ length: binCount }, (_, i) => {
    const from = -clamp + (2 * clamp * i) / binCount
    const to = -clamp + (2 * clamp * (i + 1)) / binCount
    return { from, to, count: 0, band: bandFor((from + to) / 2) }
  })
  for (const value of values) {
    const clamped = Math.max(-clamp, Math.min(clamp, value))
    const slot = Math.min(binCount - 1, Math.floor(((clamped + clamp) / (2 * clamp)) * binCount))
    bins[slot].count += 1
  }
  return { bins, max: bins.reduce((m, b) => Math.max(m, b.count), 0), total: values.length }
}

/* --- Extremes ------------------------------------------------------------- */

export interface ExtremeChip {
  id: string
  nodeKey: string
  feature: string | null
  label: string
  score: number
  band: Band
}

/**
 * The service ranks leaves and features together, so a measurement often shows
 * up twice with the same score under the same parent. Collapse those, and
 * prefer the entry that resolves to a real node so clicking it can navigate.
 */
export function topExtremes(
  index: OntologyIndex,
  extremes: Ontology['extremes'],
  limit: number,
): ExtremeChip[] {
  const seen = new Map<string, ExtremeChip>()
  for (const row of extremes) {
    if (!Number.isFinite(row.score)) continue
    const full = keyOf(row.path)
    const parentPath = row.path.slice(0, -1)
    const parentKey = keyOf(parentPath)
    let chip: ExtremeChip | null = null
    if (index.byKey.has(full)) {
      chip = {
        id: full,
        nodeKey: full,
        feature: null,
        label: prettyLabel(row.label),
        score: row.score,
        band: row.band,
      }
    } else if (index.byKey.has(parentKey)) {
      const feature = row.path[row.path.length - 1] ?? row.label
      // A feature already represented by a leaf points at that leaf instead.
      const merged = index.attachedBy.get(`${parentKey}${SEP}${SEP}${feature}`)
      chip = merged
        ? {
            id: merged,
            nodeKey: merged,
            feature: null,
            label: prettyLabel(index.byKey.get(merged)?.node.label ?? row.label),
            score: row.score,
            band: row.band,
          }
        : {
            id: `${parentKey}${SEP}${SEP}${feature}`,
            nodeKey: parentKey,
            feature,
            label: prettyLabel(row.label),
            score: row.score,
            band: row.band,
          }
    }
    if (!chip) continue
    const dedupe = `${parentKey}#${row.score.toFixed(4)}`
    const existing = seen.get(dedupe)
    if (!existing) seen.set(dedupe, chip)
    else if (existing.feature && !chip.feature) seen.set(dedupe, chip)
    if (seen.size >= limit * 3) break
  }
  return [...seen.values()].sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, limit)
}

export function featureColumns(features: OntologyFeature[]): {
  refRange: boolean
  significance: boolean
  percentile: boolean
} {
  return {
    refRange: features.some((f) => Boolean(f.ref_range)),
    significance: features.some((f) => Boolean(f.significance)),
    percentile: features.some((f) => Boolean(f.percentile)),
  }
}

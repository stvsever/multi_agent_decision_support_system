/**
 * Placement for the graph view.
 *
 * d3-hierarchy owns the geometry, so the three presentations differ only in how
 * one tree pass is read back: depth becomes a column, a row, or a radius. The
 * result is a plain array of centres, which lets the view memoise placement and
 * keep React out of the per-frame work.
 */

import { hierarchy, tree } from 'd3-hierarchy'
import type { AggregateStats, Band } from '@/lib/types'
import { ROOT_KEY, bandFor, membershipOf, prettyLabel, type Membership, type OntologyIndex } from './model'

export type GraphLayoutName = 'lr' | 'tb' | 'radial'

export interface GraphDatum {
  key: string
  parent: string | null
  label: string
  band: Band
  /** No children at all in the source tree, as opposed to a closed branch. */
  isLeaf: boolean
  collapsed: boolean
  hidden: number
  leafCount: number
  presentLeaves: number
  depth: number
  /** Signed mean over the subtree, which is what the colour and the bar read. */
  score: number | null
  magnitude: number | null
  aggregate: AggregateStats | null
  membership: Membership | null
  radius: number
  kids: GraphDatum[]
}

export interface PlacedNode {
  key: string
  datum: GraphDatum
  x: number
  y: number
  /** Which way the label hangs. Radial mirrors it on the left half. */
  side: 'left' | 'right'
}

export interface PlacedEdge {
  id: string
  source: string
  target: string
}

export interface Bounds {
  x: number
  y: number
  width: number
  height: number
}

export interface GraphLayoutResult {
  nodes: PlacedNode[]
  edges: PlacedEdge[]
  byKey: Map<string, PlacedNode>
  bounds: Bounds
}

/* --- Building the visible tree --------------------------------------------- */

const MIN_R = 6.5
const MAX_R = 19

/**
 * The visible tree, honouring the filter and the closed branches.
 *
 * A closed branch keeps its own count of what it hides, so the node can say how
 * much is behind it instead of silently dropping evidence.
 */
export function buildGraphTree(
  index: OntologyIndex,
  keep: Set<string> | null,
  collapsed: Set<string>,
  rootLabel: string,
): GraphDatum {
  const flatList: GraphDatum[] = []

  const hiddenBelow = (key: string): number => {
    const flat = index.byKey.get(key)
    if (!flat) return 0
    let total = 0
    for (const child of flat.children) {
      if (keep && !keep.has(child)) continue
      total += 1 + hiddenBelow(child)
    }
    return total
  }

  const convert = (key: string, parent: string | null): GraphDatum | null => {
    if (keep && !keep.has(key)) return null
    const flat = index.byKey.get(key)
    if (!flat) return null
    const visibleChildren = keep ? flat.children.filter((child) => keep.has(child)) : flat.children
    const closed = visibleChildren.length > 0 && collapsed.has(key)
    const datum: GraphDatum = {
      key,
      parent,
      label: prettyLabel(flat.node.label),
      band: flat.node.band,
      isLeaf: visibleChildren.length === 0,
      collapsed: closed,
      hidden: closed ? hiddenBelow(key) : 0,
      leafCount: Math.max(1, flat.node.leaf_count),
      presentLeaves: flat.node.present_leaves,
      depth: flat.node.depth + 1,
      score: flat.signedMean ?? flat.node.score,
      magnitude: flat.magnitude,
      aggregate: flat.node.aggregate ?? null,
      membership: membershipOf(flat.node),
      radius: MIN_R,
      kids: [],
    }
    flatList.push(datum)
    if (!closed) {
      for (const child of visibleChildren) {
        const converted = convert(child, key)
        if (converted) datum.kids.push(converted)
      }
    }
    return datum
  }

  const kids: GraphDatum[] = []
  for (const rootKey of index.roots) {
    const converted = convert(rootKey, ROOT_KEY)
    if (converted) kids.push(converted)
  }

  const leafTotal = kids.reduce((sum, kid) => sum + kid.leafCount, 0)
  const signed = kids.reduce<{ sum: number; count: number }>(
    (acc, kid) => (kid.score === null ? acc : { sum: acc.sum + kid.score, count: acc.count + 1 }),
    { sum: 0, count: 0 },
  )
  const rootScore = signed.count > 0 ? signed.sum / signed.count : null
  const root: GraphDatum = {
    key: ROOT_KEY,
    parent: null,
    label: rootLabel,
    band: bandFor(rootScore),
    isLeaf: kids.length === 0,
    collapsed: false,
    hidden: 0,
    leafCount: Math.max(1, leafTotal),
    presentLeaves: kids.reduce((sum, kid) => sum + kid.presentLeaves, 0),
    depth: 0,
    score: rootScore,
    magnitude: rootScore === null ? null : Math.abs(rootScore),
    aggregate: null,
    membership: null,
    radius: MAX_R,
    kids,
  }
  flatList.push(root)

  // Area, not radius, carries the count, so a branch holding a hundred leaves
  // does not swamp one holding ten.
  const maxLeaves = flatList.reduce((max, datum) => Math.max(max, datum.leafCount), 1)
  for (const datum of flatList) {
    if (datum.key === ROOT_KEY) continue
    const share = Math.sqrt(datum.leafCount) / Math.sqrt(maxLeaves)
    datum.radius = datum.isLeaf ? MIN_R : MIN_R + (MAX_R - MIN_R) * share
  }
  return root
}

export function flattenGraphTree(root: GraphDatum): Map<string, GraphDatum> {
  const out = new Map<string, GraphDatum>()
  const stack: GraphDatum[] = [root]
  while (stack.length) {
    const datum = stack.pop()
    if (!datum) continue
    out.set(datum.key, datum)
    for (const kid of datum.kids) stack.push(kid)
  }
  return out
}

/** Every key at or beneath one node, used by the focus and collapse controls. */
export function subtreeKeys(datum: GraphDatum): string[] {
  const out: string[] = []
  const stack: GraphDatum[] = [datum]
  while (stack.length) {
    const current = stack.pop()
    if (!current) continue
    out.push(current.key)
    for (const kid of current.kids) stack.push(kid)
  }
  return out
}

/* --- Placement ------------------------------------------------------------- */

interface Spacing {
  across: number
  depth: number
  ring: number
  labelX: number
  labelY: number
}

const SPACING: Record<'comfortable' | 'compact', Record<GraphLayoutName, Spacing>> = {
  comfortable: {
    lr: { across: 34, depth: 224, ring: 0, labelX: 190, labelY: 12 },
    tb: { across: 132, depth: 142, ring: 0, labelX: 74, labelY: 44 },
    radial: { across: 0, depth: 0, ring: 168, labelX: 150, labelY: 30 },
  },
  compact: {
    lr: { across: 28, depth: 196, ring: 0, labelX: 170, labelY: 10 },
    tb: { across: 116, depth: 122, ring: 0, labelX: 66, labelY: 38 },
    radial: { across: 0, depth: 0, ring: 144, labelX: 132, labelY: 26 },
  },
}

/**
 * One tree pass, read three ways.
 *
 * The radial case sizes the outer ring from the leaf count as well as from the
 * depth, because a wide tree needs circumference before it needs reach.
 */
export function layoutGraph(
  root: GraphDatum,
  name: GraphLayoutName,
  density: 'comfortable' | 'compact',
): GraphLayoutResult {
  const spacing = SPACING[density][name]
  const laid = hierarchy(root, (datum) => datum.kids)
  const leafTotal = Math.max(1, laid.leaves().length)

  const nodes: PlacedNode[] = []
  const edges: PlacedEdge[] = []

  // The layout call returns the same tree with definite coordinates, which is
  // what makes `x` and `y` readable without a null check per node.
  const positioned =
    name === 'radial'
      ? tree<GraphDatum>()
          .size([
            2 * Math.PI,
            Math.max(
              spacing.ring * Math.max(1, laid.height),
              (leafTotal * 30) / (2 * Math.PI),
            ),
          ])
          .separation((a, b) => (a.parent === b.parent ? 1 : 2) / Math.max(1, a.depth))(laid)
      : tree<GraphDatum>()
          .nodeSize([spacing.across, spacing.depth])
          .separation((a, b) => (a.parent === b.parent ? 1 : 1.5))(laid)

  if (name === 'radial') {
    for (const point of positioned.descendants()) {
      const angle = point.x - Math.PI / 2
      const cos = Math.cos(angle)
      nodes.push({
        key: point.data.key,
        datum: point.data,
        x: cos * point.y,
        y: Math.sin(angle) * point.y,
        side: cos < -0.001 ? 'left' : 'right',
      })
    }
  } else {
    for (const point of positioned.descendants()) {
      nodes.push({
        key: point.data.key,
        datum: point.data,
        x: name === 'lr' ? point.y : point.x,
        y: name === 'lr' ? point.x : point.y,
        side: 'right',
      })
    }
  }

  for (const point of positioned.descendants()) {
    for (const child of point.children ?? []) {
      edges.push({
        id: `${point.data.key}>${child.data.key}`,
        source: point.data.key,
        target: child.data.key,
      })
    }
  }

  const byKey = new Map(nodes.map((node) => [node.key, node]))

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const node of nodes) {
    const r = node.datum.radius
    minX = Math.min(minX, node.x - r - (node.side === 'left' ? spacing.labelX : 0))
    maxX = Math.max(maxX, node.x + r + (node.side === 'right' ? spacing.labelX : 0))
    minY = Math.min(minY, node.y - r - spacing.labelY)
    maxY = Math.max(maxY, node.y + r + spacing.labelY)
  }
  if (!Number.isFinite(minX)) {
    minX = 0
    minY = 0
    maxX = 1
    maxY = 1
  }

  return {
    nodes,
    edges,
    byKey,
    bounds: { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) },
  }
}

/* --- Opening state --------------------------------------------------------- */

/**
 * Which branches start closed.
 *
 * A tree of a few dozen nodes opens whole. Past that the opening view would be
 * an unreadable mat, so everything below the shallowest depth that still fits
 * the budget starts closed and the reader opens what they want.
 */
export function initialCollapsed(index: OntologyIndex, budget = 40): Set<string> {
  const closed = new Set<string>()
  if (index.order.length <= budget) return closed

  const perDepth = new Map<number, string[]>()
  for (const key of index.order) {
    const flat = index.byKey.get(key)
    if (!flat) continue
    const bucket = perDepth.get(flat.node.depth)
    if (bucket) bucket.push(key)
    else perDepth.set(flat.node.depth, [key])
  }

  // Closing at depth `cut` keeps every node up to that depth on screen, so the
  // cut is the last depth whose running total still fits.
  let running = 1
  let cut = index.maxDepth
  for (let depth = 0; depth <= index.maxDepth; depth += 1) {
    running += perDepth.get(depth)?.length ?? 0
    if (running > budget) {
      cut = Math.max(1, depth - 1)
      break
    }
  }
  for (const key of index.order) {
    const flat = index.byKey.get(key)
    if (flat && flat.node.depth >= cut && flat.children.length > 0) closed.add(key)
  }
  return closed
}

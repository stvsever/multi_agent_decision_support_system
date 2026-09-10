/**
 * Graph view.
 *
 * The same hierarchy the other views read, drawn as a node link diagram in one
 * of three presentations. React Flow owns pan, zoom and dragging; d3-hierarchy
 * owns placement; everything expensive is memoised on the visible set, so a
 * wheel tick or a hover costs nothing beyond a transform.
 *
 * Two decisions are worth stating. Positions are interpolated in an animation
 * frame loop rather than by a CSS transition, because the edges are rebuilt
 * from live node positions and would otherwise snap while the nodes glide. And
 * the viewport is driven straight to the bounds of the new content instead of
 * through `fitView`, so the frame and the nodes travel together.
 */

import '@xyflow/react/dist/style.css'
import './graph.css'

import {
  Background,
  BackgroundVariant,
  Handle,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useReactFlow,
  useStore,
  type Edge,
  type EdgeProps,
  type EdgeTypes,
  type Node,
  type NodeProps,
  type NodeTypes,
} from '@xyflow/react'
import {
  Expand,
  Map as MapIcon,
  Maximize2,
  MoveDown,
  MoveRight,
  Orbit,
  Shrink,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import {
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type JSX,
  type KeyboardEvent,
} from 'react'

import { Button, Segmented, Tooltip } from '@/components/ui/primitives'
import { compactNumber, number, signed } from '@/lib/format'
import type { Density } from '@/lib/types'
import { BoxGlyph } from './atoms'
import {
  MEMBERSHIP_LABEL,
  ROOT_KEY,
  bandColor,
  spreadOf,
  type Membership,
  type OntologyIndex,
} from './model'
import {
  buildGraphTree,
  flattenGraphTree,
  layoutGraph,
  subtreeKeys,
  type Bounds,
  type GraphDatum,
  type GraphLayoutName,
} from './graphLayout'

/* --- Policy ---------------------------------------------------------------- */

const MIN_ZOOM = 0.06
const MAX_ZOOM = 2.4
/** Ceiling for an automatic fit, so a three node tree is not blown up. */
const MAX_FIT_ZOOM = 1.15
/** Screen-space margin kept around fitted content. */
const FIT_PAD = 44
/**
 * A tall tree in a short pane fits only at a scale where a node is a smear.
 * Below this the opening view stops being a view, so it holds the floor and
 * anchors on the root instead. Panning and the fit control reveal the rest.
 */
const OPEN_FLOOR = 0.45
const MORPH_MS = 460
const MOVE_MS = 260
/** Past this many edges the links are dropped for the duration of a morph. */
const HEAVY_EDGES = 220
/** A minimap earns its space only once the tree stops fitting comfortably. */
const MINIMAP_FROM = 60

const LAYOUTS: { value: GraphLayoutName; label: JSX.Element; title: string }[] = [
  { value: 'lr', label: <MoveRight size={13} />, title: 'Left to right' },
  { value: 'tb', label: <MoveDown size={13} />, title: 'Top to bottom' },
  { value: 'radial', label: <Orbit size={13} />, title: 'Radial' },
]

const MEMBERSHIP_LEGEND: Membership[] = ['shared', 'common', 'partial', 'unique']
/** The legend is a strip of four, so the long form does not fit. */
const MEMBERSHIP_SHORT: Record<Membership, string> = {
  shared: 'all',
  common: 'most',
  partial: 'some',
  unique: 'one',
}

const EMPTY_EDGES: OntoEdgeType[] = []
const easeOut = (t: number): number => 1 - (1 - t) ** 3
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

/* --- Flow element types ---------------------------------------------------- */

interface OntoNodeData extends Record<string, unknown> {
  datum: GraphDatum
  side: 'left' | 'right'
  cohort: boolean
  dim: boolean
}
type OntoNodeType = Node<OntoNodeData, 'onto'>

interface OntoEdgeData extends Record<string, unknown> {
  shape: GraphLayoutName
  dim: boolean
}
type OntoEdgeType = Edge<OntoEdgeData, 'onto'>

interface Actions {
  toggle: (key: string) => void
}

interface Marks {
  selected: string | null
  /** Where the keyboard is, which is not always something selectable. */
  cursor: string | null
}

const SelectionContext = createContext<Marks>({ selected: null, cursor: null })
const ActionsContext = createContext<Actions>({ toggle: () => undefined })

/* --- Node ------------------------------------------------------------------ */

/**
 * A disc sized by how much evidence hangs beneath it and coloured by the band.
 * The label lives outside the disc and is hidden by CSS below a zoom threshold,
 * which is what keeps a few hundred nodes from turning into overlapping text.
 */
const OntoNode = memo(function OntoNode({ id, data }: NodeProps<OntoNodeType>) {
  const marks = useContext(SelectionContext)
  const { toggle } = useContext(ActionsContext)
  const { datum, side, cohort, dim } = data
  const size = datum.radius * 2
  const spread = cohort ? spreadOf(datum.aggregate) : null
  const isRoot = datum.key === ROOT_KEY
  const branch = !datum.isLeaf || datum.collapsed

  return (
    <div
      className="ographn"
      data-kind={isRoot ? 'root' : datum.isLeaf ? 'leaf' : 'branch'}
      data-side={side}
      data-selected={marks.selected === id || undefined}
      data-cursor={marks.cursor === id || undefined}
      data-collapsed={datum.collapsed || undefined}
      data-membership={datum.membership ?? undefined}
      data-dim={dim || undefined}
      style={{ '--tone': bandColor(datum.band), '--size': `${size}px` } as CSSProperties}
    >
      {/* Both ports sit at the centre so every layout, radial included, links
          disc to disc rather than to an edge that would be wrong off-axis. */}
      <Handle type="target" position={Position.Top} className="ographn__port" isConnectable={false} />
      <Handle type="source" position={Position.Top} className="ographn__port" isConnectable={false} />
      <span className="ographn__disc" aria-hidden="true" />
      {cohort && datum.aggregate && datum.aggregate.coverage < 0.999 && (
        <CoverageRing coverage={datum.aggregate.coverage} size={size} />
      )}
      {branch && !isRoot && (
        <button
          type="button"
          className="ographn__toggle nodrag nopan"
          aria-label={datum.collapsed ? `Expand ${datum.label}` : `Collapse ${datum.label}`}
          onClick={(event) => {
            event.stopPropagation()
            toggle(datum.key)
          }}
        >
          {datum.collapsed ? compactNumber(datum.hidden, 0) : ''}
        </button>
      )}
      <span className="ographn__label">
        <span className="ographn__name">{datum.label}</span>
        <span className="ographn__meta">
          <span className="tabular">{signed(datum.score)}</span>
          {spread && spread.n > 1 && <BoxGlyph spread={spread} clamp={4} width={54} height={9} />}
          {!datum.isLeaf && <span className="tabular">{number(datum.leafCount)} leaves</span>}
        </span>
      </span>
    </div>
  )
})

function CoverageRing({ coverage, size }: { coverage: number; size: number }) {
  const r = size / 2 + 3.5
  const circumference = 2 * Math.PI * r
  return (
    <svg
      className="ographn__ring"
      width={r * 2 + 4}
      height={r * 2 + 4}
      viewBox={`0 0 ${r * 2 + 4} ${r * 2 + 4}`}
      aria-hidden="true"
    >
      <circle
        cx={r + 2}
        cy={r + 2}
        r={r}
        transform={`rotate(-90 ${r + 2} ${r + 2})`}
        strokeDasharray={`${circumference * clamp(coverage, 0, 1)} ${circumference}`}
      />
    </svg>
  )
}

const NODE_TYPES: NodeTypes = { onto: OntoNode }

/* --- Edge ------------------------------------------------------------------ */

/**
 * Radial links are built in polar space, the same construction d3 uses for a
 * radial dendrogram: both control points sit at the mean radius, which makes
 * the link follow the ring before it reaches out. Reading the polar
 * coordinates back from the live endpoints keeps a dragged node's link correct.
 */
function linkPath(shape: GraphLayoutName, sx: number, sy: number, tx: number, ty: number): string {
  if (shape === 'radial') {
    const sourceAngle = Math.atan2(sy, sx)
    const sourceRadius = Math.hypot(sx, sy)
    const targetRadius = Math.hypot(tx, ty)
    let delta = Math.atan2(ty, tx) - sourceAngle
    if (delta > Math.PI) delta -= 2 * Math.PI
    if (delta < -Math.PI) delta += 2 * Math.PI
    const mid = (sourceRadius + targetRadius) / 2
    const c1x = Math.cos(sourceAngle) * mid
    const c1y = Math.sin(sourceAngle) * mid
    const c2x = Math.cos(sourceAngle + delta) * mid
    const c2y = Math.sin(sourceAngle + delta) * mid
    return `M ${sx},${sy} C ${c1x},${c1y} ${c2x},${c2y} ${tx},${ty}`
  }
  if (shape === 'tb') {
    const mid = sy + (ty - sy) * 0.55
    return `M ${sx},${sy} C ${sx},${mid} ${tx},${sy + (ty - sy) * 0.45} ${tx},${ty}`
  }
  const mid = sx + (tx - sx) * 0.55
  return `M ${sx},${sy} C ${mid},${sy} ${sx + (tx - sx) * 0.45},${ty} ${tx},${ty}`
}

function OntoEdge({ sourceX, sourceY, targetX, targetY, data }: EdgeProps<OntoEdgeType>): JSX.Element {
  return (
    <path
      className="ographe"
      data-dim={data?.dim || undefined}
      d={linkPath(data?.shape ?? 'lr', sourceX, sourceY, targetX, targetY)}
    />
  )
}

const EDGE_TYPES: EdgeTypes = { onto: OntoEdge }

/* --- Canvas ---------------------------------------------------------------- */

export interface GraphViewProps {
  index: OntologyIndex
  keep: Set<string> | null
  matchBranch: Set<string> | null
  rootLabel: string
  cohort: boolean
  selected: string | null
  onSelect: (key: string) => void
  onClear: () => void
  collapsed: Set<string>
  onCollapsedChange: (next: Set<string>) => void
  layout: GraphLayoutName
  onLayoutChange: (next: GraphLayoutName) => void
  density: Density
  reducedMotion: boolean
}

function GraphCanvas(props: GraphViewProps): JSX.Element {
  const {
    index,
    keep,
    matchBranch,
    rootLabel,
    cohort,
    selected,
    onSelect,
    onClear,
    collapsed,
    onCollapsedChange,
    layout,
    onLayoutChange,
    density,
    reducedMotion,
  } = props

  const { setViewport, setCenter, zoomIn, zoomOut } = useReactFlow()
  const paneW = useStore((state) => state.width)
  const paneH = useStore((state) => state.height)
  const zoomPct = useStore((state) => Math.round(state.transform[2] * 100))
  const zoom = zoomPct / 100

  const [minimap, setMinimap] = useState(true)
  const [morphing, setMorphing] = useState(false)
  const [opened, setOpened] = useState(false)
  const [cursor, setCursor] = useState<string | null>(null)

  const tree = useMemo(
    () => buildGraphTree(index, keep, collapsed, rootLabel),
    [index, keep, collapsed, rootLabel],
  )
  const byKey = useMemo(() => flattenGraphTree(tree), [tree])
  const placement = useMemo(() => layoutGraph(tree, layout, density), [tree, layout, density])

  const flowNodes = useMemo<OntoNodeType[]>(
    () =>
      placement.nodes.map((placed) => {
        const size = placed.datum.radius * 2
        return {
          id: placed.key,
          type: 'onto' as const,
          position: { x: placed.x - placed.datum.radius, y: placed.y - placed.datum.radius },
          width: size,
          height: size,
          // Explicit dimensions mean the first fit can run before the DOM has
          // been measured, so the graph never opens at the default transform.
          style: { width: size, height: size },
          data: {
            datum: placed.datum,
            side: placed.side,
            cohort,
            dim: matchBranch ? !matchBranch.has(placed.key) : false,
          },
        }
      }),
    [placement, cohort, matchBranch],
  )

  const flowEdges = useMemo<OntoEdgeType[]>(
    () =>
      placement.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: 'onto' as const,
        data: {
          shape: layout,
          dim: matchBranch ? !matchBranch.has(edge.target) : false,
        },
      })),
    [placement, layout, matchBranch],
  )

  const [nodes, setNodes, onNodesChange] = useNodesState<OntoNodeType>([])
  const nodesRef = useRef<OntoNodeType[]>(nodes)
  nodesRef.current = nodes

  const boundsRef = useRef<Bounds>(placement.bounds)
  boundsRef.current = placement.bounds
  const byKeyRef = useRef(byKey)
  byKeyRef.current = byKey

  /* --- Viewport ------------------------------------------------------------ */

  const fitTo = useCallback(
    (bounds: Bounds, duration: number, anchored = true) => {
      if (paneW < 40 || paneH < 40) return
      const raw = Math.min(
        (paneW - FIT_PAD * 2) / bounds.width,
        (paneH - FIT_PAD * 2) / bounds.height,
      )
      const held = anchored && raw < OPEN_FLOOR
      const scale = clamp(held ? OPEN_FLOOR : raw, MIN_ZOOM, MAX_FIT_ZOOM)
      const cx = bounds.x + bounds.width / 2
      const cy = bounds.y + bounds.height / 2
      let x = paneW / 2 - cx * scale
      let y = paneH / 2 - cy * scale
      // Held back, the frame lands on where the tree starts rather than on the
      // middle of a mass the reader cannot see the edges of anyway.
      if (held && layout === 'lr') x = FIT_PAD - bounds.x * scale
      if (held && layout === 'tb') y = FIT_PAD - bounds.y * scale
      void setViewport({ x, y, zoom: scale }, { duration })
    },
    [paneW, paneH, setViewport, layout],
  )
  const fitRef = useRef(fitTo)
  fitRef.current = fitTo

  const boundsOf = useCallback((keys: Iterable<string>): Bounds | null => {
    const wanted = new Set(keys)
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    for (const node of nodesRef.current) {
      if (!wanted.has(node.id)) continue
      const size = node.width ?? 12
      minX = Math.min(minX, node.position.x)
      minY = Math.min(minY, node.position.y)
      maxX = Math.max(maxX, node.position.x + size)
      maxY = Math.max(maxY, node.position.y + size)
    }
    if (!Number.isFinite(minX)) return null
    const pad = 60
    return {
      x: minX - pad,
      y: minY - pad,
      width: maxX - minX + pad * 2,
      height: maxY - minY + pad * 2,
    }
  }, [])

  /** The pane is measured a tick after mount, which is when the first fit runs. */
  const openedRef = useRef(false)
  useEffect(() => {
    if (openedRef.current || paneW < 40 || paneH < 40 || nodes.length === 0) return
    openedRef.current = true
    setOpened(true)
    fitRef.current(boundsRef.current, 0)
  }, [paneW, paneH, nodes.length])

  const rafRef = useRef(0)
  useEffect(() => {
    cancelAnimationFrame(rafRef.current)
    const target = flowNodes
    const live = nodesRef.current
    const from = new Map(live.map((node) => [node.id, node.position]))

    // A filter that only changes dimming leaves every position alone, so it
    // must not trigger a transition or a refit.
    const settled =
      live.length === target.length &&
      target.every((node) => {
        const previous = from.get(node.id)
        return (
          previous !== undefined &&
          Math.abs(previous.x - node.position.x) < 0.5 &&
          Math.abs(previous.y - node.position.y) < 0.5
        )
      })
    if (settled) {
      setNodes(target)
      return
    }

    if (!openedRef.current || reducedMotion || from.size === 0) {
      setNodes(target)
      if (openedRef.current) fitRef.current(boundsRef.current, 0)
      return
    }

    // A node revealed by an expand starts at its nearest visible ancestor, so
    // the subtree grows out of the branch that produced it.
    const start = new Map<string, { x: number; y: number }>()
    for (const node of target) {
      const own = from.get(node.id)
      if (own) {
        start.set(node.id, own)
        continue
      }
      let parent = node.data.datum.parent
      let anchor: { x: number; y: number } | undefined
      while (parent) {
        anchor = from.get(parent)
        if (anchor) break
        parent = byKeyRef.current.get(parent)?.parent ?? null
      }
      start.set(node.id, anchor ?? node.position)
    }

    setMorphing(true)
    fitRef.current(boundsRef.current, MORPH_MS)
    const began = performance.now()
    const step = (now: number) => {
      const t = Math.min(1, (now - began) / MORPH_MS)
      if (t >= 1) {
        setNodes(target)
        setMorphing(false)
        rafRef.current = 0
        return
      }
      const eased = easeOut(t)
      setNodes(
        target.map((node) => {
          const origin = start.get(node.id)
          if (!origin) return node
          return {
            ...node,
            position: {
              x: origin.x + (node.position.x - origin.x) * eased,
              y: origin.y + (node.position.y - origin.y) * eased,
            },
          }
        }),
      )
      rafRef.current = requestAnimationFrame(step)
    }
    rafRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(rafRef.current)
  }, [flowNodes, reducedMotion, setNodes])

  useEffect(() => () => cancelAnimationFrame(rafRef.current), [])

  /* --- Interaction --------------------------------------------------------- */

  const toggle = useCallback(
    (key: string) => {
      const next = new Set(collapsed)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      onCollapsedChange(next)
    },
    [collapsed, onCollapsedChange],
  )
  const actions = useMemo<Actions>(() => ({ toggle }), [toggle])
  const marks = useMemo<Marks>(() => ({ selected, cursor }), [selected, cursor])

  const pick = useCallback(
    (key: string) => {
      setCursor(key)
      if (key !== ROOT_KEY) onSelect(key)
    },
    [onSelect],
  )

  const focusSubtree = useCallback(
    (key: string) => {
      const datum = byKeyRef.current.get(key)
      if (!datum) return
      const bounds = boundsOf(subtreeKeys(datum))
      if (bounds) fitRef.current(bounds, reducedMotion ? 0 : MORPH_MS)
    },
    [boundsOf, reducedMotion],
  )

  const centreOn = useCallback(
    (key: string) => {
      const placed = placement.byKey.get(key)
      if (!placed) return
      void setCenter(placed.x, placed.y, {
        zoom: Math.max(zoom, 0.5),
        duration: reducedMotion ? 0 : MOVE_MS,
      })
    },
    [placement, setCenter, zoom, reducedMotion],
  )

  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const key = event.key
      if (key === 'Escape') {
        setCursor(null)
        onClear()
        return
      }
      const current = cursor ?? selected ?? ROOT_KEY
      const datum = byKey.get(current)
      if (!datum) return

      const vertical = layout !== 'tb'
      const siblingKeys = vertical ? ['ArrowUp', 'ArrowDown'] : ['ArrowLeft', 'ArrowRight']
      const parentKey = vertical ? 'ArrowLeft' : 'ArrowUp'
      const childKey = vertical ? 'ArrowRight' : 'ArrowDown'

      let next: string | null = null
      if (siblingKeys.includes(key)) {
        const parent = datum.parent ? byKey.get(datum.parent) : null
        const family = parent ? parent.kids : [datum]
        const at = family.findIndex((kid) => kid.key === datum.key)
        const step = key === 'ArrowUp' || key === 'ArrowLeft' ? -1 : 1
        next = family[at + step]?.key ?? null
      } else if (key === parentKey) {
        next = datum.parent
      } else if (key === childKey) {
        if (datum.collapsed) {
          event.preventDefault()
          toggle(datum.key)
          return
        }
        next = datum.kids[0]?.key ?? null
      } else if (key === 'Enter' || key === ' ') {
        event.preventDefault()
        if (!datum.isLeaf || datum.collapsed) toggle(datum.key)
        else pick(datum.key)
        return
      } else if (key === 'Home') {
        next = ROOT_KEY
      } else if (key === 'End') {
        next = byKey.get(ROOT_KEY)?.kids[0]?.key ?? null
      } else {
        return
      }

      event.preventDefault()
      if (!next) return
      pick(next)
      centreOn(next)
    },
    [byKey, centreOn, cursor, layout, onClear, pick, selected, toggle],
  )

  const collapseAll = useCallback(() => {
    if (collapsed.size > 0) {
      onCollapsedChange(new Set())
      return
    }
    const next = new Set<string>()
    for (const [key, datum] of byKey) {
      if (key !== ROOT_KEY && datum.kids.length > 0 && datum.depth === 1) next.add(key)
    }
    onCollapsedChange(next)
  }, [byKey, collapsed, onCollapsedChange])

  /* --- Chrome -------------------------------------------------------------- */

  // The tiers sit below the opening floor on purpose, so a view that had to be
  // held back still names its branches.
  const labels = zoom < 0.3 ? 'off' : zoom < 0.62 ? 'branches' : 'all'
  const hidden = useMemo(() => {
    let total = 0
    for (const datum of byKey.values()) total += datum.hidden
    return total
  }, [byKey])
  const edges = morphing && flowEdges.length > HEAVY_EDGES ? EMPTY_EDGES : flowEdges

  return (
    <div
      className="ograph"
      data-layout={layout}
      data-labels={labels}
      data-cohort={cohort || undefined}
      data-opened={opened}
      tabIndex={0}
      role="application"
      aria-label="Ontology graph"
      onKeyDown={onKeyDown}
    >
      <SelectionContext.Provider value={marks}>
        <ActionsContext.Provider value={actions}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            minZoom={MIN_ZOOM}
            maxZoom={MAX_ZOOM}
            nodesConnectable={false}
            nodesFocusable={false}
            edgesFocusable={false}
            elementsSelectable={false}
            zoomOnDoubleClick={false}
            disableKeyboardA11y
            onNodeClick={(_event, node) => pick(node.id)}
            onNodeDoubleClick={(_event, node) => {
              pick(node.id)
              focusSubtree(node.id)
            }}
            onPaneClick={() => {
              setCursor(null)
              onClear()
            }}
          >
            <Background variant={BackgroundVariant.Dots} gap={24} size={1} />

            <Panel position="top-left" className="ograph-hint t-micro faint">
              Drag to pan, wheel to zoom. Double click frames a branch, arrow keys walk the tree.
            </Panel>

            <Panel position="top-right" className="ograph-controls nopan nowheel">
              <Segmented
                value={layout}
                options={LAYOUTS}
                onChange={onLayoutChange}
                size="sm"
              />
              <div className="ograph-controls__cluster">
                <Tooltip content="Zoom out">
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={<ZoomOut size={14} />}
                    aria-label="Zoom out"
                    onClick={() => void zoomOut({ duration: reducedMotion ? 0 : 180 })}
                  />
                </Tooltip>
                <span className="ograph-controls__zoom tabular">{zoomPct}%</span>
                <Tooltip content="Zoom in">
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={<ZoomIn size={14} />}
                    aria-label="Zoom in"
                    onClick={() => void zoomIn({ duration: reducedMotion ? 0 : 180 })}
                  />
                </Tooltip>
                <span className="ograph-controls__divider" />
                <Tooltip content="Fit the whole tree">
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={<Maximize2 size={14} />}
                    aria-label="Fit the whole tree"
                    onClick={() => fitRef.current(boundsRef.current, reducedMotion ? 0 : MORPH_MS, false)}
                  />
                </Tooltip>
                <Tooltip content={collapsed.size > 0 ? 'Open every branch' : 'Close every branch'}>
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={collapsed.size > 0 ? <Expand size={14} /> : <Shrink size={14} />}
                    aria-label={collapsed.size > 0 ? 'Open every branch' : 'Close every branch'}
                    onClick={collapseAll}
                  />
                </Tooltip>
                {flowNodes.length > MINIMAP_FROM && (
                  <Tooltip content={minimap ? 'Hide the minimap' : 'Show the minimap'}>
                    <Button
                      variant="ghost"
                      size="sm"
                      iconOnly
                      icon={<MapIcon size={14} />}
                      aria-label="Toggle the minimap"
                      aria-pressed={minimap}
                      onClick={() => setMinimap((value) => !value)}
                    />
                  </Tooltip>
                )}
              </div>
            </Panel>

            <Panel position="bottom-left" className="ograph-foot nopan nowheel">
              <span className="ograph-foot__row">
                <b>{number(flowNodes.length - 1)}</b> shown
                {hidden > 0 && (
                  <>
                    <span className="ograph-foot__sep" />
                    <b>{number(hidden)}</b> closed away
                  </>
                )}
              </span>
              {cohort && (
                <span className="ograph-foot__legend">
                  {MEMBERSHIP_LEGEND.map((membership) => (
                    <span
                      key={membership}
                      className="ograph-foot__key"
                      data-membership={membership}
                      title={MEMBERSHIP_LABEL[membership]}
                    >
                      <span className="ograph-foot__swatch" aria-hidden="true" />
                      {MEMBERSHIP_SHORT[membership]}
                    </span>
                  ))}
                </span>
              )}
            </Panel>

            {minimap && flowNodes.length > MINIMAP_FROM && (
              <MiniMap
                className="nopan"
                pannable
                zoomable
                ariaLabel="Tree overview"
                nodeStrokeWidth={0}
                nodeBorderRadius={20}
                maskColor="color-mix(in srgb, var(--bg-sunken) 72%, transparent)"
                bgColor="var(--bg-elevated)"
                nodeColor={(node) => {
                  const data = node.data as OntoNodeData | undefined
                  return bandColor(data?.datum.band ?? 'missing')
                }}
              />
            )}
          </ReactFlow>
        </ActionsContext.Provider>
      </SelectionContext.Provider>

      <span className="sr-only" aria-live="polite">
        {selected ? `${byKey.get(selected)?.label ?? ''} selected` : ''}
      </span>
    </div>
  )
}

export function GraphView(props: GraphViewProps): JSX.Element {
  return (
    <ReactFlowProvider>
      <GraphCanvas {...props} />
    </ReactFlowProvider>
  )
}

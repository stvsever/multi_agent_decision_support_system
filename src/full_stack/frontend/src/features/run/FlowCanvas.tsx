/**
 * Live orchestration flow canvas.
 *
 * The service ranks plan steps by dependency depth, so one rank is literally
 * one concurrent wave of tool calls and the lane index is the position inside
 * that wave. Layout therefore reads straight off `rank` and `lane` instead of
 * running a general graph layout pass, which keeps the parallel structure of
 * the plan legible and stable across iterations.
 */

import '@xyflow/react/dist/style.css'
import './flow.css'

import {
  Background,
  BackgroundVariant,
  EdgeLabelRenderer,
  Handle,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  ReactFlowProvider,
  ViewportPortal,
  getBezierPath,
  getSmoothStepPath,
  getStraightPath,
  useReactFlow,
  useStore,
  type Edge,
  type EdgeProps,
  type EdgeTypes,
  type Node,
  type NodeProps,
  type NodeTypes,
  type Viewport,
} from '@xyflow/react'
import clsx from 'clsx'
import {
  ArrowDownUp,
  ArrowRightLeft,
  Check,
  Compass,
  FileText,
  Layers,
  List,
  LoaderCircle,
  Maximize2,
  Merge,
  Scan,
  ShieldCheck,
  Target,
  Waypoints,
  Wrench,
  X,
  Zap,
  ZapOff,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type JSX,
  type ReactNode,
} from 'react'

import { Badge, Button, Card, EmptyState, Tooltip } from '@/components/ui/primitives'
import { duration as formatDuration, tokens as formatTokens, titleCase } from '@/lib/format'
import type { GraphEdge, GraphNode, RunGraph, RunStep } from '@/lib/types'

/* --- Contract -------------------------------------------------------------- */

export interface FlowCanvasProps {
  graph: RunGraph | null
  steps: RunStep[]
  currentStage: number
  iteration: number
  running: boolean
  animate?: boolean
  direction?: 'LR' | 'TB'
  edgeStyle?: 'bezier' | 'smoothstep' | 'straight'
  showTokens?: boolean
  onSelectNode?: (node: GraphNode | null) => void
  className?: string
}

type Direction = 'LR' | 'TB'
type EdgeStyleName = 'bezier' | 'smoothstep' | 'straight'
type FlowStatus = 'pending' | 'running' | 'complete' | 'failed' | 'repairing'

/* --- Geometry -------------------------------------------------------------- */

const NODE_W = 196
const NODE_H = 62
const AGENT_W = 168
const AGENT_H = 56

/** Distance between rank columns (LR) or rows (TB), node size included. */
const RANK_STEP: Record<Direction, number> = { LR: NODE_W + 92, TB: NODE_H + 78 }
/** Distance between lanes inside one rank. */
const LANE_STEP: Record<Direction, number> = { LR: NODE_H + 34, TB: NODE_W + 40 }

const BAND_PAD = 24
const RULER_LR = { gap: 12, h: 32 }
const RULER_TB = { gap: 16, w: 116 }
const DETOUR_GAP = 52

/* --- Viewport policy ------------------------------------------------------- */

const MIN_ZOOM = 0.14
const MAX_ZOOM = 1.9
/**
 * A wide plan in a short pane fits only at a scale where a tool node is a
 * smear of about forty pixels. Below this the canvas stops being a canvas, so
 * the opening view refuses to go lower and anchors on the start of the run
 * instead. Panning and the fit control reveal the rest.
 */
const MIN_OPEN_ZOOM = 0.55
/** Ceiling for the opening view, so a two node plan is not blown up. */
const MAX_OPEN_ZOOM = 1.15
/** Fraction of the pane left clear on each side when framing bounds. */
const FIT_INSET = 0.05
/** Screen-space gap kept before the leading edge of an anchored view. */
const OPEN_MARGIN = 26
/** How far into the pane the active rank is placed when following the run. */
const FOLLOW_LEAD = 0.36
/** Slack around the active rank before a follow pan is considered worthwhile. */
const FOLLOW_INSET = 36
const OPEN_MS = 320
const FOLLOW_MS = 520
/** One tick, so the pan and zoom instance exists before the opening view. */
const OPEN_DELAY = 32
/** Backstop that reveals the surface even if the pane is never measured. */
const REVEAL_MS = 400

type FitScope = 'content' | 'full'

/* --- Static maps ----------------------------------------------------------- */

const AGENT_ICONS: Record<string, typeof Compass> = {
  orchestrator: Compass,
  integrator: Merge,
  predictor: Target,
  critic: ShieldCheck,
  communicator: FileText,
}

const STATUS_LABEL: Record<FlowStatus, string> = {
  pending: 'queued',
  running: 'running',
  complete: 'done',
  failed: 'failed',
  repairing: 'repair',
}

const EDGE_LEGEND: { kind: GraphEdge['kind']; body: string }[] = [
  { kind: 'sequential', body: 'one step feeds the next' },
  { kind: 'fan_out', body: 'one result opens several steps' },
  { kind: 'fan_in', body: 'several results join one step' },
  { kind: 'mesh', body: 'many to many between waves' },
  { kind: 'feedback', body: 'critic sends the plan back' },
]

const STATUS_LEGEND: { status: FlowStatus; body: string }[] = [
  { status: 'pending', body: 'not dispatched yet' },
  { status: 'running', body: 'executing now' },
  { status: 'complete', body: 'finished, tokens recorded' },
  { status: 'repairing', body: 'output rejected, retrying' },
  { status: 'failed', body: 'gave up on this step' },
]

/** Travelling-dot cycle length per edge kind, in seconds. */
const DOT_DUR: Record<GraphEdge['kind'], number> = {
  sequential: 2.4,
  fan_out: 2.6,
  fan_in: 2.2,
  mesh: 3,
  feedback: 3.4,
}
const DOT_DUR_ACTIVE = 1.1
const DOT_R = 2.6
/** Direction marker used when animation is off. */
const TIP_PATH = 'M -3.2,-3 L 3.2,0 L -3.2,3 Z'

/* --- Small helpers --------------------------------------------------------- */

/** SVG ids have to survive being used as a URL fragment in `mpath href`. */
const slug = (value: string) => value.replace(/[^A-Za-z0-9_-]/g, '_')

const familyVar = (node: GraphNode) =>
  node.type === 'agent' ? 'var(--fam-agent)' : `var(--fam-${node.family || 'other'})`

function planStatus(raw: string | undefined): FlowStatus {
  const value = String(raw ?? '').toLowerCase()
  if (value === 'complete' || value === 'completed' || value === 'success') return 'complete'
  if (value === 'failed' || value === 'error') return 'failed'
  if (value === 'running' || value === 'in_progress') return 'running'
  if (value === 'repairing') return 'repairing'
  return 'pending'
}

function agentStatus(stage: number, currentStage: number, running: boolean): FlowStatus {
  if (currentStage < 0) return 'pending'
  if (stage < currentStage) return 'complete'
  if (stage > currentStage) return 'pending'
  return running ? 'running' : 'complete'
}

/** Reads the appearance layer's motion switch and follows later changes to it. */
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof document !== 'undefined' && document.documentElement.dataset.motion === 'reduced',
  )
  useEffect(() => {
    const root = document.documentElement
    const read = () => setReduced(root.dataset.motion === 'reduced')
    read()
    const observer = new MutationObserver(read)
    observer.observe(root, { attributes: true, attributeFilter: ['data-motion'] })
    return () => observer.disconnect()
  }, [])
  return reduced
}

/* --- Layout ---------------------------------------------------------------- */

interface Placed {
  node: GraphNode
  x: number
  y: number
  w: number
  h: number
}

interface Box {
  x: number
  y: number
  w: number
  h: number
}

/** Graph-space rectangle in the shape React Flow's viewport helpers expect. */
interface Rect {
  x: number
  y: number
  width: number
  height: number
}

const clamp = (value: number, low: number, high: number) => Math.min(Math.max(value, low), high)

function union(boxes: Box[], pad: number): Rect {
  if (!boxes.length) return { x: 0, y: 0, width: 1, height: 1 }
  const left = Math.min(...boxes.map((b) => b.x)) - pad
  const top = Math.min(...boxes.map((b) => b.y)) - pad
  const right = Math.max(...boxes.map((b) => b.x + b.w)) + pad
  const bottom = Math.max(...boxes.map((b) => b.y + b.h)) + pad
  return { x: left, y: top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) }
}

/** Zoom at which `rect` is framed by a `w` by `h` pane, inset on every side. */
function fitZoom(rect: Rect, w: number, h: number): number {
  const usable = 1 - FIT_INSET * 2
  return Math.min((w * usable) / rect.width, (h * usable) / rect.height)
}

function centred(rect: Rect, w: number, h: number, zoom: number): Viewport {
  return {
    x: w / 2 - (rect.x + rect.width / 2) * zoom,
    y: h / 2 - (rect.y + rect.height / 2) * zoom,
    zoom,
  }
}

/** Transition length for a viewport move, zero when motion is turned down. */
const glide = (animated: boolean, reduced: boolean, ms: number) => (animated && !reduced ? ms : 0)

interface RankInfo {
  rank: number
  label: string
  sub: string
  parallel: boolean
  band: Box
  ruler: Box
}

interface Layout {
  placed: Placed[]
  ranks: RankInfo[]
  direction: Direction
  /** Absolute coordinate of the lane the feedback edge routes along. */
  detour: number
  /**
   * Nodes, their bands, and the forward edges between them. The critic detour
   * runs a long way past the last lane, so including it in the opening view
   * would cost most of the zoom for a lane that carries one dashed line.
   */
  content: Rect
  /** Everything drawn, ruler and feedback lane included, for exact fitting. */
  full: Rect
  key: string
}

function layoutGraph(graph: RunGraph, direction: Direction): Layout {
  const ranks = Array.from(new Set(graph.nodes.map((n) => n.rank))).sort((a, b) => a - b)
  const rankIndex = new Map(ranks.map((rank, index) => [rank, index]))
  const laneCount = new Map<number, number>()
  for (const node of graph.nodes) {
    laneCount.set(node.rank, Math.max(laneCount.get(node.rank) ?? 0, node.lane + 1))
  }

  const step = RANK_STEP[direction]
  const lane = LANE_STEP[direction]

  const placed: Placed[] = graph.nodes.map((node) => {
    const agent = node.type === 'agent'
    const w = agent ? AGENT_W : NODE_W
    const h = agent ? AGENT_H : NODE_H
    const along = (rankIndex.get(node.rank) ?? 0) * step
    // Lanes are centred on zero so a wave of seven grows symmetrically around
    // the spine instead of pushing every later rank downwards.
    const count = laneCount.get(node.rank) ?? 1
    const across = (node.lane - (count - 1) / 2) * lane
    if (direction === 'LR') return { node, x: along + (NODE_W - w) / 2, y: across - h / 2, w, h }
    return { node, x: across - w / 2, y: along + (NODE_H - h) / 2, w, h }
  })

  const crossMin = placed.length
    ? Math.min(...placed.map((p) => (direction === 'LR' ? p.y : p.x)))
    : 0
  const crossMax = placed.length
    ? Math.max(...placed.map((p) => (direction === 'LR' ? p.y + p.h : p.x + p.w)))
    : 0
  const bandStart = crossMin - BAND_PAD
  const bandSpan = crossMax - crossMin + BAND_PAD * 2

  const infos: RankInfo[] = []
  let wave = 0
  for (const rank of ranks) {
    const here = graph.nodes.filter((n) => n.rank === rank)
    const tools = here.filter((n) => n.type === 'tool')
    const agent = here.find((n) => n.type === 'agent')
    const parallel = tools.length > 1
    let label: string
    let sub: string
    if (agent) {
      label = `Stage ${agent.stage}`
      sub = agent.label
    } else {
      wave += 1
      label = `Wave ${wave}`
      sub = tools.length > 1 ? `${tools.length} parallel` : '1 step'
    }
    const along = (rankIndex.get(rank) ?? 0) * step

    if (direction === 'LR') {
      const bandX = along - BAND_PAD / 2
      const bandW = NODE_W + BAND_PAD
      infos.push({
        rank,
        label,
        sub,
        parallel,
        band: { x: bandX, y: bandStart, w: bandW, h: bandSpan },
        ruler: { x: bandX, y: bandStart - RULER_LR.gap - RULER_LR.h, w: bandW, h: RULER_LR.h },
      })
    } else {
      const bandY = along - BAND_PAD / 2
      const bandH = NODE_H + BAND_PAD
      infos.push({
        rank,
        label,
        sub,
        parallel,
        band: { x: bandStart, y: bandY, w: bandSpan, h: bandH },
        ruler: { x: bandStart - RULER_TB.gap - RULER_TB.w, y: bandY, w: RULER_TB.w, h: bandH },
      })
    }
  }

  const detour = crossMax + BAND_PAD + DETOUR_GAP
  // The ruler, the bands, and the feedback lane all sit outside the node set,
  // so the fit has to work from an explicit union rather than from fitView.
  const nodeBoxes: Box[] = placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h }))
  // Rank zero starts at the origin, so a zero on the other axis is always
  // inside the graph and only the detour side of this box matters.
  const detourBox: Box =
    direction === 'LR' ? { x: 0, y: detour, w: 1, h: BAND_PAD } : { x: detour, y: 0, w: BAND_PAD, h: 1 }

  return {
    placed,
    ranks: infos,
    direction,
    detour,
    // A band never reaches further than BAND_PAD past its nodes, and a forward
    // edge stays between the two ranks it joins, so the node union plus one
    // pad is the whole forward picture.
    content: union(nodeBoxes, BAND_PAD),
    full: union([...nodeBoxes, ...infos.map((i) => i.band), ...infos.map((i) => i.ruler), detourBox], 0),
    key: `${graph.plan_id}|${graph.iteration}|${graph.nodes.length}|${graph.edges.length}|${direction}`,
  }
}

/* --- Canvas contexts ------------------------------------------------------- */

interface UiValue {
  direction: Direction
  showTokens: boolean
  selectedId: string | null
  select: (node: GraphNode) => void
}

const UiContext = createContext<UiValue>({
  direction: 'LR',
  showTokens: true,
  selectedId: null,
  select: () => {},
})

/**
 * Live step readings live in their own context so a token tick re-renders the
 * two or three node bodies that changed rather than rebuilding the graph.
 */
const LiveContext = createContext<Map<number, RunStep>>(new Map())

/* --- Node components ------------------------------------------------------- */

interface FlowNodeData extends Record<string, unknown> {
  node: GraphNode
  status: FlowStatus
}

type ToolNodeType = Node<FlowNodeData, 'tool'>
type AgentNodeType = Node<FlowNodeData, 'agent'>
type CanvasNode = ToolNodeType | AgentNodeType

function statusMark(status: FlowStatus): ReactNode {
  if (status === 'running') return <LoaderCircle size={11} className="spin" />
  if (status === 'complete') return <Check size={11} />
  if (status === 'failed') return <X size={11} />
  if (status === 'repairing') return <Wrench size={11} />
  return null
}

/**
 * Pointer selection runs through React Flow's `onNodeClick`, because the node
 * wrapper only receives pointer events when React Flow itself has a handler.
 * Keyboard activation has to be wired here instead.
 */
function useNodeShell(node: GraphNode) {
  const { direction, selectedId, select } = useContext(UiContext)
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key !== 'Enter' && event.key !== ' ') return
      event.preventDefault()
      select(node)
    },
    [select, node],
  )
  return {
    direction,
    targetPosition: direction === 'LR' ? Position.Left : Position.Top,
    sourcePosition: direction === 'LR' ? Position.Right : Position.Bottom,
    selected: selectedId === node.id,
    onKeyDown,
  }
}

function ToolNode({ data }: NodeProps<ToolNodeType>) {
  const { node, status } = data
  const { showTokens } = useContext(UiContext)
  const live = useContext(LiveContext)
  const { targetPosition, sourcePosition, selected, onKeyDown } = useNodeShell(node)
  const step = node.step_id === undefined ? undefined : live.get(node.step_id)
  const estimated = Number(node.meta?.estimated_tokens ?? 0)
  const tokenCount = step?.tokens ?? (status === 'pending' ? estimated : 0)

  return (
    <div
      className="flow-node flow-node--tool"
      data-status={status}
      data-family={node.family || 'other'}
      data-selected={selected}
      role="button"
      tabIndex={0}
      title={node.detail || node.label}
      style={{ '--fam': familyVar(node), '--st': `var(--st-${status})` } as React.CSSProperties}
      onKeyDown={onKeyDown}
    >
      <Handle type="target" id="in" position={targetPosition} isConnectable={false} />
      <span className="flow-node__rail" />
      <div className="flow-node__body">
        <div className="flow-node__head">
          <span className="flow-node__dot" />
          <span className="flow-node__title truncate">{node.label || 'Tool'}</span>
          <span className="flow-node__mark">{statusMark(status)}</span>
        </div>
        <span className="flow-node__desc">{node.detail || 'No description supplied.'}</span>
      </div>
      <div className="flow-node__foot">
        <span className="flow-node__state">{STATUS_LABEL[status]}</span>
        {showTokens && tokenCount > 0 && (
          <span className="flow-node__stat flow-node__stat--tokens">
            {formatTokens(tokenCount)}
            {status === 'pending' ? ' est' : ''}
          </span>
        )}
        {step?.duration ? <span className="flow-node__stat">{formatDuration(step.duration)}</span> : null}
      </div>
      <Handle type="source" id="out" position={sourcePosition} isConnectable={false} />
    </div>
  )
}

function AgentNode({ data }: NodeProps<AgentNodeType>) {
  const { node, status } = data
  const { direction, targetPosition, sourcePosition, selected, onKeyDown } = useNodeShell(node)
  const Icon = AGENT_ICONS[node.role ?? ''] ?? Waypoints
  // The feedback loop leaves and re-enters on the cross axis so it can be
  // routed clear of the forward path.
  const feedbackPosition = direction === 'LR' ? Position.Bottom : Position.Right

  return (
    <div
      className="flow-node flow-node--agent"
      data-status={status}
      data-role={node.role}
      data-selected={selected}
      role="button"
      tabIndex={0}
      title={node.detail || node.label}
      style={{ '--fam': 'var(--fam-agent)', '--st': `var(--st-${status})` } as React.CSSProperties}
      onKeyDown={onKeyDown}
    >
      <Handle type="target" id="in" position={targetPosition} isConnectable={false} />
      <Handle type="target" id="fb-in" position={feedbackPosition} isConnectable={false} />
      <span className="flow-node__icon">
        {status === 'running' ? <LoaderCircle size={14} className="spin" /> : <Icon size={14} />}
      </span>
      <span className="flow-node__agent-text">
        <span className="flow-node__agent-label truncate">{node.label}</span>
        <span className="flow-node__agent-sub truncate">
          {STATUS_LABEL[status]} · stage {node.stage}
        </span>
      </span>
      <Handle type="source" id="out" position={sourcePosition} isConnectable={false} />
      <Handle type="source" id="fb-out" position={feedbackPosition} isConnectable={false} />
    </div>
  )
}

const NODE_TYPES: NodeTypes = { tool: ToolNode, agent: AgentNode }

/* --- Edge components ------------------------------------------------------- */

interface FlowEdgeData extends Record<string, unknown> {
  kind: GraphEdge['kind']
  index: number
  active: boolean
  animate: boolean
  shape: EdgeStyleName
  direction: Direction
  detour: number
  from: string
  to: string
  label?: string
}

type CanvasEdge = Edge<FlowEdgeData, 'flow' | 'feedback'>

interface PathInput {
  sourceX: number
  sourceY: number
  targetX: number
  targetY: number
  sourcePosition: Position
  targetPosition: Position
}

function shapePath(shape: EdgeStyleName, input: PathInput): [string, number, number] {
  if (shape === 'straight') {
    const [d, x, y] = getStraightPath({
      sourceX: input.sourceX,
      sourceY: input.sourceY,
      targetX: input.targetX,
      targetY: input.targetY,
    })
    return [d, x, y]
  }
  if (shape === 'smoothstep') {
    const [d, x, y] = getSmoothStepPath({ ...input, borderRadius: 14 })
    return [d, x, y]
  }
  const [d, x, y] = getBezierPath({ ...input, curvature: 0.34 })
  return [d, x, y]
}

/**
 * A dot bound to the path with `<animateMotion><mpath href="#id">`. Referencing
 * the rendered path rather than re-deriving the geometry means the dot tracks
 * whichever edge shape is in use without a second implementation.
 */
function MotionDot({
  pathId,
  dur,
  begin,
  className,
}: {
  pathId: string
  dur: number
  begin: number
  className: string
}) {
  return (
    // A dur change has to remount for SMIL to pick it up, hence the key.
    <circle key={`${dur}-${begin}`} r={DOT_R} className={className}>
      <animateMotion dur={`${dur}s`} begin={`${begin}s`} repeatCount="indefinite">
        <mpath href={`#${pathId}`} />
      </animateMotion>
    </circle>
  )
}

function StaticDot({ x, y, angle }: { x: number; y: number; angle: number }) {
  return (
    <>
      <circle cx={x} cy={y} r={DOT_R} className="flow-edge__dot" />
      <path
        className="flow-edge__tip"
        d={TIP_PATH}
        transform={`translate(${x} ${y}) rotate(${angle}) translate(9 0)`}
      />
    </>
  )
}

function FlowEdge(props: EdgeProps<CanvasEdge>): JSX.Element {
  const { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data } = props
  const kind = data?.kind ?? 'sequential'
  const shape = data?.shape ?? 'bezier'
  const animate = data?.animate ?? true
  const active = data?.active ?? false
  const index = data?.index ?? 0
  const [d, labelX, labelY] = shapePath(shape, {
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  })
  const pathId = `fp-${slug(id)}`
  const gradientId = `fg-${slug(id)}`
  const dur = active ? DOT_DUR_ACTIVE : DOT_DUR[kind]
  // A negative begin starts each dot part-way through its cycle, so a fan of
  // edges reads as a stream rather than a metronome.
  const begin = -((index % 7) * 0.34)
  const gradient = kind === 'fan_out' && data ? { stroke: `url(#${gradientId})` } : undefined
  const angle = data?.direction === 'TB' ? 90 : 0

  return (
    <g className={`flow-edge flow-edge--${kind}`} data-animate={animate} data-active={active}>
      {kind === 'fan_out' && data && (
        <defs>
          <linearGradient
            id={gradientId}
            gradientUnits="userSpaceOnUse"
            x1={sourceX}
            y1={sourceY}
            x2={targetX}
            y2={targetY}
          >
            <stop offset="0" style={{ stopColor: data.from }} />
            <stop offset="1" style={{ stopColor: data.to }} />
          </linearGradient>
        </defs>
      )}
      <path id={pathId} className="flow-edge__path" d={d} style={active ? undefined : gradient} />
      {animate ? (
        <>
          <MotionDot pathId={pathId} dur={dur} begin={begin} className="flow-edge__dot" />
          {(kind === 'fan_out' || kind === 'fan_in') && (
            <MotionDot
              pathId={pathId}
              dur={dur}
              begin={begin - dur / 2}
              className="flow-edge__dot"
            />
          )}
        </>
      ) : (
        <StaticDot x={labelX} y={labelY} angle={angle} />
      )}
    </g>
  )
}

/**
 * Routes the critic's revise signal down below the layout (LR) or out past its
 * right edge (TB) and back to the orchestrator, so the return leg never crosses
 * the forward path it is commenting on.
 */
function feedbackRoute(
  sx: number,
  sy: number,
  tx: number,
  ty: number,
  detour: number,
  direction: Direction,
) {
  const r = 16
  if (direction === 'LR') {
    const y = Math.max(detour, sy + r * 2, ty + r * 2)
    return {
      d: `M ${sx},${sy} L ${sx},${y - r} Q ${sx},${y} ${sx - r},${y} L ${tx + r},${y} Q ${tx},${y} ${tx},${y - r} L ${tx},${ty}`,
      labelX: (sx + tx) / 2,
      labelY: y,
      angle: 180,
    }
  }
  const x = Math.max(detour, sx + r * 2, tx + r * 2)
  return {
    d: `M ${sx},${sy} L ${x - r},${sy} Q ${x},${sy} ${x},${sy - r} L ${x},${ty + r} Q ${x},${ty} ${x - r},${ty} L ${tx},${ty}`,
    labelX: x,
    labelY: (sy + ty) / 2,
    angle: 270,
  }
}

function FeedbackEdge(props: EdgeProps<CanvasEdge>): JSX.Element {
  const { id, sourceX, sourceY, targetX, targetY, data } = props
  const animate = data?.animate ?? true
  const direction = data?.direction ?? 'LR'
  const route = feedbackRoute(sourceX, sourceY, targetX, targetY, data?.detour ?? 0, direction)
  const pathId = `fp-${slug(id)}`

  return (
    <>
      <g className="flow-edge flow-edge--feedback" data-animate={animate} data-active={false}>
        <path id={pathId} className="flow-edge__path" d={route.d} />
        {animate ? (
          <MotionDot pathId={pathId} dur={DOT_DUR.feedback} begin={-0.6} className="flow-edge__dot" />
        ) : (
          <StaticDot x={route.labelX} y={route.labelY} angle={route.angle} />
        )}
      </g>
      <EdgeLabelRenderer>
        <div
          className="flow-edge__pill nodrag nopan"
          style={{ transform: `translate(-50%, -50%) translate(${route.labelX}px, ${route.labelY}px)` }}
        >
          {data?.label || 'revise'}
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

const EDGE_TYPES: EdgeTypes = { flow: FlowEdge, feedback: FeedbackEdge }

/* --- Legend ---------------------------------------------------------------- */

function EdgeSample({ kind }: { kind: GraphEdge['kind'] }) {
  return (
    <svg className="flow-legend__sample" viewBox="0 0 42 12" aria-hidden>
      <g className={`flow-edge flow-edge--${kind}`} data-animate="false" data-active="false">
        <path className="flow-edge__path" d="M 1,6 L 41,6" />
        <circle className="flow-edge__dot" cx={kind === 'feedback' ? 14 : 28} cy={6} r={DOT_R} />
      </g>
    </svg>
  )
}

function Legend({ onClose }: { onClose: () => void }) {
  return (
    <Card
      className="flow-legend nopan nowheel"
      title="Legend"
      actions={
        <Button variant="ghost" size="sm" iconOnly icon={<X size={13} />} onClick={onClose} aria-label="Close legend" />
      }
    >
      <div className="flow-legend__group">
        {EDGE_LEGEND.map((row) => (
          <div key={row.kind} className="flow-legend__row">
            <EdgeSample kind={row.kind} />
            <span>
              <b>{titleCase(row.kind)}</b>: {row.body}
            </span>
          </div>
        ))}
      </div>
      <div className="flow-legend__group">
        {STATUS_LEGEND.map((row) => (
          <div key={row.status} className="flow-legend__row">
            <span
              className="flow-legend__swatch"
              data-status={row.status}
              style={{ '--st': `var(--st-${row.status})` } as React.CSSProperties}
            />
            <span>
              <b>{titleCase(row.status)}</b>: {row.body}
            </span>
          </div>
        ))}
      </div>
    </Card>
  )
}

/* --- Canvas ---------------------------------------------------------------- */

type CanvasProps = Omit<FlowCanvasProps, 'graph'> & { graph: RunGraph }

function Canvas({
  graph,
  steps,
  currentStage,
  iteration,
  running,
  animate = true,
  direction = 'LR',
  edgeStyle = 'bezier',
  showTokens = true,
  onSelectNode,
  className,
}: CanvasProps): JSX.Element {
  const instanceId = slug(useId())
  const reduced = useReducedMotion()
  const { setViewport, getViewport, zoomIn, zoomOut, zoomTo } = useReactFlow()
  // The pane size and the live zoom come from the store so the readout and the
  // viewport maths react to a resize without a second measurement path.
  const paneW = useStore((state) => state.width)
  const paneH = useStore((state) => state.height)
  const zoomNow = useStore((state) => state.transform[2])

  const [dir, setDir] = useState<Direction>(direction)
  const [edgeAnim, setEdgeAnim] = useState(animate)
  const [legendOpen, setLegendOpen] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selectedRef = useRef<string | null>(null)
  const [fitScope, setFitScope] = useState<FitScope>('content')
  /** True when the opening view was pinned rather than fitted. */
  const [anchored, setAnchored] = useState(false)
  /** Set by any pan or zoom the user drove themselves. Stops auto-following. */
  const [userMoved, setUserMoved] = useState(false)
  /** The first painted frame sits at the default transform, so it is hidden. */
  const [opened, setOpened] = useState(false)

  useEffect(() => setDir(direction), [direction])
  useEffect(() => setEdgeAnim(animate), [animate])

  const animateEdges = edgeAnim && !reduced

  const stepById = useMemo(() => {
    const map = new Map<number, RunStep>()
    for (const step of steps) map.set(step.id, step)
    return map
  }, [steps])

  // Everything the graph shape depends on, folded into one string. Token and
  // duration ticks deliberately stay out of it.
  const statusSig = useMemo(
    () =>
      `${steps.map((s) => `${s.id}:${s.status}`).join(',')}#${currentStage}#${running ? 1 : 0}#${iteration}`,
    [steps, currentStage, running, iteration],
  )
  // Read through a ref so the memo below can key on `statusSig` alone.
  const stepsRef = useRef(stepById)
  stepsRef.current = stepById

  const layout = useMemo(() => layoutGraph(graph, dir), [graph, dir])

  const statusById = useMemo(() => {
    const map = new Map<string, FlowStatus>()
    for (const node of graph.nodes) {
      if (node.type === 'agent') {
        map.set(node.id, agentStatus(node.stage, currentStage, running))
      } else {
        const step = node.step_id === undefined ? undefined : stepsRef.current.get(node.step_id)
        map.set(node.id, step ? step.status : planStatus(node.status))
      }
    }
    return map
    // `statusSig` stands in for the parts of `steps`, `currentStage` and
    // `running` that can change the answer.
  }, [graph, statusSig, currentStage, running])

  const nodes = useMemo<CanvasNode[]>(
    () =>
      layout.placed.map((p) => ({
        id: p.node.id,
        type: p.node.type,
        position: { x: p.x, y: p.y },
        width: p.w,
        height: p.h,
        draggable: false,
        selectable: false,
        connectable: false,
        focusable: false,
        data: { node: p.node, status: statusById.get(p.node.id) ?? 'pending' },
      })),
    [layout, statusById],
  )

  const nodeById = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph])

  const edges = useMemo<CanvasEdge[]>(
    () =>
      graph.edges.map((edge, index) => {
        const feedback = edge.kind === 'feedback'
        const source = nodeById.get(edge.source)
        const target = nodeById.get(edge.target)
        const active =
          !feedback &&
          statusById.get(edge.source) === 'complete' &&
          statusById.get(edge.target) === 'running'
        return {
          id: `${instanceId}-${edge.id}`,
          source: edge.source,
          target: edge.target,
          sourceHandle: feedback ? 'fb-out' : 'out',
          targetHandle: feedback ? 'fb-in' : 'in',
          type: feedback ? 'feedback' : 'flow',
          selectable: false,
          focusable: false,
          data: {
            kind: edge.kind,
            index,
            active,
            animate: animateEdges,
            shape: edgeStyle,
            direction: dir,
            detour: layout.detour,
            from: source ? familyVar(source) : 'var(--fam-other)',
            to: target ? familyVar(target) : 'var(--fam-other)',
            label: edge.label,
          },
        }
      }),
    [graph, nodeById, statusById, animateEdges, edgeStyle, dir, layout.detour, instanceId],
  )

  const select = useCallback(
    (node: GraphNode) => {
      const next = selectedRef.current === node.id ? null : node
      selectedRef.current = next?.id ?? null
      setSelectedId(next?.id ?? null)
      onSelectNode?.(next)
    },
    [onSelectNode],
  )

  const clearSelection = useCallback(() => {
    if (selectedRef.current === null) return
    selectedRef.current = null
    setSelectedId(null)
    onSelectNode?.(null)
  }, [onSelectNode])

  const ui = useMemo<UiValue>(
    () => ({ direction: dir, showTokens, selectedId, select }),
    [dir, showTokens, selectedId, select],
  )

  /**
   * The leading rank of the run, as a box in graph space. Waves can overlap
   * while an agent hands over to the next one, and the far end of that overlap
   * is the part worth watching.
   */
  const front = useMemo<Rect | null>(() => {
    const live = layout.placed.filter((p) => statusById.get(p.node.id) === 'running')
    if (!live.length) return null
    const rank = Math.max(...live.map((p) => p.node.rank))
    const wave = live.filter((p) => p.node.rank === rank)
    return union(
      wave.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
      BAND_PAD,
    )
  }, [layout, statusById])

  // Viewport work reads the current layout, pane, and front through refs so the
  // callbacks below keep a stable identity and the effects stay in control of
  // when a view is applied.
  const layoutRef = useRef(layout)
  layoutRef.current = layout
  const paneRef = useRef({ w: paneW, h: paneH })
  paneRef.current = { w: paneW, h: paneH }
  const frontRef = useRef(front)
  frontRef.current = front
  // A viewport transition is motion like any other, so the appearance setting
  // turns it into a cut.
  const reducedRef = useRef(reduced)
  reducedRef.current = reduced

  /**
   * Frames one of the two bounds exactly. This is the explicit request, so it
   * is allowed all the way down to `MIN_ZOOM`: fitting works off the layout
   * rather than fitView because the ruler, the bands, and the feedback lane
   * are not nodes.
   */
  const applyFit = useCallback(
    (scope: FitScope, animated: boolean) => {
      const { w, h } = paneRef.current
      if (w < 2 || h < 2) return undefined
      const rect = scope === 'full' ? layoutRef.current.full : layoutRef.current.content
      const zoom = clamp(fitZoom(rect, w, h), MIN_ZOOM, MAX_ZOOM)
      return setViewport(centred(rect, w, h, zoom), {
        duration: glide(animated, reducedRef.current, OPEN_MS),
      })
    },
    [setViewport],
  )

  /**
   * The opening view. It frames the forward graph when that is legible, and
   * otherwise pins the zoom at `MIN_OPEN_ZOOM` and anchors on the start of the
   * run: the leading edge of the graph, across from the active rank. Returns
   * whether it had to anchor, which is also the signal for auto-following.
   */
  const openView = useCallback(
    (animated: boolean) => {
      const { w, h } = paneRef.current
      if (w < 2 || h < 2) return false
      const rect = layoutRef.current.content
      const room = fitZoom(rect, w, h)
      const duration = glide(animated, reducedRef.current, OPEN_MS)
      if (room >= MIN_OPEN_ZOOM) {
        void setViewport(centred(rect, w, h, Math.min(room, MAX_OPEN_ZOOM)), { duration })
        return false
      }
      const zoom = MIN_OPEN_ZOOM
      const lr = layoutRef.current.direction === 'LR'
      const focus = frontRef.current ?? rect
      const start = lr ? rect.x : rect.y
      const mid = lr ? focus.y + focus.height / 2 : focus.x + focus.width / 2
      const lead = OPEN_MARGIN - start * zoom
      const across = (lr ? h : w) / 2 - mid * zoom
      void setViewport(lr ? { x: lead, y: across, zoom } : { x: across, y: lead, zoom }, { duration })
      return true
    },
    [setViewport],
  )

  // Reopen only when the graph itself or the axis changes, never on a state
  // tick. The pane size is a dependency because it is zero on the first pass.
  // React Flow's own `fitView` is deliberately not used: it lands whenever the
  // node measurements settle, which is after this and would undo the anchor.
  const openedRef = useRef('')
  useEffect(() => {
    if (paneW < 2 || paneH < 2 || openedRef.current === layout.key) return
    const first = openedRef.current === ''
    const timer = window.setTimeout(() => {
      openedRef.current = layout.key
      setFitScope('content')
      setUserMoved(false)
      setAnchored(openView(!first))
      setOpened(true)
    }, OPEN_DELAY)
    return () => window.clearTimeout(timer)
  }, [layout.key, paneW, paneH, openView])

  useEffect(() => {
    const timer = window.setTimeout(() => setOpened(true), REVEAL_MS)
    return () => window.clearTimeout(timer)
  }, [])

  // Follow the front of the run, but only while the opening view could not show
  // the whole plan and only until the user takes the viewport into their own
  // hands. Pressing the fit control hands it back.
  useEffect(() => {
    if (!running || !anchored || userMoved || !front || paneW < 2 || paneH < 2) return
    const view = getViewport()
    const zoom = view.zoom
    const left = -view.x / zoom
    const top = -view.y / zoom
    const slack = FOLLOW_INSET / zoom
    const shown =
      front.x >= left + slack &&
      front.y >= top + slack &&
      front.x + front.width <= left + paneW / zoom - slack &&
      front.y + front.height <= top + paneH / zoom - slack
    if (shown) return
    const lr = dir === 'LR'
    const cx = front.x + front.width / 2
    const cy = front.y + front.height / 2
    const lead = (lr ? paneW : paneH) * FOLLOW_LEAD - (lr ? cx : cy) * zoom
    const across = (lr ? paneH : paneW) / 2 - (lr ? cy : cx) * zoom
    void setViewport(lr ? { x: lead, y: across, zoom } : { x: across, y: lead, zoom }, {
      duration: glide(true, reduced, FOLLOW_MS),
    })
  }, [front, running, anchored, userMoved, paneW, paneH, dir, reduced, getViewport, setViewport])

  // Cycling the fit control is how the return path becomes discoverable without
  // spending zoom on it every time the canvas opens.
  const onFit = useCallback(() => {
    const done = applyFit(fitScope, true)
    setFitScope((current) => (current === 'content' ? 'full' : 'content'))
    // Following resumes only once the transition has landed, otherwise it would
    // read a half finished viewport and pull against it.
    void done?.then(() => setUserMoved(false))
  }, [applyFit, fitScope])

  const onMoveStart = useCallback((event: MouseEvent | TouchEvent | null) => {
    // A programmatic transition arrives with a null source event.
    if (event) setUserMoved(true)
  }, [])

  const zoomPct = Math.round(zoomNow * 100)

  const totalTokens = steps.reduce((sum, step) => sum + (step.tokens || 0), 0)
  const maxParallel = graph.meta.max_parallel || Math.max(1, ...graph.lanes.map((l) => l.size))

  return (
    <div
      className={clsx('flow', className)}
      data-direction={dir}
      data-animate={animateEdges}
      data-opened={opened}
    >
      <UiContext.Provider value={ui}>
        <LiveContext.Provider value={stepById}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            minZoom={MIN_ZOOM}
            maxZoom={MAX_ZOOM}
            onMoveStart={onMoveStart}
            nodesDraggable={false}
            nodesConnectable={false}
            nodesFocusable={false}
            edgesFocusable={false}
            elementsSelectable={false}
            zoomOnDoubleClick={false}
            onNodeClick={(_event, node) => select(node.data.node)}
            onPaneClick={clearSelection}
          >
            <Background variant={BackgroundVariant.Dots} gap={22} size={1.1} />

            <ViewportPortal>
              <div className="flow-layer">
                {layout.ranks
                  .filter((rank) => rank.parallel)
                  .map((rank) => (
                    <div
                      key={`band-${rank.rank}`}
                      className="flow-band"
                      style={{
                        transform: `translate(${rank.band.x}px, ${rank.band.y}px)`,
                        width: rank.band.w,
                        height: rank.band.h,
                      }}
                    />
                  ))}
                {layout.ranks.map((rank) => (
                  <div
                    key={`ruler-${rank.rank}`}
                    className="flow-ruler"
                    data-parallel={rank.parallel}
                    style={{
                      transform: `translate(${rank.ruler.x}px, ${rank.ruler.y}px)`,
                      width: rank.ruler.w,
                      height: rank.ruler.h,
                    }}
                  >
                    <span className="flow-ruler__label">{rank.label}</span>
                    <span className="flow-ruler__sub">{rank.sub}</span>
                  </div>
                ))}
              </div>
            </ViewportPortal>

            <Panel position="top-right" className="flow-controls nopan nowheel">
              <div className="flow-controls__cluster">
                <Tooltip content="Zoom in">
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={<ZoomIn size={14} />}
                    aria-label="Zoom in"
                    onClick={() => void zoomIn({ duration: 200 })}
                  />
                </Tooltip>
                <Tooltip content="Zoom out">
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={<ZoomOut size={14} />}
                    aria-label="Zoom out"
                    onClick={() => void zoomOut({ duration: 200 })}
                  />
                </Tooltip>
                <Tooltip content="Reset zoom to 100 percent">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="flow-controls__zoom"
                    aria-label={`Zoom is ${zoomPct} percent, reset to 100 percent`}
                    onClick={() => void zoomTo(1, { duration: glide(true, reduced, 200) })}
                  >
                    {zoomPct}%
                  </Button>
                </Tooltip>
                <Tooltip
                  content={
                    fitScope === 'full'
                      ? 'Fit everything, critic return path included'
                      : 'Fit the plan. Press again to include the return path'
                  }
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={fitScope === 'full' ? <Scan size={14} /> : <Maximize2 size={14} />}
                    aria-label={fitScope === 'full' ? 'Fit everything' : 'Fit the plan'}
                    onClick={onFit}
                  />
                </Tooltip>
                <span className="flow-controls__divider" />
                <Tooltip content={dir === 'LR' ? 'Switch to top down' : 'Switch to left to right'}>
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={dir === 'LR' ? <ArrowRightLeft size={14} /> : <ArrowDownUp size={14} />}
                    aria-label="Toggle flow direction"
                    onClick={() => setDir((current) => (current === 'LR' ? 'TB' : 'LR'))}
                  />
                </Tooltip>
                <Tooltip
                  content={
                    reduced
                      ? 'Motion is reduced in appearance settings'
                      : edgeAnim
                        ? 'Stop edge animation'
                        : 'Animate edges'
                  }
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    disabled={reduced}
                    icon={edgeAnim ? <Zap size={14} /> : <ZapOff size={14} />}
                    aria-label="Toggle edge animation"
                    aria-pressed={animateEdges}
                    onClick={() => setEdgeAnim((current) => !current)}
                  />
                </Tooltip>
                <Tooltip content="Legend">
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={<List size={14} />}
                    aria-label="Legend"
                    aria-expanded={legendOpen}
                    onClick={() => setLegendOpen((current) => !current)}
                  />
                </Tooltip>
              </div>
              {legendOpen && <Legend onClose={() => setLegendOpen(false)} />}
            </Panel>

            <Panel position="bottom-left" className="flow-stats nopan nowheel">
              <span className="flow-stat">
                <b>{graph.meta.step_count}</b> steps
              </span>
              <span className="flow-stats__sep" />
              <span className="flow-stat">
                <b>{graph.meta.depth}</b> deep
              </span>
              <span className="flow-stats__sep" />
              <span className="flow-stat">
                <Layers size={11} />
                <b>{maxParallel}</b> parallel
              </span>
              <span className="flow-stats__sep" />
              <span className="flow-stat">
                <b>{formatTokens(totalTokens)}</b> tokens
              </span>
              {iteration > 1 && <Badge tone="caution">iteration {iteration}</Badge>}
            </Panel>

            <MiniMap
              className="nopan"
              pannable
              zoomable
              ariaLabel="Plan overview"
              nodeBorderRadius={4}
              nodeStrokeWidth={0}
              maskColor="color-mix(in srgb, var(--bg-sunken) 74%, transparent)"
              bgColor="var(--bg-elevated)"
              nodeColor={(node) => {
                const status = (node.data as FlowNodeData | undefined)?.status ?? 'pending'
                return `var(--st-${status})`
              }}
            />
          </ReactFlow>
        </LiveContext.Provider>
      </UiContext.Provider>
    </div>
  )
}

/* --- Entry point ----------------------------------------------------------- */

export function FlowCanvas(props: FlowCanvasProps): JSX.Element {
  if (!props.graph) {
    return (
      <div className={clsx('flow', 'flow--empty', props.className)}>
        <EmptyState
          icon={<Waypoints size={20} />}
          title="No execution plan yet"
          body="The orchestrator writes the plan before anything runs. Once it lands, every tool call, its dependencies, and the critic loop appear here and update live."
        />
      </div>
    )
  }
  return (
    <ReactFlowProvider>
      <Canvas {...props} graph={props.graph} />
    </ReactFlowProvider>
  )
}

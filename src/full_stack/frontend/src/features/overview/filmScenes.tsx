/**
 * The eight scenes of the methodology film, plus the persistent stage chrome.
 *
 * Every scene is a pure function of act progress. Nothing here schedules a
 * timer or holds state, which is what lets the transport scrub to an exact
 * frame. Geometry is derived from the measured stage width so the stage
 * reflows rather than scales, and the type stays at its authored size.
 */

import type { JSX, ReactNode } from 'react'
import {
  CHECKLIST,
  EVIDENCE,
  LANES,
  PREDICTION,
  REPORT_SECTIONS,
  UNKNOWNS,
  WAVE_COUNT,
  WAVE_NOTES,
  WORKERS,
  clamp01,
  easeInOut,
  easeOut,
  fit,
  lerp,
  rise,
  seg,
  type Box,
  type FilmVocabulary,
  type ResolvedStep,
  type StageLayout,
} from './filmScript'

export interface SceneProps {
  p: number
  L: StageLayout
  v: FilmVocabulary
}

const WAVE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
const PLATE_W = 240
const PLATE_H = 96

/* --- Small drawing helpers ------------------------------------------------- */

/**
 * Filters every scene shares.
 *
 * Rendered once at the top of the stage svg. The shadow is what separates a
 * card from the field behind it: without one, a panel with a hairline border on
 * a flat ground reads as an outline rather than an object.
 */
export function FilmDefs(): JSX.Element {
  return (
    <defs>
      <filter id="film-lift" x="-20%" y="-20%" width="140%" height="150%">
        <feDropShadow dx="0" dy="1" stdDeviation="1" floodColor="var(--shadow-ink)" floodOpacity="0.10" />
        <feDropShadow dx="0" dy="4" stdDeviation="7" floodColor="var(--shadow-ink)" floodOpacity="0.09" />
      </filter>
      <filter id="film-lift-strong" x="-25%" y="-25%" width="150%" height="160%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="var(--shadow-ink)" floodOpacity="0.12" />
        <feDropShadow dx="0" dy="10" stdDeviation="14" floodColor="var(--shadow-ink)" floodOpacity="0.13" />
      </filter>
    </defs>
  )
}

function Panel({
  box,
  radius = 10,
  fill = 'var(--bg-elevated)',
  stroke = 'var(--line)',
  strokeWidth = 1,
  opacity = 1,
  dashed,
  lift = 'none',
}: {
  box: Box
  radius?: number
  fill?: string
  stroke?: string
  strokeWidth?: number
  opacity?: number
  dashed?: boolean
  lift?: 'none' | 'soft' | 'strong'
}) {
  return (
    <rect
      x={box.x}
      y={box.y}
      width={Math.max(0, box.w)}
      height={Math.max(0, box.h)}
      rx={radius}
      fill={fill}
      stroke={stroke}
      strokeWidth={strokeWidth}
      strokeDasharray={dashed ? '4 3' : undefined}
      opacity={opacity}
      filter={lift === 'strong' ? 'url(#film-lift-strong)' : lift === 'soft' ? 'url(#film-lift)' : undefined}
    />
  )
}

function Caption({
  x,
  y,
  text,
  size = 12,
  fill = 'var(--text-muted)',
  weight = 400,
  anchor = 'start',
  opacity = 1,
  mono,
}: {
  x: number
  y: number
  text: string
  size?: number
  fill?: string
  weight?: number
  anchor?: 'start' | 'middle' | 'end'
  opacity?: number
  mono?: boolean
}) {
  return (
    <text
      x={x}
      y={y}
      fontSize={size}
      fontWeight={weight}
      fill={fill}
      textAnchor={anchor}
      opacity={opacity}
      dominantBaseline="middle"
      fontFamily={mono ? 'var(--font-mono)' : undefined}
      style={{ letterSpacing: weight >= 600 && size <= 10 ? '0.07em' : undefined }}
    >
      {text}
    </text>
  )
}

/** An agent as it appears on stage: a plate with its role and its own summary. */
function AgentPlate({
  x,
  y,
  label,
  summary,
  glow = 0,
  appear,
}: {
  x: number
  y: number
  label: string
  summary: string
  glow?: number
  appear: number
}) {
  if (appear <= 0) return null
  const box: Box = { x, y, w: PLATE_W, h: PLATE_H }
  return (
    <g opacity={appear} transform={`translate(0 ${(1 - appear) * 8})`}>
      {glow > 0 && (
        <rect
          x={box.x - 5 * glow}
          y={box.y - 5 * glow}
          width={box.w + 10 * glow}
          height={box.h + 10 * glow}
          rx={15}
          fill="var(--accent)"
          opacity={0.1 * glow}
        />
      )}
      <Panel box={box} radius={13} stroke="var(--accent)" strokeWidth={1 + glow * 0.6} lift="strong" />
      <foreignObject x={box.x + 16} y={box.y + 12} width={box.w - 32} height={box.h - 24}>
        <div className="film__agent-copy"><strong>{label}</strong><span>{summary}</span></div>
      </foreignObject>
    </g>
  )
}

type StepState = 'pending' | 'queued' | 'running' | 'failed' | 'done'

const STEP_STROKE: Record<StepState, string> = {
  pending: 'var(--line-faint)',
  queued: 'var(--line-strong)',
  running: 'var(--accent)',
  failed: 'var(--caution)',
  done: 'var(--line)',
}

function StepNode({
  step,
  box,
  state,
  progress = 0,
  appear = 1,
  compact,
}: {
  step: ResolvedStep
  box: Box
  state: StepState
  progress?: number
  appear?: number
  compact?: boolean
}) {
  if (appear <= 0 || !box) return null
  const fam = `var(--fam-${step.familyIndex})`
  const dim = state === 'pending' ? 0.82 : state === 'queued' ? 0.9 : 1
  return (
    <g opacity={appear * dim} transform={`translate(0 ${(1 - appear) * 10})`}>
      <Panel
        box={box}
        radius={9}
        stroke={STEP_STROKE[state]}
        strokeWidth={state === 'running' || state === 'failed' ? 1.5 : 1}
        dashed={state === 'queued'}
        fill={state === 'failed' ? 'var(--caution-soft)' : 'var(--bg-elevated)'}
      />
      <rect x={box.x} y={box.y + 7} width={3} height={Math.max(4, box.h - 14)} rx={1.5} fill={fam} />
      <foreignObject x={box.x + 12} y={box.y + 5} width={box.w - 34} height={box.h - 10}>
        <div className="film__step-copy" data-compact={compact}>
          <strong>{step.toolName.split(/(?=[A-Z][a-z])/).map((part, index) => <span key={index}>{index > 0 && <wbr />}{part}</span>)}</strong>
          <span>{step.subject}</span>
        </div>
      </foreignObject>
      {state === 'running' && (
        <rect
          x={box.x + 1}
          y={box.y + box.h - 3}
          width={Math.max(0, (box.w - 2) * clamp01(progress))}
          height={2}
          rx={1}
          fill={fam}
        />
      )}
      {state === 'done' && <Mark kind="check" x={box.x + box.w - 15} y={box.y + box.h / 2} />}
      {state === 'failed' && <Mark kind="warn" x={box.x + box.w - 15} y={box.y + box.h / 2} />}
      {state === 'running' && (
        <circle cx={box.x + box.w - 15} cy={box.y + box.h / 2} r={3.5} fill="var(--accent)" opacity={0.5 + 0.5 * Math.sin(progress * Math.PI * 8)} />
      )}
    </g>
  )
}

/** Status glyphs, drawn rather than imported so they sit on the SVG grid. */
function Mark({ kind, x, y, size = 5 }: { kind: 'check' | 'warn' | 'cross'; x: number; y: number; size?: number }) {
  if (kind === 'check') {
    return (
      <g>
        <circle cx={x} cy={y} r={size + 2.5} fill="var(--positive-soft)" />
        <path
          d={`M ${x - size * 0.7} ${y} L ${x - size * 0.15} ${y + size * 0.6} L ${x + size * 0.75} ${y - size * 0.6}`}
          fill="none"
          stroke="var(--positive)"
          strokeWidth={1.7}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    )
  }
  if (kind === 'warn') {
    return (
      <g>
        <circle cx={x} cy={y} r={size + 2.5} fill="var(--caution-soft)" />
        <line x1={x} y1={y - size * 0.7} x2={x} y2={y + size * 0.1} stroke="var(--caution)" strokeWidth={1.7} strokeLinecap="round" />
        <circle cx={x} cy={y + size * 0.72} r={0.95} fill="var(--caution)" />
      </g>
    )
  }
  return (
    <g>
      <circle cx={x} cy={y} r={size + 2.5} fill="var(--critical-soft)" />
      <path
        d={`M ${x - size * 0.55} ${y - size * 0.55} L ${x + size * 0.55} ${y + size * 0.55} M ${x + size * 0.55} ${y - size * 0.55} L ${x - size * 0.55} ${y + size * 0.55}`}
        stroke="var(--critical)"
        strokeWidth={1.7}
        strokeLinecap="round"
      />
    </g>
  )
}

/** Markers share the film clock. Pausing and scrubbing stop them exactly. */
function Edge({ id, from, to, draw, travel, compact }: { id: string; from: Box; to: Box; draw: number; travel: number; compact?: boolean }) {
  if (draw <= 0 || !from || !to) return null
  const bend = compact ? 18 : Math.max(16, (to.x - (from.x + from.w)) * 0.6)
  const start = compact ? { x: from.x + from.w / 2, y: from.y + from.h } : { x: from.x + from.w, y: from.y + from.h / 2 }
  const end = compact ? { x: to.x + to.w / 2, y: to.y } : { x: to.x, y: to.y + to.h / 2 }
  const c1 = compact ? { x: start.x, y: start.y + bend } : { x: start.x + bend, y: start.y }
  const c2 = compact ? { x: end.x, y: end.y - bend } : { x: end.x - bend, y: end.y }
  const path = `M ${start.x} ${start.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${end.x} ${end.y}`
  const cubic = (a: number, b: number, c: number, d: number, t: number) =>
    (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d
  return <g>
    <path id={id} d={path} fill="none" stroke="var(--line-strong)" strokeWidth={1.2}
      pathLength={1} strokeDasharray={1} strokeDashoffset={1 - clamp01(draw)} opacity={0.85} />
    {draw >= 0.999 && <>
      <path d={compact ? `M ${end.x - 3} ${end.y - 5} L ${end.x} ${end.y} L ${end.x + 3} ${end.y - 5}` : `M ${end.x - 5} ${end.y - 3} L ${end.x} ${end.y} L ${end.x - 5} ${end.y + 3}`}
        fill="none" stroke="var(--accent)" strokeWidth={1.2} opacity={0.65} />
      {[0].map((slot) => {
        const offset = Array.from(id).reduce((sum, ch) => sum + ch.charCodeAt(0), 0) * 0.037
        const t = (travel * 3 + offset) % 1
        return <circle key={slot} className="film__traveler" r={2.3}
          cx={cubic(start.x, c1.x, c2.x, end.x, t)} cy={cubic(start.y, c1.y, c2.y, end.y, t)}
          fill="var(--accent)" opacity={Math.min(1, t * 8, (1 - t) * 8) * 0.9} />
      })}
    </>}
  </g>
}

/* --- Shared plan geometry -------------------------------------------------- */

export interface PlanGeometry {
  pos: Record<string, Box>
  waves: { letter: string; count: number; x: number; y: number; w: number }[]
  agent: Box
}

/**
 * One geometry for the plan, shared by orchestration, execution, and
 * integration, so a step never jumps when the act changes.
 */
export function planGeometry(L: StageLayout, steps: ResolvedStep[]): PlanGeometry {
  const pos: Record<string, Box> = {}
  const waves: PlanGeometry['waves'] = []
  const byWave: ResolvedStep[][] = Array.from({ length: WAVE_COUNT }, () => [])
  for (const step of steps) byWave[step.wave]?.push(step)

  if (L.wide) {
    const colGap = 36
    // The agent moves above the graph unless the columns can still hold a full
    // tool name beside it. `ClinicalRelevanceRanker` is the longest the engine
    // ships, and a column narrower than this had it rendering as an ellipsis,
    // which is the one thing a diagram of named tools must not do.
    const MIN_NODE_W = 230
    const gutter = PLATE_W + 30
    const beside = (L.work.w - gutter - colGap * (WAVE_COUNT - 1)) / WAVE_COUNT >= MIN_NODE_W
    const agentAbove = !beside
    // One inset for both the width the graph gets and where it starts. Deriving
    // them separately is how the graph ended up indented by a gutter that was
    // no longer there, and clipped off the right edge.
    const inset = agentAbove ? 0 : gutter
    const available = L.work.w - inset
    const nodeW = Math.min(268, (available - colGap * (WAVE_COUNT - 1)) / WAVE_COUNT)
    const graphW = nodeW * WAVE_COUNT + colGap * (WAVE_COUNT - 1)
    const originX = L.work.x + inset + (available - graphW) / 2
    const maxRows = byWave.reduce((max, rows) => Math.max(max, rows.length), 1)
    // The tallest column decides the row size, so a wave of six reads at the
    // same rhythm as a wave of two instead of running off the stage.
    const nodeH = nodeW >= 224 ? 52 : 64
    const rowGap = 18
    const blockH = maxRows * nodeH + (maxRows - 1) * rowGap
    const top = L.work.y + (agentAbove ? PLATE_H + 38 : 28)

    byWave.forEach((rows, wave) => {
      const x = originX + wave * (nodeW + colGap)
      const columnH = rows.length * nodeH + Math.max(0, rows.length - 1) * rowGap
      const y0 = top + (blockH - columnH) / 2
      rows.forEach((step, index) => {
        pos[step.id] = { x, y: y0 + index * (nodeH + rowGap), w: nodeW, h: nodeH }
      })
      waves.push({ letter: WAVE_LETTERS[wave] ?? String(wave + 1), count: rows.length, x, y: top - 16, w: nodeW })
    })

    return {
      pos,
      waves,
      agent: agentAbove
        ? { x: L.work.x + (L.work.w - PLATE_W) / 2, y: L.work.y, w: PLATE_W, h: PLATE_H }
        : { x: L.work.x, y: top + blockH / 2 - PLATE_H / 2, w: PLATE_W, h: PLATE_H },
    }
  }

  const nodeW = L.work.w
  const nodeH = 58
  const gap = 9
  const labelH = 20
  let cursor = L.work.y + PLATE_H + 16
  byWave.forEach((rows, wave) => {
    waves.push({ letter: WAVE_LETTERS[wave] ?? String(wave + 1), count: rows.length, x: L.work.x, y: cursor, w: nodeW })
    cursor += labelH
    rows.forEach((step) => {
      pos[step.id] = { x: L.work.x, y: cursor, w: nodeW, h: nodeH }
      cursor += nodeH + gap
    })
    cursor += 5
  })
  return { pos, waves, agent: { x: L.work.x, y: L.work.y, w: PLATE_W, h: PLATE_H - 16 } }
}

/** Tiny diagrams illustrate the actual output structures without renaming them. */
function TaskShapeGlyph({ value, x, y, selected }: { value: string; x: number; y: number; selected: boolean }) {
  const color = selected ? 'var(--accent)' : 'var(--text-muted)'
  const dot = (cx: number, cy: number, key: number) => <circle key={key} cx={cx} cy={cy} r={5} fill="var(--bg-elevated)" stroke={color} strokeWidth={1.5} />
  if (value === 'hierarchical') return <g transform={`translate(${x} ${y})`}>
    <path d="M 0 -13 V 0 H -25 V 13 M 0 0 H 25 V 13" fill="none" stroke={color} strokeWidth={1.5} />
    {dot(0, -13, 0)}{dot(-25, 13, 1)}{dot(25, 13, 2)}
  </g>
  if (value.startsWith('regression')) return <g transform={`translate(${x} ${y})`}>
    <path d="M -29 -16 V 18 H 30" fill="none" stroke="var(--line-strong)" />
    <path d="M -23 10 L -9 3 L 8 5 L 26 -12" fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" />
    {value === 'regression_multivariate' && <path d="M -23 -7 L -7 -1 L 9 -10 L 26 -3" fill="none" stroke="var(--fam-1)" strokeWidth={1.5} strokeLinecap="round" />}
  </g>
  const count = value === 'binary' ? 2 : 3
  return <g transform={`translate(${x} ${y})`}>{Array.from({ length: count }, (_, i) =>
    <rect key={i} x={(i - (count - 1) / 2) * 27 - 9} y={-10} width={18} height={22} rx={5}
      fill={i === 0 ? 'var(--accent-50)' : 'var(--bg-elevated)'} stroke={i === 0 ? color : 'var(--line-strong)'} strokeWidth={1.5} />
  )}</g>
}

/* --- Act 1: the question --------------------------------------------------- */

export function SceneQuestion({ p, L, v }: SceneProps): JSX.Element {
  const shapes = v.shapes
  const chosen = v.chosenShape
  const count = Math.max(1, shapes.length)
  const gap = L.wide ? 14 : 10
  const columns = L.scene.w >= 1150 ? count : L.scene.w >= 600 ? 3 : L.scene.w >= 380 ? 2 : 1
  const rows = Math.ceil(count / columns)
  const cardW = (L.scene.w - gap * (columns - 1)) / columns
  const cardH = 158
  const gridH = rows * cardH + (rows - 1) * gap
  // The grid is the whole act now, so it sits in the middle of the stage
  // rather than at the top of a column it used to share with a contract panel.
  const top = L.scene.y + Math.max(6, (L.scene.h - gridH) / 2)

  const boxFor = (index: number): Box => ({
    x: L.scene.x + (index % columns) * (cardW + gap) + (Math.floor(index / columns) === rows - 1 ? (columns - (count - (rows - 1) * columns)) * (cardW + gap) / 2 : 0),
    y: top + Math.floor(index / columns) * (cardH + gap),
    w: cardW,
    h: cardH,
  })

  // Visit every actual task type from right to left. Each stop gets a dwell
  // before the next movement, so multivariate regression cannot be skipped.
  const tour = Array.from({ length: count }, (_, i) => count - 1 - i)
  if (tour[tour.length - 1] !== chosen) tour.push(chosen)
  const travel = seg(p, 0.22, 0.88)
  const leg = travel * Math.max(1, tour.length - 1)
  const legIndex = Math.min(tour.length - 2, Math.floor(leg))
  const legT = easeInOut(seg(leg - legIndex, 0.38, 1))
  const fromBox = boxFor(tour[Math.max(0, legIndex)])
  const toBox = boxFor(tour[Math.min(tour.length - 1, legIndex + 1)])
  const highlight = { x: lerp(fromBox.x, toBox.x, legT), y: lerp(fromBox.y, toBox.y, legT) }
  const settled = seg(p, 0.90, 0.99)

  return (
    <g>
      {shapes.map((shape, index) => {
        const box = boxFor(index)
        const enter = rise(p, 0.02 + index * 0.02, 0.12 + index * 0.02)
        const isChosen = index === chosen
        const fade = isChosen ? 1 : lerp(1, 0.86, settled)
        return (
          <g key={shape.value} opacity={enter * fade} transform={`translate(0 ${(1 - enter) * 12})`}>
            <Panel
              box={box}
              radius={11}
              stroke={isChosen && settled > 0.4 ? 'var(--accent)' : 'var(--line)'}
              lift={isChosen && settled > 0.4 ? 'soft' : 'none'}
            />
            <TaskShapeGlyph value={shape.value} x={box.x + box.w / 2} y={box.y + 34} selected={isChosen && settled > 0.4} />
            <foreignObject x={box.x + 12} y={box.y + 64} width={Math.max(10, box.w - 24)} height={84}>
              <div className="film__shape-copy"><strong>{shape.label}</strong><span>{shape.summary}</span></div>
            </foreignObject>
          </g>
        )
      })}

      {p >= 0.18 && settled < 1 && (
        <rect
          className="film__shape-highlight"
          x={highlight.x - 3}
          y={highlight.y - 3}
          width={cardW + 6}
          height={cardH + 6}
          rx={13}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.6}
          opacity={1 - settled * 0.5}
        />
      )}

    </g>
  )
}

/* --- Act 2: the evidence --------------------------------------------------- */

export function SceneEvidence({ p, L }: SceneProps): JSX.Element {
  /*
   * The participant's evidence as a radial hierarchy.
   *
   * A single subject has no cohort to be a fraction of, so nothing here is
   * counted: the earlier version reported "22 of 100 leaves" per domain, which
   * invited a comparison against a reference that does not exist for one
   * person. What matters is the shape, so the shape is what is drawn: the
   * participant at the centre, the domains around them, and what sits under
   * each domain one ring further out.
   */
  const cx = L.scene.x + L.scene.w / 2
  const cy = L.scene.y + L.scene.h / 2
  const ringA = { rx: Math.min(224, L.scene.w * 0.2), ry: Math.min(104, L.scene.h * 0.27) }
  const ringB = { rx: Math.min(392, L.scene.w * 0.36), ry: Math.min(168, L.scene.h * 0.42) }
  const step = 360 / LANES.length
  const at = (angle: number, ring: { rx: number; ry: number }) => {
    const radians = ((angle - 90) * Math.PI) / 180
    return { x: cx + Math.cos(radians) * ring.rx, y: cy + Math.sin(radians) * ring.ry }
  }
  const centre = rise(p, 0, 0.14)

  return (
    <g>
      {LANES.map((lane, index) => {
        const angle = index * step
        const node = at(angle, ringA)
        const limb = rise(p, 0.1 + index * 0.06, 0.34 + index * 0.06)
        const spread = step * 0.34
        return (
          <g key={lane.key}>
            <path
              d={`M ${cx} ${cy} Q ${(cx + node.x) / 2} ${(cy + node.y) / 2}, ${node.x} ${node.y}`}
              fill="none"
              stroke="var(--line-strong)"
              strokeWidth={1.2}
              pathLength={1}
              strokeDasharray={1}
              strokeDashoffset={1 - limb}
              opacity={0.75}
            />
            {lane.branches.map((branch, twig) => {
              const leafAngle = angle + (twig - (lane.branches.length - 1) / 2) * spread
              const tip = at(leafAngle, ringB)
              const grow = rise(p, 0.3 + index * 0.05 + twig * 0.03, 0.6 + index * 0.05 + twig * 0.03)
              const anchor = Math.cos(((leafAngle - 90) * Math.PI) / 180) >= 0 ? 'start' : 'end'
              const nudge = anchor === 'start' ? 9 : -9
              return (
                <g key={branch}>
                  <path
                    d={`M ${node.x} ${node.y} Q ${(node.x + tip.x) / 2} ${(node.y + tip.y) / 2}, ${tip.x} ${tip.y}`}
                    fill="none"
                    stroke="var(--line)"
                    strokeWidth={1}
                    pathLength={1}
                    strokeDasharray={1}
                    strokeDashoffset={1 - grow}
                    opacity={0.6}
                  />
                  <g opacity={grow}>
                    <circle cx={tip.x} cy={tip.y} r={3.4} fill={`var(--fam-${index % 8})`} />
                    <Caption
                      x={tip.x + nudge}
                      y={tip.y}
                      text={branch}
                      size={10.5}
                      anchor={anchor}
                      fill="var(--text-muted)"
                    />
                  </g>
                </g>
              )
            })}
            <g opacity={limb}>
              <circle cx={node.x} cy={node.y} r={7} fill={`var(--fam-${index % 8})`} />
              <circle cx={node.x} cy={node.y} r={12} fill="none" stroke={`var(--fam-${index % 8})`} opacity={0.35} />
              <Caption
                x={node.x}
                y={node.y - 22}
                text={lane.label}
                size={12}
                weight={600}
                anchor="middle"
                fill="var(--text)"
              />
            </g>
          </g>
        )
      })}

      <g opacity={centre}>
        <circle cx={cx} cy={cy} r={26} fill="var(--accent-50)" stroke="var(--accent)" />
        <circle cx={cx} cy={cy - 6} r={6.5} fill="var(--accent)" opacity={0.8} />
        <path d={`M ${cx - 11} ${cy + 14} a 11 11 0 0 1 22 0 z`} fill="var(--accent)" opacity={0.8} />
      </g>
      <Caption
        x={cx}
        y={cy + 42}
        text="one participant"
        size={10.5}
        anchor="middle"
        opacity={centre}
      />
    </g>
  )
}

/* --- Act 3: orchestration -------------------------------------------------- */

export function SceneOrchestration({ p, L, v }: SceneProps): JSX.Element {
  const geometry = planGeometry(L, v.steps)
  const agent = v.agents.orchestrator
  const glow = p < 0.42 ? 0.5 + 0.5 * Math.sin(seg(p, 0, 0.4) * Math.PI * 3) : 0
  const waveIn = rise(p, 0.8, 0.96)

  return (
    <g>
      <AgentPlate
        x={geometry.agent.x}
        y={geometry.agent.y}
        label={agent.label}
        summary={agent.summary}
        glow={glow}
        appear={rise(p, 0, 0.14)}
      />

      {v.steps.map((step) =>
        step.deps.map((dep) => {
          const index = v.steps.findIndex((row) => row.id === step.id)
          return (
            <Edge
              key={`${dep}-${step.id}`}
              id={`plan-${dep}-${step.id}`}
              from={geometry.pos[dep]}
              to={geometry.pos[step.id]}
              draw={rise(p, 0.18 + index * 0.035, 0.34 + index * 0.035)}
              travel={p}
              compact={!L.wide}
            />
          )
        }),
      )}

      {v.steps.map((step, index) => (
        <StepNode
          key={step.id}
          step={step}
          box={geometry.pos[step.id]}
          state="pending"
          appear={rise(p, 0.12 + index * 0.035, 0.28 + index * 0.035)}
          compact={!L.wide}
        />
      ))}

      {geometry.waves.map((wave, index) => (
        <g key={wave.letter} opacity={waveIn}>
          {L.wide ? (
            <>
              <line x1={wave.x} y1={wave.y + 8} x2={wave.x + wave.w} y2={wave.y + 8} stroke="var(--line-strong)" />
              {/* What the wave is, centred over the column it labels. A
                  letter and a step count said nothing the column did not
                  already show. */}
              <Caption
                x={wave.x + wave.w / 2}
                y={wave.y - 4}
                text={fit((WAVE_NOTES[index] ?? '').toUpperCase(), wave.w, 9.5)}
                size={9.5}
                weight={700}
                anchor="middle"
              />
            </>
          ) : (
            <Caption
              x={wave.x}
              y={wave.y + 10}
              text={(WAVE_NOTES[index] ?? '').toUpperCase()}
              size={9.5}
              weight={700}
            />
          )}
        </g>
      ))}

    </g>
  )
}

/* --- Act 4: execution ------------------------------------------------------ */

function stepStateAt(step: ResolvedStep, p: number): { state: StepState; progress: number } {
  if (p < step.queued) return { state: 'pending', progress: 0 }
  if (p < step.start) return { state: 'queued', progress: 0 }
  if (step.failAt !== undefined && p >= step.failAt && p < (step.retryAt ?? step.failAt)) {
    return { state: 'failed', progress: seg(p, step.start, step.end) }
  }
  if (p < step.end) return { state: 'running', progress: seg(p, step.start, step.end) }
  return { state: 'done', progress: 1 }
}

export function SceneExecution({ p, L, v }: SceneProps): JSX.Element {
  const geometry = planGeometry(L, v.steps)
  const done = v.steps.filter((step) => p >= step.end).length
  const queue = v.steps.filter((step) => p >= step.queued && p < step.start).length
  const laneH = L.wide ? 24 : 22
  const laneGap = L.wide ? 6 : 5
  const poolTop = L.strip.y + (L.wide ? 30 : 26)
  const queueW = L.wide ? 128 : 96
  const trackX = L.strip.x + 16 + queueW + 16
  const trackW = L.strip.x + L.strip.w - 16 - trackX

  return (
    <g>
      <AgentPlate x={geometry.agent.x} y={geometry.agent.y} label={v.agents.executor.label}
        summary={v.agents.executor.summary} appear={1} />
      {v.steps.map((step) =>
        step.deps.map((dep) => (
          <Edge
            key={`${dep}-${step.id}`}
            id={`exec-${dep}-${step.id}`}
            from={geometry.pos[dep]}
            to={geometry.pos[step.id]}
            draw={1}
            travel={p}
            compact={!L.wide}
          />
        )),
      )}

      {v.steps.map((step) => {
        const { state, progress } = stepStateAt(step, p)
        const box = geometry.pos[step.id]
        const repairing = step.failAt !== undefined && p >= step.failAt && p < step.end
        const label = p < (step.retryAt ?? 1) ? 'step failed' : 'repaired in place, retry 1'
        return (
          <g key={step.id}>
            <StepNode step={step} box={box} state={state} progress={progress} compact={!L.wide} />
            {repairing && box && (
              <g opacity={seg(p, step.failAt ?? 0, (step.failAt ?? 0) + 0.02)}>
                <rect
                  x={box.x + box.w - 12 - label.length * 5.6}
                  y={box.y - 15}
                  width={label.length * 5.6 + 12}
                  height={16}
                  rx={8}
                  fill="var(--caution-soft)"
                />
                <Caption
                  x={box.x + box.w - 6}
                  y={box.y - 7}
                  text={label}
                  size={10}
                  anchor="end"
                  fill="var(--caution)"
                  weight={600}
                />
              </g>
            )}
          </g>
        )
      })}

      <g>
        <Panel box={L.strip} radius={11} fill="var(--bg-inset)" stroke="var(--line-faint)" />
        <Caption x={L.strip.x + 16} y={L.strip.y + 17} text={`WORKER POOL, ${WORKERS} CONCURRENT`} size={9.5} weight={700} />
        <Caption
          x={L.strip.x + L.strip.w - 16}
          y={L.strip.y + 17}
          text={`${done} of ${v.steps.length} steps complete`}
          size={10.5}
          anchor="end"
          mono
        />

        <Panel
          box={{ x: L.strip.x + 16, y: poolTop, w: queueW, h: WORKERS * laneH + (WORKERS - 1) * laneGap }}
          radius={8}
          fill="var(--bg-sunken)"
          stroke="var(--line-faint)"
          dashed
        />
        <Caption
          x={L.strip.x + 16 + queueW / 2}
          y={poolTop + 20}
          text={queue === 0 ? 'queue empty' : `${queue} waiting`}
          size={11}
          anchor="middle"
          weight={600}
          fill={queue === 0 ? 'var(--text-faint)' : 'var(--text-secondary)'}
        />
        {queue > 0 &&
          Array.from({ length: Math.min(4, queue) }).map((_, index) => (
            <rect
              key={index}
              x={L.strip.x + 16 + queueW / 2 - Math.min(4, queue) * 7 + index * 14}
              y={poolTop + 32}
              width={10}
              height={14}
              rx={2}
              fill="var(--accent)"
              opacity={0.32}
            />
          ))}

        {Array.from({ length: WORKERS }).map((_, worker) => {
          const y = poolTop + worker * (laneH + laneGap)
          const busy = v.steps.find((step) => step.worker === worker && p >= step.start && p < step.end)
          const failing = busy && busy.failAt !== undefined && p >= busy.failAt && p < (busy.retryAt ?? busy.failAt)
          const progress = busy ? seg(p, busy.start, busy.end) : 0
          return (
            <g key={worker}>
              <rect x={trackX} y={y} width={trackW} height={laneH} rx={6} fill="var(--bg-sunken)" />
              {busy ? (
                <>
                  <rect
                    x={trackX}
                    y={y}
                    width={Math.max(6, trackW * progress)}
                    height={laneH}
                    rx={6}
                    fill={`var(--fam-${busy.familyIndex})`}
                    opacity={failing ? 0.12 : 0.2}
                  />
                  <Caption
                    x={trackX + 10}
                    y={y + laneH / 2}
                    text={fit(`${busy.toolName}, ${busy.subject}`, trackW - 76, 11)}
                    size={11}
                    weight={600}
                    fill="var(--text)"
                  />
                  <Caption
                    x={trackX + trackW - 10}
                    y={y + laneH / 2}
                    text={failing ? 'retrying' : `${Math.round(progress * 100)}%`}
                    size={10}
                    anchor="end"
                    mono
                    fill={failing ? 'var(--caution)' : 'var(--text-muted)'}
                  />
                </>
              ) : (
                <Caption x={trackX + 10} y={y + laneH / 2} text={`worker ${worker + 1}, idle`} size={10.5} fill="var(--text-faint)" />
              )}
            </g>
          )
        })}
      </g>
    </g>
  )
}

/* --- Act 5: integration ---------------------------------------------------- */

export function SceneIntegration({ p, L, v }: SceneProps): JSX.Element {
  const columns = 3
  const gap = 32
  const nodeW = (L.scene.w - gap * (columns - 1) - 24) / columns
  const nodeH = 64
  const rowGap = 18
  const gridX = L.scene.x + 12
  const gridY = L.scene.y + PLATE_H + 42
  const rows = Math.ceil(v.steps.length / columns)
  const gridH = rows * nodeH + (rows - 1) * rowGap
  const bundle: Box = { x: L.scene.x + (L.scene.w - 430) / 2, y: gridY + gridH + 52, w: 430, h: 78 }
  const bundleIn = rise(p, 0.66, 0.86)
  return <g>
    <AgentPlate x={L.scene.x + (L.scene.w - PLATE_W) / 2} y={L.scene.y}
      label={v.agents.integrator.label} summary={v.agents.integrator.summary} appear={rise(p, 0, 0.14)} />
    {Array.from({ length: columns }, (_, column) => {
      const source: Box = { x: gridX + column * (nodeW + gap), y: gridY, w: nodeW, h: gridH }
      return <g key={column}>
        <Panel box={{ x: source.x - 7, y: source.y - 8, w: source.w + 14, h: source.h + 16 }} radius={13}
          fill="var(--bg-sunken)" stroke="var(--line-faint)" opacity={rise(p, 0.03, 0.15)} />
        <Edge id={`integrate-${column}`} from={{ ...source, h: source.h + 8 }}
          to={{ x: bundle.x + bundle.w * (column + 0.5) / columns - 1, y: bundle.y, w: 2, h: bundle.h }}
          draw={rise(p, 0.57 + column * 0.04, 0.76 + column * 0.04)} travel={p} compact />
      </g>
    })}
    {v.steps.map((step, index) => {
      const column = Math.floor(index / rows)
      const row = index % rows
      return <StepNode key={step.id} step={step} state="done"
        box={{ x: gridX + column * (nodeW + gap), y: gridY + row * (nodeH + rowGap), w: nodeW, h: nodeH }}
        appear={rise(p, 0.06 + index * 0.035, 0.22 + index * 0.035)} />
    })}
    <g opacity={bundleIn}>
      <Panel box={bundle} radius={14} fill="var(--accent-50)" stroke="var(--accent-200)" lift="soft" />
      <Caption x={bundle.x + bundle.w / 2} y={bundle.y + 27} text="INTEGRATED EVIDENCE BUNDLE" size={12} weight={650} fill="var(--accent)" anchor="middle" />
      <Caption x={bundle.x + bundle.w / 2} y={bundle.y + 51} text="Step outputs combined, source references retained" size={11.5} anchor="middle" />
    </g>
  </g>
}

/* --- Act 6: prediction ----------------------------------------------------- */

export function ScenePrediction({ p, L, v }: SceneProps): JSX.Element {
  const gap = 54
  const cardW = Math.min(450, (L.scene.w - gap) / 2)
  const groupX = L.scene.x + (L.scene.w - cardW * 2 - gap) / 2
  const cardY = L.scene.y + PLATE_H + 62
  const cardH = 336
  const left = { x: groupX, y: cardY, w: cardW, h: cardH }
  const right = { x: groupX + cardW + gap, y: cardY, w: cardW, h: cardH }
  const agentX = L.scene.x + (L.scene.w - PLATE_W) / 2
  const agentY = L.scene.y
  const frame = rise(p, 0.02, 0.18)
  const bar = easeInOut(seg(p, 0.28, 0.62))
  const confidence = rise(p, 0.72, 0.95)
  const shape = v.shapes[v.chosenShape]
  const incoming = `M ${left.x + left.w / 2} ${left.y} C ${left.x + left.w / 2} ${agentY + PLATE_H / 2}, ${agentX - 28} ${agentY + PLATE_H / 2}, ${agentX} ${agentY + PLATE_H / 2}`
  const outgoing = `M ${agentX + PLATE_W} ${agentY + PLATE_H / 2} C ${right.x + right.w / 2} ${agentY + PLATE_H / 2}, ${right.x + right.w / 2} ${right.y - 28}, ${right.x + right.w / 2} ${right.y}`
  return <g>
    <path d={incoming} fill="none" stroke="var(--accent)" strokeWidth={1.5} pathLength={1} strokeDasharray={1} strokeDashoffset={1 - rise(p, 0.12, 0.35)} opacity={0.55} />
    <path d={outgoing} fill="none" stroke="var(--accent)" strokeWidth={1.5} pathLength={1} strokeDasharray={1} strokeDashoffset={1 - rise(p, 0.28, 0.48)} opacity={0.55} />
    <path d={`M ${right.x + right.w / 2 - 4} ${right.y - 6} l 4 6 l 4 -6`} fill="none" stroke="var(--accent)" strokeWidth={1.5} opacity={rise(p, 0.4, 0.5)} />
    <AgentPlate x={agentX} y={agentY} label={v.agents.predictor.label} summary={v.agents.predictor.summary} appear={rise(p, 0, 0.14)} />
    <g opacity={frame}>
      <Panel box={left} radius={16} lift="soft" />
      <Caption x={left.x + left.w / 2} y={left.y + 28} text="EVIDENCE CHAIN" size={10.5} weight={650} anchor="middle" />
      {LANES.map((lane, index) => {
        const evidence = EVIDENCE.find((row) => row.lane === lane.key)
        const y = left.y + 64 + index * 51
        const appear = rise(p, 0.10 + index * 0.07, 0.25 + index * 0.07)
        return <g key={lane.key} opacity={appear}>
          <line x1={left.x + 22} y1={y - 16} x2={left.x + left.w - 22} y2={y - 16} stroke="var(--line-faint)" />
          <circle cx={left.x + 28} cy={y + 6} r={4} fill={`var(--fam-${index})`} />
          <Caption x={left.x + 43} y={y} text={evidence?.leaf ?? lane.label} size={12} weight={550} fill="var(--text)" />
          <Caption x={left.x + 43} y={y + 18} text={evidence?.reading ?? lane.key} size={10.5} />
        </g>
      })}
      <Panel box={right} radius={16} lift="soft" stroke="var(--accent-200)" />
      <Caption x={right.x + right.w / 2} y={right.y + 28} text={shape?.label ?? 'Prediction'} size={12} weight={650} anchor="middle" fill="var(--accent)" />
      <Caption x={right.x + right.w / 2} y={right.y + 66} text={bar < 0.15 ? 'Resolving prediction' : 'Predicted label: target'} size={17} weight={550} anchor="middle" fill="var(--text)" />
      <Caption x={right.x + right.w / 2} y={right.y + 124} text={(PREDICTION.probability * bar).toFixed(2)} size={46} weight={500} anchor="middle" fill="var(--text)" />
      <Caption x={right.x + right.w / 2} y={right.y + 158} text="PROBABILITY" size={9.5} weight={650} anchor="middle" />
      <rect x={right.x + 30} y={right.y + 184} width={right.w - 60} height={7} rx={3.5} fill="var(--bg-inset)" />
      <rect x={right.x + 30} y={right.y + 184} width={(right.w - 60) * PREDICTION.probability * bar} height={7} rx={3.5} fill="var(--accent)" />
      <line x1={right.x + 24} y1={right.y + 214} x2={right.x + right.w - 24} y2={right.y + 214} stroke="var(--line)" />
      <g opacity={confidence}>
        <Caption x={right.x + right.w / 2} y={right.y + 247} text={`Confidence ${(PREDICTION.confidence * confidence).toFixed(2)}`} size={15} weight={550} anchor="middle" fill="var(--text)" />
        <Caption x={right.x + right.w / 2} y={right.y + 276} text="Probability and confidence are distinct estimates." size={10.5} anchor="middle" />
        <Caption x={right.x + right.w / 2} y={right.y + 309} text={`${PREDICTION.supporting} supporting, ${PREDICTION.contradicting} contradicting`} size={11} anchor="middle" />
      </g>
    </g>
  </g>
}

/* --- Act 7: evaluation ----------------------------------------------------- */

export function SceneEvaluation({ p, L, v }: SceneProps): JSX.Element {
  const centre = L.scene.x + L.scene.w / 2
  const listW = Math.min(620, L.scene.w - 80)
  const listX = centre - listW / 2
  const listY = L.scene.y + PLATE_H + 36
  const rowH = 36
  const listH = CHECKLIST.length * rowH + 62
  const secondPass = p >= 0.64
  const routeY = listY + listH + 84
  const leftX = L.scene.x + L.scene.w * 0.25 - PLATE_W / 2
  const rightX = L.scene.x + L.scene.w * 0.75 - PLATE_W / 2
  const feedback = rise(p, 0.43, 0.61)
  const accepted = rise(p, 0.86, 0.98)
  return <g>
    <AgentPlate x={centre - PLATE_W / 2} y={L.scene.y} label={v.agents.critic.label}
      summary={v.agents.critic.summary} appear={rise(p, 0, 0.12)} glow={p < 0.45 ? 0.3 : 0} />
    <path d={`M ${centre} ${L.scene.y + PLATE_H} V ${listY}`} stroke="var(--accent)" strokeWidth={1.4} opacity={rise(p, 0.08, 0.2) * 0.5} />
    <g opacity={rise(p, 0.04, 0.16)}>
      <Panel box={{ x: listX, y: listY, w: listW, h: listH }} radius={16} lift="soft" />
      <Caption x={centre} y={listY + 26} text={secondPass ? 'CHECKLIST, SECOND PASS' : 'CHECKLIST, FIRST PASS'} size={10.5} weight={650} anchor="middle" />
      {CHECKLIST.map((item, index) => {
        const y = listY + 48 + index * rowH
        const start = secondPass ? 0.65 + index * 0.045 : 0.12 + index * 0.05
        const resolved = seg(p, start, start + 0.05)
        const passed = secondPass || item.first === 'pass'
        return <g key={item.label}>
          {!passed && resolved > 0.5 && <rect x={listX + 13} y={y - 5} width={listW - 26} height={32} rx={7} fill="var(--caution-soft)" opacity={0.55} />}
          {resolved > 0.5 ? <Mark kind={passed ? 'check' : 'warn'} x={listX + 31} y={y + 10} /> : <circle cx={listX + 31} cy={y + 10} r={5} fill="none" stroke="var(--line-strong)" />}
          <foreignObject x={listX + 51} y={y - 3} width={listW - 118} height={30}>
            <div className="film__check-label">{item.label}</div>
          </foreignObject>
          {resolved > 0.5 && <Caption x={listX + listW - 22} y={y + 10} text={passed ? 'pass' : 'revise'} size={10} anchor="end" fill={passed ? 'var(--text-muted)' : 'var(--caution)'} />}
        </g>
      })}
    </g>
    {[false, true].map((positive) => {
      const x = positive ? rightX : leftX
      const colour = positive ? 'var(--positive)' : 'var(--caution)'
      const amount = positive ? Math.max(feedback * 0.25, accepted) : feedback * (secondPass ? 0.58 : 1)
      const startX = listX + listW * (positive ? 0.75 : 0.25)
      return <g key={String(positive)} opacity={amount}>
        <path d={`M ${startX} ${listY + listH} C ${startX} ${routeY - 30}, ${x + PLATE_W / 2} ${routeY - 30}, ${x + PLATE_W / 2} ${routeY}`}
          fill="none" stroke={colour} strokeWidth={1.5} />
        <path d={`M ${x + PLATE_W / 2 - 4} ${routeY - 6} l 4 6 l 4 -6`} fill="none" stroke={colour} strokeWidth={1.5} />
        <rect x={x + 38} y={routeY - 49} width={PLATE_W - 76} height={24} rx={12} fill="var(--bg-elevated)" stroke={colour} />
        <Caption x={x + PLATE_W / 2} y={routeY - 37} text={positive ? 'SATISFACTORY' : 'UNSATISFACTORY'} size={10} weight={650} fill={colour} anchor="middle" />
        <AgentPlate x={x} y={routeY} label={positive ? v.agents.communicator.label : v.agents.orchestrator.label}
          summary={positive ? v.agents.communicator.summary : 'Revises the plan using the Critic’s feedback.'} appear={1} glow={positive ? accepted * 0.2 : 0} />
      </g>
    })}
    <Caption x={centre} y={routeY + PLATE_H + 28} text="Feedback returns to planning when another pass is allowed." size={11} anchor="middle" opacity={feedback} />
  </g>
}

/* --- Act 8: communication -------------------------------------------------- */

export function SceneCommunication({ p, L, v }: SceneProps): JSX.Element {
  const agent = v.agents.communicator
  const pageW = Math.min(530, L.scene.w)
  const pageH = 500
  const pageX = L.scene.x + (L.scene.w - pageW) / 2
  const pageY = L.scene.y + PLATE_H + 32
  const frame = rise(p, 0, 0.12)
  const plateX = L.scene.x + (L.scene.w - PLATE_W) / 2
  const plateY = L.scene.y

  let cursor = pageY + 52
  const blocks: ReactNode[] = []

  REPORT_SECTIONS.forEach((section, sectionIndex) => {
    const at = 0.12 + sectionIndex * 0.12
    blocks.push(
      <Caption
        key={section.title}
        x={pageX + 24}
        y={cursor}
        text={fit(section.title, pageW - 48, 11.5)}
        size={11.5}
        weight={650}
        fill="var(--text)"
        opacity={rise(p, at, at + 0.06)}
      />,
    )
    cursor += 14
    for (let line = 0; line < section.lines; line += 1) {
      const grow = easeOut(seg(p, at + 0.03 + line * 0.02, at + 0.12 + line * 0.02))
      const width = (pageW - 48) * (line === section.lines - 1 ? 0.62 : 1)
      blocks.push(
        <rect
          key={`${section.title}-${line}`}
          x={pageX + 24}
          y={cursor}
          width={Math.max(0, width * grow)}
          height={5}
          rx={2.5}
          fill="var(--line)"
        />,
      )
      cursor += 11
    }
    cursor += 12

    if (sectionIndex === 1) {
      const figure = rise(p, 0.4, 0.64)
      const figH = 78
      const base = cursor + figH - 12
      blocks.push(
        <g key="figure" opacity={figure}>
          <rect x={pageX + 24} y={cursor} width={pageW - 48} height={figH} rx={7} fill="var(--bg-sunken)" />
          {LANES.map((lane, index) => {
            const barW = (pageW - 72) / LANES.length
            const height = (10 + (lane.present / lane.total) * 40) * figure
            return (
              <rect
                key={lane.key}
                x={pageX + 36 + index * barW}
                y={base - height}
                width={barW - 10}
                height={Math.max(1, height)}
                rx={2}
                fill={`var(--fam-${index % 8})`}
                opacity={0.75}
              />
            )
          })}
          <Caption x={pageX + 24} y={cursor + figH + 12} text="Figure 1. Evidence available by domain." size={9.5} fill="var(--text-faint)" />
        </g>,
      )
      cursor += figH + 24
    }
  })

  UNKNOWNS.forEach((unknown, index) => {
    const appear = rise(p, 0.72 + index * 0.07, 0.86 + index * 0.07)
    const y = cursor + 14
    blocks.push(
      <g key={unknown} opacity={appear}>
        <Panel
          box={{ x: pageX + 24, y: cursor, w: pageW - 48, h: 28 }}
          radius={9}
          fill="var(--bg-sunken)"
          stroke="var(--line-strong)"
          dashed
        />
        <rect x={pageX + 33} y={y - 4} width={8} height={8} rx={2} fill="none" stroke="var(--text-faint)" strokeDasharray="2 2" />
        <foreignObject x={pageX + 48} y={cursor + 2} width={pageW - 72} height={24}>
          <div className="film__unknown-label">{unknown}</div>
        </foreignObject>
      </g>,
    )
    cursor += 32
  })

  return (
    <g>
      <g opacity={frame}>
        <Panel box={{ x: pageX, y: pageY, w: pageW, h: pageH }} radius={10} />
        <Caption x={pageX + 24} y={pageY + 26} text="DEEP PHENOTYPE REPORT" size={9.5} weight={700} />
        {blocks}
      </g>

      <AgentPlate x={plateX} y={plateY} label={agent.label} summary={agent.summary} appear={rise(p, 0.02, 0.16)} />
      <path d={`M ${plateX + PLATE_W / 2} ${plateY + PLATE_H} V ${pageY}`}
        fill="none" stroke="var(--accent)" strokeWidth={1.4} opacity={rise(p, 0.1, 0.24) * 0.6} />
      <foreignObject x={pageX} y={pageY + pageH + 12} width={pageW} height={38} opacity={rise(p, 0.84, 1)}>
        <div className="film__report-caption">Written from {v.steps.length} step outputs, with {UNKNOWNS.length} gaps marked unknown.</div>
      </foreignObject>
    </g>
  )
}

/* --- Persistent chrome ----------------------------------------------------- */

export function StageHud({ L, act, p, v }: { L: StageLayout; act: number; p: number; v: FilmVocabulary }): JSX.Element {
  const shape = v.shapes[v.chosenShape]
  const targetIn = act > 0 ? 1 : rise(p, 0.7, 0.94)

  // One thing only: what this run is being asked. The coverage dial and the
  // iteration badge that used to sit beside it were counters about the
  // illustration rather than part of the explanation, and they competed with
  // the diagram for the top of the stage.
  const chipW = Math.min(L.hud.w, L.wide ? 360 : L.hud.w)
  const chipX = L.hud.x + (L.hud.w - chipW) / 2

  return (
    <g>
      {targetIn > 0 && (
        <g opacity={targetIn} transform={`translate(0 ${(1 - targetIn) * -6})`}>
          <Panel
            box={{ x: chipX, y: L.hud.y, w: chipW, h: L.hud.h }}
            radius={10}
            fill="var(--accent-50)"
            stroke="var(--accent-200)"
            lift="soft"
          />
          <Caption
            x={chipX + chipW / 2}
            y={L.hud.y + 15}
            text="THE RUN'S SUBJECT"
            size={9}
            weight={700}
            anchor="middle"
            fill="var(--accent)"
          />
          <Caption
            x={chipX + chipW / 2}
            y={L.hud.y + 31}
            text={fit(shape?.label ?? 'Binary classification', chipW - 28, 13)}
            size={13}
            weight={600}
            anchor="middle"
            fill="var(--text)"
          />
        </g>
      )}
    </g>
  )
}

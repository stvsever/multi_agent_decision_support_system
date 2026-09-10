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
const PLATE_W = 210
const PLATE_H = 72

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
      <Caption x={box.x + 14} y={box.y + 21} text={label} size={13.5} weight={600} fill="var(--text)" />
      <foreignObject x={box.x + 14} y={box.y + 30} width={box.w - 28} height={box.h - 36}>
        <div className="film__plate-note">{summary}</div>
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
  const titleSize = compact ? 11.5 : 12
  // Left padding plus the status mark on the right, and nothing more: the old
  // reserve was wide enough to truncate names that had room to spare.
  const room = box.w - 46
  const dim = state === 'pending' ? 0.42 : state === 'queued' ? 0.78 : 1
  const offset = compact ? 7 : 9
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
      <Caption
        x={box.x + 12}
        y={box.y + box.h / 2 - offset}
        text={fit(step.toolName, room, titleSize)}
        size={titleSize}
        weight={600}
        fill="var(--text)"
      />
      <Caption
        x={box.x + 12}
        y={box.y + box.h / 2 + offset}
        text={fit(step.subject, room, 10.5)}
        size={10.5}
        fill="var(--text-muted)"
      />
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
        <circle cx={box.x + box.w - 15} cy={box.y + box.h / 2} r={3.5} fill="var(--accent)">
          <animate attributeName="opacity" values="1;0.25;1" dur="1.2s" repeatCount="indefinite" />
        </circle>
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

/** A dependency edge, drawn on as `draw` runs from 0 to 1. */
function Edge({ id, from, to, draw, compact }: { id: string; from: Box; to: Box; draw: number; compact?: boolean }) {
  if (draw <= 0 || !from || !to) return null
  const bend = compact ? 14 : Math.max(16, (to.x - (from.x + from.w)) * 0.6)
  const path = compact
    ? `M ${from.x + from.w / 2} ${from.y + from.h} C ${from.x + from.w / 2} ${from.y + from.h + bend}, ${
        to.x + to.w / 2
      } ${to.y - bend}, ${to.x + to.w / 2} ${to.y}`
    : `M ${from.x + from.w} ${from.y + from.h / 2} C ${from.x + from.w + bend} ${from.y + from.h / 2}, ${
        to.x - bend
      } ${to.y + to.h / 2}, ${to.x} ${to.y + to.h / 2}`
  const complete = clamp01(draw) >= 0.999
  const startX = compact ? from.x + from.w / 2 : from.x + from.w
  const startY = compact ? from.y + from.h : from.y + from.h / 2
  return (
    <g>
      <path
        id={id}
        d={path}
        fill="none"
        stroke="var(--line-strong)"
        strokeWidth={1}
        pathLength={1}
        strokeDasharray={1}
        strokeDashoffset={1 - clamp01(draw)}
        opacity={0.85}
      />
      {/* Direction, once the line has finished drawing. A static curve between
          two boxes says they are related but not which way the work flows, and
          on a graph with two fusion layers that is the whole point. Markers
          ride the path itself, so they follow whatever curve it takes. */}
      {complete &&
        [0, 1, 2].map((slot) => (
          // Parked on the edge's origin rather than at the default 0,0: if the
          // motion never runs, a marker should sit where the work starts, not
          // in the corner of the stage.
          <circle
            key={slot}
            cx={startX}
            cy={startY}
            r={compact ? 1.8 : 2.1}
            fill="var(--accent)"
            opacity={0.9}
          >
            <animateMotion
              dur="1.9s"
              begin={`${slot * 0.63}s`}
              repeatCount="indefinite"
              keyPoints="0;1"
              keyTimes="0;1"
              calcMode="linear"
            >
              <mpath href={`#${id}`} />
            </animateMotion>
            <animate
              attributeName="opacity"
              dur="1.9s"
              begin={`${slot * 0.63}s`}
              repeatCount="indefinite"
              values="0;0.95;0.95;0"
              keyTimes="0;0.15;0.8;1"
            />
          </circle>
        ))}
    </g>
  )
}

/** A short connector between an agent plate and the thing it produces. */
function Feed({ from, to, y, draw }: { from: number; to: number; y: number; draw: number }) {
  if (draw <= 0) return null
  return (
    <path
      d={`M ${from} ${y} L ${to} ${y}`}
      stroke="var(--line-strong)"
      strokeWidth={1}
      pathLength={1}
      strokeDasharray={1}
      strokeDashoffset={1 - clamp01(draw)}
    />
  )
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
    const colGap = 20
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
    const nodeH = maxRows >= 6 ? 38 : maxRows === 5 ? 42 : 46
    const rowGap = maxRows >= 6 ? 9 : maxRows === 5 ? 11 : 14
    const blockH = maxRows * nodeH + (maxRows - 1) * rowGap
    const bound = L.work.y + (agentAbove ? PLATE_H + 16 : 0)
    const top = bound + Math.max(18, (L.work.y + L.work.h - bound - blockH) / 2)

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
        ? { x: L.work.x, y: L.work.y, w: PLATE_W, h: PLATE_H }
        : { x: L.work.x, y: top + blockH / 2 - PLATE_H / 2, w: PLATE_W, h: PLATE_H },
    }
  }

  const nodeW = L.work.w
  const nodeH = 34
  const gap = 5
  const labelH = 20
  let cursor = L.work.y + PLATE_H - 4
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

/* --- Act 1: the question --------------------------------------------------- */

export function SceneQuestion({ p, L, v }: SceneProps): JSX.Element {
  const shapes = v.shapes
  const chosen = v.chosenShape
  const count = Math.max(1, shapes.length)
  const gap = L.wide ? 14 : 10
  const columns = L.wide ? count : 2
  const rows = Math.ceil(count / columns)
  const cardW = (L.scene.w - gap * (columns - 1)) / columns
  const cardH = L.wide ? 118 : 84
  const gridH = rows * cardH + (rows - 1) * gap
  // The grid is the whole act now, so it sits in the middle of the stage
  // rather than at the top of a column it used to share with a contract panel.
  const top = L.scene.y + Math.max(6, (L.scene.h - gridH) / 2)

  const boxFor = (index: number): Box => ({
    x: L.scene.x + (index % columns) * (cardW + gap),
    y: top + Math.floor(index / columns) * (cardH + gap),
    w: cardW,
    h: cardH,
  })

  // The highlight visits a few shapes before it settles on the chosen one.
  const tour = [count - 1, Math.max(0, count - 3), 1, chosen, chosen]
  const travel = seg(p, 0.3, 0.66)
  const legIndex = Math.min(tour.length - 2, Math.floor(travel * (tour.length - 1)))
  const legT = easeInOut(clamp01(travel * (tour.length - 1) - legIndex))
  const fromBox = boxFor(tour[legIndex])
  const toBox = boxFor(tour[legIndex + 1])
  const highlight = { x: lerp(fromBox.x, toBox.x, legT), y: lerp(fromBox.y, toBox.y, legT) }
  const settled = seg(p, 0.62, 0.78)

  return (
    <g>
      {shapes.map((shape, index) => {
        const box = boxFor(index)
        const enter = rise(p, 0.02 + index * 0.05, 0.3 + index * 0.05)
        const isChosen = index === chosen
        const fade = isChosen ? 1 : lerp(1, 0.28, settled)
        return (
          <g key={shape.value} opacity={enter * fade} transform={`translate(0 ${(1 - enter) * 12})`}>
            <Panel
              box={box}
              radius={11}
              stroke={isChosen && settled > 0.4 ? 'var(--accent)' : 'var(--line)'}
              lift={isChosen && settled > 0.4 ? 'soft' : 'none'}
            />
            <Caption
              x={box.x + box.w / 2}
              y={box.y + box.h / 2 - 12}
              text={fit(shape.label, box.w - 24, 13)}
              size={13}
              weight={600}
              anchor="middle"
              fill="var(--text)"
            />
            <foreignObject x={box.x + 12} y={box.y + box.h / 2 - 2} width={Math.max(10, box.w - 24)} height={L.wide ? 44 : 30}>
              <div className="film__card-note film__card-note--center">{shape.summary}</div>
            </foreignObject>
          </g>
        )
      })}

      {travel > 0 && settled < 1 && (
        <rect
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
              draw={rise(p, 0.24 + index * 0.05, 0.4 + index * 0.05)}
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
          appear={rise(p, 0.18 + index * 0.05, 0.34 + index * 0.05)}
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
      {v.steps.map((step) =>
        step.deps.map((dep) => (
          <Edge
            key={`${dep}-${step.id}`}
            id={`exec-${dep}-${step.id}`}
            from={geometry.pos[dep]}
            to={geometry.pos[step.id]}
            draw={1}
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

interface SpineSlot {
  key: string
  step: ResolvedStep
  label: string
  box: Box
  arrive: number
}

function spineSlots(L: StageLayout, steps: ResolvedStep[]): SpineSlot[] {
  const segments: { step: ResolvedStep; label: string }[] = []
  for (const step of steps) {
    if (step.chunked) {
      segments.push({ step, label: `${step.toolName}, chunk 1/2` })
      segments.push({ step, label: `${step.toolName}, chunk 2/2` })
    } else {
      segments.push({ step, label: step.toolName })
    }
  }
  const h = 18
  const gap = 4
  const blockH = segments.length * h + (segments.length - 1) * gap
  const w = L.wide ? 300 : Math.min(320, L.scene.w - 24)
  const x = L.scene.x + L.scene.w / 2 - w / 2
  const top = L.scene.y + Math.max(14, (L.scene.h - blockH) / 2 - 8)
  return segments.map((segment, index) => ({
    key: `${segment.step.id}-${index}`,
    step: segment.step,
    label: segment.label,
    box: { x, y: top + index * (h + gap), w, h },
    arrive: 0.1 + index * 0.05,
  }))
}

export function SceneIntegration({ p, L, v }: SceneProps): JSX.Element {
  const geometry = planGeometry(L, v.steps)
  const slots = spineSlots(L, v.steps)
  const agent = v.agents.integrator
  const first = slots[0].box
  const last = slots[slots.length - 1].box
  const spineTop = first.y - 10
  const spineBottom = last.y + last.h + 10
  const middle = (spineTop + spineBottom) / 2

  return (
    <g>
      <rect
        x={first.x - 10}
        y={spineTop}
        width={first.w + 20}
        height={spineBottom - spineTop}
        rx={12}
        fill="var(--bg-inset)"
        stroke="var(--line-faint)"
        opacity={rise(p, 0, 0.12)}
      />

      {slots.map((slot) => {
        const from = geometry.pos[slot.step.id]
        const travel = easeInOut(seg(p, slot.arrive, slot.arrive + 0.18))
        if (travel <= 0 || !from) return null
        const box: Box = {
          x: lerp(from.x, slot.box.x, travel),
          y: lerp(from.y, slot.box.y, travel),
          w: lerp(from.w, slot.box.w, travel),
          h: lerp(from.h, slot.box.h, travel),
        }
        const arrived = travel > 0.98
        return (
          <g key={slot.key}>
            <rect
              x={box.x}
              y={box.y}
              width={box.w}
              height={box.h}
              rx={lerp(9, 5, travel)}
              fill="var(--bg-elevated)"
              stroke={arrived ? 'var(--line-faint)' : 'var(--line)'}
            />
            <rect
              x={box.x}
              y={box.y + 3}
              width={3}
              height={Math.max(2, box.h - 6)}
              rx={1.5}
              fill={`var(--fam-${slot.step.familyIndex})`}
            />
            <Caption
              x={box.x + 11}
              y={box.y + box.h / 2}
              text={fit(arrived ? slot.label : slot.step.toolName, box.w - 66, 10.5)}
              size={10.5}
              fill="var(--text-secondary)"
              opacity={travel}
            />
            <Caption
              x={box.x + box.w - 9}
              y={box.y + box.h / 2}
              text={slot.step.chunked ? 'split' : ''}
              size={9.5}
              anchor="end"
              mono
              fill="var(--text-faint)"
              opacity={travel}
            />
          </g>
        )
      })}

      {L.wide && (
        <>
          <AgentPlate
            x={first.x - 34 - PLATE_W}
            y={middle - PLATE_H / 2}
            label={agent.label}
            summary={agent.summary}
            appear={rise(p, 0.04, 0.2)}
          />
          <g opacity={rise(p, 0.5, 0.68)}>
            <Caption
              x={first.x + first.w + 24}
              y={middle + 4}
              text="one output too large to pass whole"
              size={10.5}
              fill="var(--caution)"
              weight={600}
            />
            <Caption x={first.x + first.w + 24} y={middle + 20} text="is split into two chunks" size={10.5} />
          </g>
        </>
      )}

      <Caption
        x={L.scene.x + L.scene.w / 2}
        y={Math.min(spineBottom + 22, L.scene.y + L.scene.h - 8)}
        text={`one evidence bundle, ${slots.length} blocks`}
        size={11.5}
        anchor="middle"
        opacity={rise(p, 0.82, 1)}
      />
    </g>
  )
}

/* --- Act 6: prediction ----------------------------------------------------- */

export function ScenePrediction({ p, L, v }: SceneProps): JSX.Element {
  const agent = v.agents.predictor
  const shape = v.shapes[v.chosenShape]
  const laneH = 34
  const laneGap = 8
  const laneW = L.wide ? 250 : L.scene.w
  const laneX = L.scene.x
  const laneBlock = LANES.length * laneH + (LANES.length - 1) * laneGap

  const cardW = L.wide ? Math.min(430, L.scene.w - laneW - 300) : L.scene.w
  const cardX = L.scene.x + L.scene.w - cardW
  const cardH = 250
  const cardY = L.wide ? L.scene.y + (L.scene.h - cardH) / 2 : L.scene.y + laneBlock + 44
  const laneTop = L.wide ? cardY + cardH / 2 - laneBlock / 2 : L.scene.y

  const plateX = laneX + laneW + (cardX - laneX - laneW - PLATE_W) / 2
  const plateY = cardY + cardH / 2 - PLATE_H / 2
  const frame = rise(p, 0, 0.16)
  const decided = seg(p, 0.2, 0.42)
  const bar = easeInOut(seg(p, 0.32, 0.6))
  const tail = rise(p, 0.88, 1)
  const rowY = (index: number) => cardY + 142 + index * 30

  return (
    <g>
      {LANES.map((lane, index) => {
        const y = laneTop + index * (laneH + laneGap)
        const cited = EVIDENCE.findIndex((row) => row.lane === lane.key)
        const lit = cited >= 0 ? rise(p, 0.46 + cited * 0.12, 0.62 + cited * 0.12) : 0
        return (
          <g key={lane.key} opacity={frame}>
            <Panel
              box={{ x: laneX, y, w: laneW, h: laneH }}
              radius={8}
              stroke={lit > 0.5 ? 'var(--accent)' : 'var(--line-faint)'}
              fill={lit > 0.5 ? 'var(--accent-50)' : 'var(--bg-elevated)'}
            />
            <Caption
              x={laneX + 12}
              y={y + laneH / 2}
              text={fit(cited >= 0 ? EVIDENCE[cited].leaf : lane.label, laneW - 24, 11.5)}
              size={11.5}
              weight={cited >= 0 ? 600 : 400}
              fill={cited >= 0 ? 'var(--text)' : 'var(--text-muted)'}
            />
          </g>
        )
      })}

      {/* Evidence flows through the Predictor, which is why the plate sits on
          the path rather than beside it. */}
      {L.wide &&
        EVIDENCE.map((row, index) => {
          const laneIndex = LANES.findIndex((lane) => lane.key === row.lane)
          if (laneIndex < 0) return null
          const fromY = laneTop + laneIndex * (laneH + laneGap) + laneH / 2
          const toY = plateY + 18 + index * 18
          const draw = rise(p, 0.44 + index * 0.1, 0.64 + index * 0.1)
          return (
            <path
              key={row.leaf}
              d={`M ${laneX + laneW} ${fromY} C ${laneX + laneW + 40} ${fromY}, ${plateX - 40} ${toY}, ${plateX} ${toY}`}
              fill="none"
              stroke="var(--accent)"
              strokeWidth={1.2}
              opacity={0.5}
              pathLength={1}
              strokeDasharray={1}
              strokeDashoffset={1 - draw}
            />
          )
        })}

      {L.wide && (
        <>
          <AgentPlate x={plateX} y={plateY} label={agent.label} summary={agent.summary} appear={rise(p, 0.04, 0.2)} />
          <Feed from={plateX + PLATE_W} to={cardX} y={plateY + PLATE_H / 2} draw={rise(p, 0.2, 0.34)} />
        </>
      )}

      <g opacity={frame}>
        <Panel box={{ x: cardX, y: cardY, w: cardW, h: cardH }} radius={12} />
        <Caption
          x={cardX + 18}
          y={cardY + 22}
          text={fit((shape?.label ?? 'Prediction').toUpperCase(), cardW - 36, 9.5)}
          size={9.5}
          weight={700}
        />
        <Caption
          x={cardX + 18}
          y={cardY + 48}
          text={decided < 0.5 ? 'resolving' : 'Predicted label: target'}
          size={16}
          weight={650}
          fill={decided < 0.5 ? 'var(--text-faint)' : 'var(--text)'}
        />
        <rect x={cardX + 18} y={cardY + 66} width={cardW - 36} height={8} rx={4} fill="var(--bg-inset)" />
        <rect
          x={cardX + 18}
          y={cardY + 66}
          width={Math.max(0, (cardW - 36) * PREDICTION.probability * bar)}
          height={8}
          rx={4}
          fill="var(--accent)"
        />
        <Caption
          x={cardX + 18}
          y={cardY + 88}
          text={`probability ${(PREDICTION.probability * bar).toFixed(2)}`}
          size={11}
          mono
          fill="var(--text-secondary)"
        />
        <Caption
          x={cardX + cardW - 18}
          y={cardY + 88}
          text={`confidence ${(PREDICTION.confidence * tail).toFixed(2)}`}
          size={11}
          mono
          anchor="end"
          fill="var(--text-secondary)"
          opacity={tail}
        />
        <line x1={cardX + 18} y1={cardY + 104} x2={cardX + cardW - 18} y2={cardY + 104} stroke="var(--line-faint)" />
        <Caption x={cardX + 18} y={cardY + 120} text="EVIDENCE CHAIN" size={9.5} weight={700} />

        {EVIDENCE.map((row, index) => {
          const y = rowY(index)
          const appear = rise(p, 0.56 + index * 0.1, 0.72 + index * 0.1)
          return (
            <g key={row.leaf} opacity={appear}>
              <circle cx={cardX + 24} cy={y} r={3.5} fill="var(--accent)" />
              <Caption x={cardX + 36} y={y - 6} text={fit(row.leaf, cardW - 124, 11.5)} size={11.5} weight={600} fill="var(--text)" />
              <Caption x={cardX + 36} y={y + 8} text={fit(row.reading, cardW - 124, 10.5)} size={10.5} />
              <rect x={cardX + cardW - 62} y={y - 3} width={44} height={6} rx={3} fill="var(--bg-inset)" />
              <rect x={cardX + cardW - 62} y={y - 3} width={44 * row.weight * appear} height={6} rx={3} fill="var(--accent)" opacity={0.7} />
            </g>
          )
        })}

        <Caption
          x={cardX + 18}
          y={cardY + cardH - 16}
          text={`${PREDICTION.supporting} supporting, ${PREDICTION.contradicting} contradicting`}
          size={10.5}
          fill="var(--text-faint)"
          opacity={tail}
        />
      </g>
    </g>
  )
}

/* --- Act 7: evaluation ----------------------------------------------------- */

export function SceneEvaluation({ p, L, v }: SceneProps): JSX.Element {
  const agent = v.agents.critic
  const rowH = 32
  const listW = L.wide ? Math.min(460, L.scene.w - 340) : L.scene.w
  const listX = L.wide ? L.scene.x + L.scene.w - listW - 40 : L.scene.x
  const listH = CHECKLIST.length * rowH + 84
  const listY = L.scene.y + Math.max(10, (L.scene.h - listH) / 2 - 14)

  const secondPass = p >= 0.6
  const verdictOne = rise(p, 0.4, 0.5)
  const loop = rise(p, 0.48, 0.64)
  const verdictTwo = rise(p, 0.88, 1)
  const verdictW = secondPass ? 148 : 172
  const verdictX = listX + listW - 16 - verdictW
  const verdictY = listY + listH - 44
  const loopY = listY + listH - 12
  const loopEndX = L.wide ? L.scene.x + 24 : L.scene.x + 6

  return (
    <g>
      <g opacity={rise(p, 0, 0.12)}>
        <Panel box={{ x: listX, y: listY, w: listW, h: listH }} radius={12} />
        <Caption
          x={listX + 18}
          y={listY + 24}
          text={secondPass ? 'CHECKLIST, SECOND PASS' : 'CHECKLIST, FIRST PASS'}
          size={9.5}
          weight={700}
        />
        {CHECKLIST.map((item, index) => {
          const y = listY + 52 + index * rowH
          const from = secondPass ? 0.64 + index * 0.05 : 0.12 + index * 0.05
          const resolved = seg(p, from, from + 0.08)
          const outcome = secondPass ? item.second : item.first
          return (
            <g key={item.label}>
              <Caption
                x={listX + 42}
                y={y}
                text={fit(item.label, listW - 70, 12)}
                size={12}
                fill={resolved > 0.5 ? 'var(--text)' : 'var(--text-faint)'}
              />
              {resolved > 0.5 ? (
                <Mark kind={outcome === 'pass' ? 'check' : 'cross'} x={listX + 24} y={y} />
              ) : (
                <circle cx={listX + 24} cy={y} r={5} fill="none" stroke="var(--line-strong)" strokeDasharray="2 2" />
              )}
            </g>
          )
        })}
      </g>

      {!secondPass && verdictOne > 0 && (
        <g opacity={verdictOne}>
          <Panel
            box={{ x: verdictX, y: verdictY, w: verdictW, h: 28 }}
            radius={14}
            fill="var(--caution-soft)"
            stroke="var(--caution)"
          />
          <Caption
            x={verdictX + verdictW / 2}
            y={verdictY + 14}
            text="NEEDS WORK, ITERATE"
            size={11}
            weight={700}
            anchor="middle"
            fill="var(--caution)"
          />
        </g>
      )}

      {verdictTwo > 0 && (
        <g opacity={verdictTwo}>
          <Panel
            box={{ x: verdictX, y: verdictY, w: verdictW, h: 28 }}
            radius={14}
            fill="var(--positive-soft)"
            stroke="var(--positive)"
          />
          <Caption
            x={verdictX + verdictW / 2}
            y={verdictY + 14}
            text="SATISFACTORY"
            size={11}
            weight={700}
            anchor="middle"
            fill="var(--positive)"
          />
        </g>
      )}

      {!secondPass && loop > 0 && (
        <g>
          <path
            d={`M ${listX} ${loopY} C ${listX - 60} ${loopY + 34}, ${loopEndX - 20} ${loopY + 34}, ${loopEndX} ${
              loopY - 2
            }`}
            fill="none"
            stroke="var(--caution)"
            strokeWidth={1.4}
            pathLength={1}
            strokeDasharray={1}
            strokeDashoffset={1 - loop}
          />
          <path
            d={`M ${loopEndX - 4} ${loopY + 6} L ${loopEndX} ${loopY - 4} L ${loopEndX + 5} ${loopY + 5}`}
            fill="none"
            stroke="var(--caution)"
            strokeWidth={1.4}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={seg(loop, 0.86, 1)}
          />
          <Caption
            x={loopEndX + 14}
            y={loopY + 30}
            text="back to the Orchestrator"
            size={10.5}
            weight={600}
            fill="var(--caution)"
            opacity={seg(loop, 0.5, 0.9)}
          />
        </g>
      )}

      {L.wide && (
        <AgentPlate
          x={L.scene.x}
          y={listY + listH / 2 - PLATE_H / 2}
          label={agent.label}
          summary={agent.summary}
          appear={rise(p, 0.02, 0.16)}
        />
      )}

      <Caption
        x={listX + listW / 2}
        y={listY - 20}
        text={
          secondPass
            ? 'The second pass clears every check, so the run moves on.'
            : 'Two checks fail, so the plan is revisited rather than the answer being shipped.'
        }
        size={11.5}
        anchor="middle"
        opacity={rise(p, 0.24, 0.36)}
      />
    </g>
  )
}

/* --- Act 8: communication -------------------------------------------------- */

export function SceneCommunication({ p, L, v }: SceneProps): JSX.Element {
  const agent = v.agents.communicator
  const pageW = L.wide ? Math.min(440, L.scene.w - 320) : L.scene.w
  const pageH = Math.min(L.scene.h - 12, 430)
  const pageX = L.scene.x + L.scene.w - pageW - (L.wide ? 24 : 0)
  const pageY = L.scene.y + (L.scene.h - pageH) / 2
  const frame = rise(p, 0, 0.12)
  const plateX = pageX - 40 - PLATE_W
  const plateY = pageY + pageH / 2 - PLATE_H / 2

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
    const y = cursor + 9
    blocks.push(
      <g key={unknown} opacity={appear}>
        <Panel
          box={{ x: pageX + 24, y: cursor, w: pageW - 48, h: 18 }}
          radius={9}
          fill="var(--bg-sunken)"
          stroke="var(--line-strong)"
          dashed
        />
        <rect x={pageX + 33} y={y - 4} width={8} height={8} rx={2} fill="none" stroke="var(--text-faint)" strokeDasharray="2 2" />
        <Caption x={pageX + 48} y={y} text={fit(unknown, pageW - 82, 10.5)} size={10.5} />
      </g>,
    )
    cursor += 22
  })

  return (
    <g>
      <g opacity={frame}>
        <Panel box={{ x: pageX, y: pageY, w: pageW, h: pageH }} radius={10} />
        <Caption x={pageX + 24} y={pageY + 26} text="DEEP PHENOTYPE REPORT" size={9.5} weight={700} />
        {blocks}
      </g>

      {L.wide && (
        <>
          <AgentPlate x={plateX} y={plateY} label={agent.label} summary={agent.summary} appear={rise(p, 0.02, 0.16)} />
          <Feed from={plateX + PLATE_W} to={pageX} y={plateY + PLATE_H / 2} draw={rise(p, 0.1, 0.24)} />
          <Caption
            x={plateX}
            y={plateY + PLATE_H + 24}
            text={`Written from ${v.steps.length} step outputs,`}
            size={11.5}
            opacity={rise(p, 0.84, 1)}
          />
          <Caption
            x={plateX}
            y={plateY + PLATE_H + 40}
            text={`with ${UNKNOWNS.length} gaps marked unknown.`}
            size={11.5}
            opacity={rise(p, 0.86, 1)}
          />
        </>
      )}
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

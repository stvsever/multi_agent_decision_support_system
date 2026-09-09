/**
 * Zoomable sunburst.
 *
 * The partition is computed once. Zooming interpolates the focus rectangle and
 * writes arc geometry straight onto the mounted paths, so a transition over a
 * thousand nodes stays on one animation frame budget instead of a thousand
 * React updates.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { hierarchy, partition, type HierarchyRectangularNode } from 'd3-hierarchy'
import { arc as arcGenerator } from 'd3-shape'
import { number, signed } from '@/lib/format'
import { Breadcrumb, ChartTooltip } from './atoms'
import {
  ROOT_KEY,
  bandColor,
  prettyLabel,
  type HierarchyDatum,
  type OntologyIndex,
} from './model'
import { useAnimatedView, useMeasure, type View } from './useOntologyIndex'

/** Rings drawn outward from the focus. Beyond this the arcs are unreadable. */
const RINGS = 4
const TAU = Math.PI * 2

type Node = HierarchyRectangularNode<HierarchyDatum>

export interface PartitionViewProps {
  index: OntologyIndex
  data: HierarchyDatum
  matchBranch: Set<string> | null
  selected: string | null
  focusKey: string
  onSelect: (key: string) => void
  onFocusChange: (key: string) => void
  reducedMotion: boolean
}

export function SunburstView({
  index,
  data,
  matchBranch,
  selected,
  focusKey,
  onSelect,
  onFocusChange,
  reducedMotion,
}: PartitionViewProps) {
  const [wrapRef, size] = useMeasure<HTMLDivElement>()
  const width = Math.max(280, size.width || 640)
  const height = Math.max(300, Math.min(width, 540))
  const radius = Math.min(width, height) / 2 - 6
  const innerRadius = Math.max(46, radius * 0.24)
  const ringStep = (radius - innerRadius) / RINGS

  const layout = useMemo(() => {
    const root = hierarchy<HierarchyDatum>(data, (d) => d.kids).sum((d) =>
      d.kids.length ? 0 : Math.max(1, d.leaves),
    )
    return partition<HierarchyDatum>().size([TAU, root.height + 1])(root)
  }, [data])

  const nodes = useMemo(() => layout.descendants().filter((n) => n.depth > 0), [layout])
  const byKey = useMemo(() => {
    const map = new Map<string, Node>()
    map.set(ROOT_KEY, layout)
    for (const node of nodes) map.set(node.data.key, node)
    return map
  }, [layout, nodes])

  // Labelling every sliver would double the element count for no gain, so the
  // branches and the widest leaves carry text and the rest rely on the tooltip.
  const labelled = useMemo(() => {
    const set = new Set<string>()
    const leaves: Node[] = []
    for (const node of nodes) {
      if (node.children?.length) set.add(node.data.key)
      else leaves.push(node)
    }
    leaves.sort((a, b) => b.x1 - b.x0 - (a.x1 - a.x0))
    for (const leaf of leaves.slice(0, 300)) set.add(leaf.data.key)
    return set
  }, [nodes])

  const focus = byKey.get(focusKey) ?? layout
  const target = useMemo<View>(() => [focus.x0, focus.x1, focus.y0], [focus])

  const paths = useRef(new Map<string, SVGPathElement>())
  const labels = useRef(new Map<string, SVGTextElement>())
  const shown = useRef(new Map<string, boolean>())

  const arcOf = useMemo(
    () =>
      arcGenerator<{ a0: number; a1: number; r0: number; r1: number }>()
        .startAngle((d) => d.a0)
        .endAngle((d) => d.a1)
        .padAngle((d) => Math.min((d.a1 - d.a0) / 2, 0.0035))
        .padRadius(radius)
        .innerRadius((d) => d.r0)
        .outerRadius((d) => d.r1),
    [radius],
  )

  const apply = useCallback(
    (view: View) => {
      const [vx0, vx1, vy0] = view
      const scale = TAU / Math.max(vx1 - vx0, 1e-9)
      for (const node of nodes) {
        const key = node.data.key
        const element = paths.current.get(key)
        if (!element) continue
        const a0 = (node.x0 - vx0) * scale
        const a1 = (node.x1 - vx0) * scale
        const d0 = node.y0 - vy0
        const d1 = node.y1 - vy0
        // Thickness is measured inside the ring window, so the focus itself
        // collapses to nothing at rest and grows back out as the zoom reverses.
        const thickness = Math.min(RINGS + 1, d1) - Math.max(1, d0)
        const visible =
          thickness > 0.002 && a1 > 1e-3 && a0 < TAU - 1e-3 && a1 - a0 > 0.0018

        const label = labels.current.get(key)
        if (!visible) {
          if (shown.current.get(key) !== false) {
            element.style.display = 'none'
            if (label) label.style.display = 'none'
            shown.current.set(key, false)
          }
          continue
        }

        const start = Math.max(0, a0)
        const end = Math.min(TAU, a1)
        const inner = innerRadius + (Math.max(1, d0) - 1) * ringStep
        const outer = innerRadius + (Math.min(RINGS + 1, d1) - 1) * ringStep
        element.setAttribute(
          'd',
          arcOf({ a0: start, a1: end, r0: inner, r1: Math.max(inner + 0.6, outer) }) ?? '',
        )
        if (shown.current.get(key) !== true) {
          element.style.display = ''
          shown.current.set(key, true)
        }

        if (!label) continue
        const mid = (start + end) / 2
        const midRadius = (inner + outer) / 2
        const arcLength = (end - start) * midRadius
        if (arcLength < 30 || outer - inner < 13) {
          label.style.display = 'none'
          continue
        }
        const degrees = (mid * 180) / Math.PI - 90
        const flip = degrees > 90 || degrees < -90
        label.setAttribute(
          'transform',
          `rotate(${degrees}) translate(${midRadius},0) rotate(${flip ? 180 : 0})`,
        )
        const room = Math.floor(arcLength / 6.5)
        const text = node.data.label
        label.textContent = text.length > room ? `${text.slice(0, Math.max(1, room - 1))}…` : text
        label.style.display = ''
      }
    },
    [nodes, arcOf, innerRadius, ringStep],
  )

  useAnimatedView(target, apply, reducedMotion ? 0 : 620)

  // A selection made elsewhere should be on screen: if it sits outside the
  // current rings, pull the focus back to its parent so it appears in context.
  const revealed = useRef<string | null>(null)
  useEffect(() => {
    if (!selected || revealed.current === selected) return
    revealed.current = selected
    const node = byKey.get(selected)
    if (!node) return
    const within =
      node.y0 >= focus.y0 - 1e-9 &&
      node.y0 < focus.y0 + RINGS + 1 &&
      node.x0 >= focus.x0 - 1e-9 &&
      node.x1 <= focus.x1 + 1e-9
    if (!within) onFocusChange(node.parent?.data.key ?? ROOT_KEY)
  }, [selected, byKey, focus, onFocusChange])

  useEffect(() => {
    if (!byKey.has(focusKey)) onFocusChange(ROOT_KEY)
  }, [byKey, focusKey, onFocusChange])

  const [hover, setHover] = useState<{ key: string; x: number; y: number } | null>(null)

  const keyFromEvent = (event: React.MouseEvent): string | null => {
    const target = event.target as Element | null
    return target?.getAttribute?.('data-key') ?? null
  }

  const crumbs = useMemo(() => {
    const chain: { key: string; label: string }[] = []
    let cursor: Node | null = focus
    while (cursor) {
      chain.unshift({ key: cursor.data.key, label: cursor.data.label })
      cursor = cursor.parent
    }
    return chain
  }, [focus])

  const hovered = hover ? index.byKey.get(hover.key) : null
  const focusMagnitude = focus.data.magnitude

  return (
    <div className="onto-chart">
      <div className="onto-chart__head">
        <Breadcrumb chain={crumbs} onSelect={onFocusChange} compact />
        <span className="grow" />
        <span className="t-tiny muted">{number(focus.value ?? 0)} leaves in view</span>
      </div>
      <div
        ref={wrapRef}
        className="onto-chart__canvas"
        style={{ height }}
        onMouseMove={(event) => {
          const key = keyFromEvent(event)
          if (!key) {
            setHover(null)
            return
          }
          const box = event.currentTarget.getBoundingClientRect()
          setHover({ key, x: event.clientX - box.left + 12, y: event.clientY - box.top - 10 })
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(event) => {
          const key = keyFromEvent(event)
          if (!key) return
          onSelect(key)
          const node = byKey.get(key)
          if (node?.children?.length) onFocusChange(key)
        }}
      >
        <svg width={width} height={height} role="img" aria-label="Ontology sunburst">
          <g transform={`translate(${width / 2},${height / 2})`}>
            {nodes.map((node) => {
              const dim = matchBranch ? !matchBranch.has(node.data.key) : false
              const opacity = Math.max(0.42, 1 - (node.depth - 1) * 0.08) * (dim ? 0.22 : 1)
              return (
                <path
                  key={node.data.key}
                  data-key={node.data.key}
                  data-selected={selected === node.data.key || undefined}
                  className="onto-arc"
                  ref={(element) => {
                    if (element) paths.current.set(node.data.key, element)
                    else {
                      // Drop the cached visibility with the element, or a
                      // remount would inherit a stale display flag.
                      paths.current.delete(node.data.key)
                      shown.current.delete(node.data.key)
                    }
                  }}
                  style={{ fill: bandColor(node.data.band), fillOpacity: opacity, display: 'none' }}
                />
              )
            })}
            {nodes
              .filter((node) => labelled.has(node.data.key))
              .map((node) => (
                <text
                  key={node.data.key}
                  className="onto-arc__label"
                  ref={(element) => {
                    if (element) labels.current.set(node.data.key, element)
                    else labels.current.delete(node.data.key)
                  }}
                  style={{ display: 'none' }}
                />
              ))}
            <circle
              className="onto-sun__core"
              r={innerRadius - 2}
              onClick={(event) => {
                event.stopPropagation()
                onFocusChange(focus.parent?.data.key ?? ROOT_KEY)
              }}
            />
            <text className="onto-sun__title" y={-6}>
              {trim(focus.data.label, Math.floor((innerRadius * 1.7) / 6.4))}
            </text>
            <text className="onto-sun__value" y={16}>
              {focusMagnitude === null ? 'mean |z| n/a' : `mean |z| ${focusMagnitude.toFixed(2)}`}
            </text>
          </g>
        </svg>
        {hover && hovered && (
          <ChartTooltip x={hover.x} y={hover.y} width={width}>
            <span className="onto-tip__path">{hovered.node.path.map(prettyLabel).join(' / ')}</span>
            <span className="onto-tip__row">
              <strong>{prettyLabel(hovered.node.label)}</strong>
            </span>
            <span className="onto-tip__row t-tiny">
              signed {signed(hovered.signedMean)} - mean |z|{' '}
              {hovered.node.mean_abs_score === null ? '-' : hovered.node.mean_abs_score.toFixed(2)} -{' '}
              {number(hovered.node.leaf_count)} leaves
            </span>
          </ChartTooltip>
        )}
      </div>
    </div>
  )
}

function trim(text: string, room: number): string {
  if (room <= 1) return ''
  return text.length > room ? `${text.slice(0, Math.max(1, room - 1))}…` : text
}

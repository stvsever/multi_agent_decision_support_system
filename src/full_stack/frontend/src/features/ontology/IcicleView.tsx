/**
 * Zoomable icicle, horizontal.
 *
 * Same partition and same animation technique as the sunburst: depth runs left
 * to right, siblings stack, and only the columns inside the focus window are
 * touched during a transition. This is the view that stays readable when the
 * taxonomy is both deep and wide.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { hierarchy, partition, type HierarchyRectangularNode } from 'd3-hierarchy'
import { number, signed } from '@/lib/format'
import { Breadcrumb, ChartTooltip } from './atoms'
import { ROOT_KEY, bandColor, prettyLabel, type HierarchyDatum } from './model'
import { useAnimatedView, useMeasure, type View } from './useOntologyIndex'
import type { PartitionViewProps } from './SunburstView'

/** Depth columns visible from the focus. */
const COLS = 6
const HEIGHT = 520

type Node = HierarchyRectangularNode<HierarchyDatum>

export function IcicleView({
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
  const width = Math.max(320, size.width || 720)
  const columnWidth = width / COLS

  const layout = useMemo(() => {
    const root = hierarchy<HierarchyDatum>(data, (d) => d.kids).sum((d) =>
      d.kids.length ? 0 : Math.max(1, d.leaves),
    )
    return partition<HierarchyDatum>().size([1, root.height + 1])(root)
  }, [data])

  const nodes = useMemo(() => layout.descendants(), [layout])
  const byKey = useMemo(() => {
    const map = new Map<string, Node>()
    for (const node of nodes) map.set(node.data.key, node)
    return map
  }, [nodes])

  const focus = byKey.get(focusKey) ?? layout
  const target = useMemo<View>(() => [focus.x0, focus.x1, focus.y0], [focus])

  const rects = useRef(new Map<string, SVGRectElement>())
  const labels = useRef(new Map<string, SVGTextElement>())
  const shown = useRef(new Map<string, boolean>())

  const apply = useCallback(
    (view: View) => {
      const [vx0, vx1, vy0] = view
      const scale = HEIGHT / Math.max(vx1 - vx0, 1e-9)
      for (const node of nodes) {
        const key = node.data.key
        const rect = rects.current.get(key)
        if (!rect) continue
        const top = (node.x0 - vx0) * scale
        const bottom = (node.x1 - vx0) * scale
        const column = node.y0 - vy0
        const label = labels.current.get(key)
        const visible =
          column >= -1e-6 &&
          column < COLS - 1e-6 &&
          bottom > 0 &&
          top < HEIGHT &&
          bottom - top > 0.7

        if (!visible) {
          if (shown.current.get(key) !== false) {
            rect.style.display = 'none'
            if (label) label.style.display = 'none'
            shown.current.set(key, false)
          }
          continue
        }

        const y = Math.max(0, top)
        const boxHeight = Math.max(0.7, Math.min(HEIGHT, bottom) - y - 1)
        rect.setAttribute('x', String(column * columnWidth))
        rect.setAttribute('y', String(y))
        rect.setAttribute('width', String(Math.max(1, columnWidth - 2)))
        rect.setAttribute('height', String(boxHeight))
        if (shown.current.get(key) !== true) {
          rect.style.display = ''
          shown.current.set(key, true)
        }

        if (!label) continue
        if (boxHeight < 13 || columnWidth < 46) {
          label.style.display = 'none'
          continue
        }
        label.setAttribute('x', String(column * columnWidth + 7))
        label.setAttribute('y', String(y + boxHeight / 2 + 3.5))
        label.style.display = ''
      }
    },
    [nodes, columnWidth],
  )

  useAnimatedView(target, apply, reducedMotion ? 0 : 560)

  const revealed = useRef<string | null>(null)
  useEffect(() => {
    if (!selected || revealed.current === selected) return
    revealed.current = selected
    const node = byKey.get(selected)
    if (!node) return
    const within =
      node.y0 >= focus.y0 - 1e-9 &&
      node.y0 < focus.y0 + COLS &&
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
  const room = Math.max(2, Math.floor((columnWidth - 16) / 6.1))

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
        style={{ height: HEIGHT }}
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
          if (!key) {
            onFocusChange(focus.parent?.data.key ?? ROOT_KEY)
            return
          }
          if (key !== ROOT_KEY) onSelect(key)
          const node = byKey.get(key)
          if (node?.children?.length) onFocusChange(key)
        }}
      >
        <svg width={width} height={HEIGHT} role="img" aria-label="Ontology icicle">
          {nodes.map((node) => {
            const dim = matchBranch && node.depth > 0 ? !matchBranch.has(node.data.key) : false
            const opacity = Math.max(0.45, 1 - node.depth * 0.07) * (dim ? 0.22 : 1)
            return (
              <rect
                key={node.data.key}
                data-key={node.data.key}
                data-selected={selected === node.data.key || undefined}
                className="onto-cell"
                rx={3}
                ref={(element) => {
                  if (element) rects.current.set(node.data.key, element)
                  else {
                    // Drop the cached visibility with the element, or a
                    // remount would inherit a stale display flag.
                    rects.current.delete(node.data.key)
                    shown.current.delete(node.data.key)
                  }
                }}
                style={{ fill: bandColor(node.data.band), fillOpacity: opacity, display: 'none' }}
              />
            )
          })}
          {nodes.map((node) => (
            <text
              key={node.data.key}
              data-key={node.data.key}
              className="onto-cell__label"
              ref={(element) => {
                if (element) labels.current.set(node.data.key, element)
                else labels.current.delete(node.data.key)
              }}
              style={{ display: 'none' }}
            >
              {node.data.label.length > room
                ? `${node.data.label.slice(0, Math.max(1, room - 1))}…`
                : node.data.label}
            </text>
          ))}
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

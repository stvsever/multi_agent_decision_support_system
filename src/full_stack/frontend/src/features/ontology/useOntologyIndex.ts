/**
 * Hooks the explorer needs that the shared library does not provide: the
 * memoised index, the derived filter sets, container measurement, row
 * virtualisation, and the animated view used by the two partition charts.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { Ontology } from '@/lib/types'
import {
  ancestorsOf,
  buildIndex,
  computeKeep,
  computeMatches,
  subtreeMatches,
  type Filters,
  type OntologyIndex,
} from './model'

export function useOntologyIndex(ontology: Ontology | undefined): OntologyIndex | null {
  return useMemo(() => (ontology ? buildIndex(ontology) : null), [ontology])
}

export interface FilterResult {
  /** Null means every node survives, which lets the views skip the lookup. */
  keep: Set<string> | null
  matches: Set<string> | null
  /** Matches plus every node with a match beneath it, for chart dimming. */
  matchBranch: Set<string> | null
  /** Ancestors of matches, merged into the tree's expanded set. */
  autoExpand: Set<string>
  keptCount: number
}

export function useOntologyFilter(index: OntologyIndex | null, filters: Filters): FilterResult {
  return useMemo(() => {
    if (!index) {
      return { keep: null, matches: null, matchBranch: null, autoExpand: new Set<string>(), keptCount: 0 }
    }
    const keep = computeKeep(index, filters)
    const matches = computeMatches(index, filters.search)
    const matchBranch = matches ? subtreeMatches(index, matches) : null
    const autoExpand = matches ? ancestorsOf(index, matches) : new Set<string>()
    return {
      keep,
      matches,
      matchBranch,
      autoExpand,
      keptCount: keep ? keep.size : index.order.length,
    }
  }, [index, filters])
}

/* --- Measurement ---------------------------------------------------------- */

export interface Size {
  width: number
  height: number
}

/** SVG views size themselves from the container rather than a fixed width. */
export function useMeasure<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })

  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    const apply = (width: number, height: number) => {
      setSize((prev) =>
        Math.abs(prev.width - width) < 0.5 && Math.abs(prev.height - height) < 0.5
          ? prev
          : { width, height },
      )
    }
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect
      if (box) apply(box.width, box.height)
    })
    observer.observe(element)
    apply(element.clientWidth, element.clientHeight)
    return () => observer.disconnect()
  }, [])

  return [ref, size] as const
}

/* --- Virtualisation ------------------------------------------------------- */

export interface VirtualWindow {
  start: number
  end: number
}

/**
 * Windowed rendering for the long lists.
 *
 * A fixed row height means the scroll position maps straight onto an index, so
 * the whole list is a spacer of `count * rowHeight` with a translated slice on
 * top. Scroll events are coalesced into one measurement per animation frame,
 * and the range only lands in state when it actually changes, which keeps a
 * fast scroll at one render per frame regardless of how many rows exist.
 */
export function useVirtualRows(count: number, rowHeight: number, overscan = 10) {
  const ref = useRef<HTMLDivElement>(null)
  const frame = useRef(0)
  const [range, setRange] = useState<VirtualWindow>({ start: 0, end: Math.min(count, 60) })

  const measure = useCallback(() => {
    const element = ref.current
    if (!element) return
    const viewport = element.clientHeight || rowHeight * 20
    const start = Math.max(0, Math.floor(element.scrollTop / rowHeight) - overscan)
    const end = Math.min(count, Math.ceil((element.scrollTop + viewport) / rowHeight) + overscan)
    setRange((prev) => (prev.start === start && prev.end === end ? prev : { start, end }))
  }, [count, rowHeight, overscan])

  useEffect(() => {
    measure()
  }, [measure])

  useEffect(() => {
    const element = ref.current
    if (!element) return
    const observer = new ResizeObserver(() => measure())
    observer.observe(element)
    return () => observer.disconnect()
  }, [measure])

  useEffect(() => () => cancelAnimationFrame(frame.current), [])

  const onScroll = useCallback(() => {
    if (frame.current) return
    frame.current = requestAnimationFrame(() => {
      frame.current = 0
      measure()
    })
  }, [measure])

  const scrollToIndex = useCallback(
    (index: number) => {
      const element = ref.current
      if (!element || index < 0) return
      const top = index * rowHeight
      if (top < element.scrollTop) element.scrollTop = top
      else if (top + rowHeight > element.scrollTop + element.clientHeight) {
        element.scrollTop = top + rowHeight - element.clientHeight
      }
      measure()
    },
    [rowHeight, measure],
  )

  return { ref, range, onScroll, scrollToIndex, totalHeight: count * rowHeight }
}

/* --- Animated partition view ---------------------------------------------- */

/** Focus rectangle in partition space: [x0, x1, y0]. */
export type View = readonly [number, number, number]

const easeInOut = (t: number): number => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2)
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t

/**
 * Drives a zoom transition outside React.
 *
 * Re-rendering a thousand arcs per frame would drop frames, so the target view
 * is interpolated in an animation frame loop and `apply` writes the geometry
 * straight onto the already mounted elements. React still owns which elements
 * exist and what colour they are.
 */
export function useAnimatedView(target: View, apply: (view: View) => void, duration: number) {
  const current = useRef<View | null>(null)
  const previousTarget = useRef<View | null>(null)

  useEffect(() => {
    const from = current.current
    const settled = previousTarget.current
    const unchanged =
      settled && settled[0] === target[0] && settled[1] === target[1] && settled[2] === target[2]
    previousTarget.current = target

    // A hidden document never fires animation frames, so anything queued there
    // would leave the geometry stale. Those cases jump straight to the target.
    if (!from || unchanged || duration <= 0 || document.hidden) {
      current.current = target
      apply(target)
      return
    }

    let raf = 0
    const started = performance.now()
    const step = (now: number) => {
      const t = Math.min(1, (now - started) / duration)
      const eased = easeInOut(t)
      const view: View = [
        lerp(from[0], target[0], eased),
        lerp(from[1], target[1], eased),
        lerp(from[2], target[2], eased),
      ]
      current.current = view
      apply(view)
      if (t < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, apply, duration])
}

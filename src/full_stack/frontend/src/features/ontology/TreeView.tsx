/**
 * Virtualised indented tree.
 *
 * The visible rows are a flat projection of the expanded state, so scrolling a
 * thousand leaves costs one slice and one translate rather than a thousand
 * elements.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { ChevronRight, ListTree } from 'lucide-react'
import { Button, EmptyState, Select } from '@/components/ui/primitives'
import { compactNumber } from '@/lib/format'
import { BandDot, DeviationBar, Highlight, ScoreChip } from './atoms'
import { bandColor, prettyLabel, type FlatNode, type OntologyIndex } from './model'
import { useVirtualRows, type FilterResult } from './useOntologyIndex'

interface Row {
  key: string
  flat: FlatNode
  expandable: boolean
  open: boolean
  dim: boolean
  setSize: number
  posInSet: number
}

export interface TreeViewProps {
  index: OntologyIndex
  filter: FilterResult
  query: string
  expanded: Set<string>
  onExpandedChange: (next: Set<string>) => void
  selected: string | null
  onSelect: (key: string) => void
  rowHeight: number
}

export function TreeView({
  index,
  filter,
  query,
  expanded,
  onExpandedChange,
  selected,
  onSelect,
  rowHeight,
}: TreeViewProps) {
  const { keep, matchBranch } = filter

  // Branches the reader closed again while a search was open. Without this the
  // auto-expansion would immediately undo every manual collapse.
  const [suppressed, setSuppressed] = useState<Set<string>>(() => new Set())
  useEffect(() => setSuppressed(new Set()), [query])

  // Search never removes a node, it only reveals the ones that matched, so the
  // surrounding structure stays readable while the hits are highlighted.
  const effectiveExpanded = useMemo(() => {
    if (filter.autoExpand.size === 0) return expanded
    const merged = new Set(expanded)
    for (const key of filter.autoExpand) if (!suppressed.has(key)) merged.add(key)
    return merged
  }, [expanded, filter.autoExpand, suppressed])

  const rows = useMemo(() => {
    const out: Row[] = []
    const visit = (keys: string[]) => {
      const siblings = keep ? keys.filter((key) => keep.has(key)) : keys
      siblings.forEach((key, position) => {
        const flat = index.byKey.get(key)
        if (!flat) return
        const childKeys = keep ? flat.children.filter((child) => keep.has(child)) : flat.children
        const open = childKeys.length > 0 && effectiveExpanded.has(key)
        out.push({
          key,
          flat,
          expandable: childKeys.length > 0,
          open,
          dim: matchBranch ? !matchBranch.has(key) : false,
          setSize: siblings.length,
          posInSet: position + 1,
        })
        if (open) visit(childKeys)
      })
    }
    visit(index.roots)
    return out
  }, [index, keep, effectiveExpanded, matchBranch])

  const rowIndex = useMemo(() => {
    const map = new Map<string, number>()
    rows.forEach((row, at) => map.set(row.key, at))
    return map
  }, [rows])

  const { ref, range, onScroll, scrollToIndex, totalHeight } = useVirtualRows(rows.length, rowHeight)
  const [focusKey, setFocusKey] = useState<string | null>(null)
  const focusRef = useRef<HTMLDivElement>(null)
  const pendingFocus = useRef(0)

  const requestFocus = useCallback((key: string) => {
    pendingFocus.current = 3
    setFocusKey(key)
  }, [])

  // Roving tabindex over a windowed list: move the focus target into range
  // first, then hand DOM focus to the row once it has actually rendered.
  useLayoutEffect(() => {
    if (pendingFocus.current <= 0 || !focusKey) return
    const at = rowIndex.get(focusKey)
    if (at === undefined) {
      pendingFocus.current = 0
      return
    }
    if (at < range.start || at >= range.end) {
      pendingFocus.current -= 1
      scrollToIndex(at)
      return
    }
    pendingFocus.current = 0
    focusRef.current?.focus({ preventScroll: true })
  })

  const activeKey = focusKey && rowIndex.has(focusKey) ? focusKey : (rows[0]?.key ?? null)

  // A selection made in another view or from the extremes strip should land in
  // view here too, once the ancestors it needed have been expanded.
  const revealed = useRef<string | null>(null)
  useEffect(() => {
    if (!selected || revealed.current === selected) return
    const at = rowIndex.get(selected)
    if (at === undefined) return
    revealed.current = selected
    scrollToIndex(at)
  }, [selected, rowIndex, scrollToIndex])

  const toggle = useCallback(
    (key: string, open: boolean) => {
      setSuppressed((previous) => {
        if (open === !previous.has(key)) return previous
        const next = new Set(previous)
        if (open) next.delete(key)
        else next.add(key)
        return next
      })
      const next = new Set(effectiveExpanded)
      if (open) next.add(key)
      else next.delete(key)
      onExpandedChange(next)
    },
    [effectiveExpanded, onExpandedChange],
  )

  const moveTo = useCallback(
    (at: number) => {
      const clamped = Math.max(0, Math.min(rows.length - 1, at))
      const row = rows[clamped]
      if (row) requestFocus(row.key)
    },
    [rows, requestFocus],
  )

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (rows.length === 0) return
      const at = activeKey ? (rowIndex.get(activeKey) ?? 0) : 0
      const row = rows[at]
      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault()
          moveTo(at + 1)
          break
        case 'ArrowUp':
          event.preventDefault()
          moveTo(at - 1)
          break
        case 'Home':
          event.preventDefault()
          moveTo(0)
          break
        case 'End':
          event.preventDefault()
          moveTo(rows.length - 1)
          break
        case 'PageDown':
          event.preventDefault()
          moveTo(at + 12)
          break
        case 'PageUp':
          event.preventDefault()
          moveTo(at - 12)
          break
        case 'ArrowRight':
          event.preventDefault()
          if (!row) break
          if (row.expandable && !row.open) toggle(row.key, true)
          else if (row.open) moveTo(at + 1)
          break
        case 'ArrowLeft': {
          event.preventDefault()
          if (!row) break
          if (row.open) toggle(row.key, false)
          else if (row.flat.parent && rowIndex.has(row.flat.parent)) requestFocus(row.flat.parent)
          break
        }
        case 'Enter':
        case ' ':
          event.preventDefault()
          if (row) onSelect(row.key)
          break
        default:
          break
      }
    },
    [rows, rowIndex, activeKey, moveTo, toggle, onSelect, requestFocus],
  )

  const expandAll = () => {
    const next = new Set<string>()
    for (const key of index.order) {
      const flat = index.byKey.get(key)
      if (flat && flat.children.length > 0) next.add(key)
    }
    setSuppressed(new Set())
    onExpandedChange(next)
  }

  const collapseAll = () => {
    setSuppressed(new Set(filter.autoExpand))
    onExpandedChange(new Set())
  }

  const expandToDepth = (depth: number) => {
    const next = new Set<string>()
    for (const key of index.order) {
      const flat = index.byKey.get(key)
      if (flat && flat.children.length > 0 && flat.depth < depth) next.add(key)
    }
    setSuppressed(new Set())
    onExpandedChange(next)
  }

  const slice = rows.slice(range.start, range.end)

  return (
    <div className="onto-tree">
      <div className="onto-tree__bar">
        <span className="t-tiny muted">
          {compactNumber(rows.length)} {rows.length === 1 ? 'row' : 'rows'} shown
        </span>
        <span className="grow" />
        <Button size="sm" variant="ghost" onClick={expandAll}>
          Expand all
        </Button>
        <Button size="sm" variant="ghost" onClick={collapseAll}>
          Collapse all
        </Button>
        <Select
          aria-label="Expand to depth"
          value=""
          onChange={(event) => {
            const depth = Number(event.target.value)
            if (depth > 0) expandToDepth(depth)
          }}
          style={{ width: 132, height: 'var(--control-h-sm)', fontSize: 'var(--t-tiny)' }}
        >
          <option value="">To depth...</option>
          {Array.from({ length: Math.max(1, index.maxDepth) }, (_, i) => i + 1).map((depth) => (
            <option key={depth} value={depth}>
              Depth {depth}
            </option>
          ))}
        </Select>
      </div>

      {rows.length === 0 ? (
        <div className="onto-tree__empty">
          <EmptyState
            icon={<ListTree size={20} />}
            title="Nothing matches these filters"
            body="Widen the deviation threshold or clear a band to bring nodes back."
          />
        </div>
      ) : (
        <div
          ref={ref}
          className="onto-tree__scroll"
          onScroll={onScroll}
          role="tree"
          aria-label="Ontology tree"
          aria-multiselectable={false}
          onKeyDown={onKeyDown}
        >
          <div style={{ height: totalHeight, position: 'relative' }} role="presentation">
            <div
              role="presentation"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                transform: `translateY(${range.start * rowHeight}px)`,
              }}
            >
              {slice.map((row) => (
                <TreeRow
                  key={row.key}
                  row={row}
                  clamp={index.clamp}
                  query={query}
                  height={rowHeight}
                  selected={selected === row.key}
                  focused={activeKey === row.key}
                  rowRef={activeKey === row.key ? focusRef : undefined}
                  onSelect={() => {
                    setFocusKey(row.key)
                    onSelect(row.key)
                  }}
                  onToggle={() => toggle(row.key, !row.open)}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function TreeRow({
  row,
  clamp,
  query,
  height,
  selected,
  focused,
  rowRef,
  onSelect,
  onToggle,
}: {
  row: Row
  clamp: number
  query: string
  height: number
  selected: boolean
  focused: boolean
  rowRef?: React.Ref<HTMLDivElement>
  onSelect: () => void
  onToggle: () => void
}) {
  const { flat } = row
  const node = flat.node
  const isLeaf = flat.children.length === 0
  const coverage = node.leaf_count > 0 ? node.present_leaves / node.leaf_count : null

  return (
    <div
      ref={rowRef}
      role="treeitem"
      tabIndex={focused ? 0 : -1}
      aria-level={flat.depth + 1}
      aria-posinset={row.posInSet}
      aria-setsize={row.setSize}
      aria-expanded={row.expandable ? row.open : undefined}
      aria-selected={selected}
      className="onto-row"
      data-selected={selected || undefined}
      data-dim={row.dim || undefined}
      style={{ height, paddingLeft: 6 + flat.depth * 15 }}
      onClick={onSelect}
      onDoubleClick={(event) => {
        event.preventDefault()
        if (row.expandable) onToggle()
      }}
    >
      <span
        className="onto-row__rail"
        aria-hidden="true"
        style={{ background: bandColor(node.band) }}
      />
      <span
        role="presentation"
        className="onto-row__twist"
        data-open={row.open || undefined}
        data-hidden={!row.expandable || undefined}
        onClick={(event) => {
          event.stopPropagation()
          if (row.expandable) onToggle()
        }}
      >
        <ChevronRight size={13} />
      </span>
      {isLeaf && <BandDot band={node.band} size={6} />}
      <span className="onto-row__label truncate" title={node.path.join(' / ')}>
        <Highlight text={prettyLabel(node.label)} query={query} />
      </span>
      {/* Features usually restate the leaves one for one, so the count only
          earns a place when it says something the leaf count does not. */}
      {!isLeaf && node.feature_count > 0 && node.feature_count !== node.leaf_count && (
        <span className="onto-row__meta t-micro tabular" title="Features in this subtree">
          {compactNumber(node.feature_count)}f
        </span>
      )}
      <span className="onto-row__score">
        <ScoreChip
          band={node.band}
          value={isLeaf ? node.score : node.mean_abs_score}
          aggregate={!isLeaf}
          title={
            isLeaf
              ? `Signed deviation ${node.score ?? 'not measured'}`
              : `Mean absolute deviation across ${node.leaf_count} leaves`
          }
        />
      </span>
      <span className="onto-row__bar">
        <DeviationBar score={isLeaf ? node.score : flat.signedMean} clamp={clamp} width={78} />
      </span>
      <span className="onto-row__num tabular" title="Leaves beneath this node">
        {compactNumber(node.leaf_count)}
      </span>
      <span className="onto-row__num tabular" title="Measured share of those leaves">
        {coverage === null ? '-' : `${Math.round(coverage * 100)}%`}
      </span>
    </div>
  )
}

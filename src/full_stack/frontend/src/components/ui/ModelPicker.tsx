/**
 * Shared model chooser.
 *
 * The catalog is a live provider listing of several hundred rows, so the list
 * is searched locally, grouped by provider, and windowed to whatever is near
 * the viewport. The component owns no stylesheet: every colour resolves through
 * a token so it drops into any surface unchanged.
 */

import clsx from 'clsx'
import { useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, Check, ChevronsUpDown, RotateCw, Search, X } from 'lucide-react'
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type JSX,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react'
import { createPortal } from 'react-dom'
import { api } from '@/lib/api'
import { compactNumber, dateTime, usdPerMillion } from '@/lib/format'
import { queryKeys, useCatalog, useDebounced } from '@/lib/hooks'
import type { ModelRow } from '@/lib/types'
import { Badge, Button, Callout, Input, Spinner } from './primitives'

export interface ModelPickerProps {
  value: string
  onChange: (modelId: string) => void
  /** Filter the catalog to embedding models. */
  embedding?: boolean
  placeholder?: string
  compact?: boolean
  /** Offer an "Inherit default" row that emits an empty string. */
  allowInherit?: boolean
  inheritLabel?: string
}

const PAGE = 40
const POPOVER_WIDTH = 520

export function ModelPicker({
  value,
  onChange,
  embedding = false,
  placeholder = 'Choose a model',
  compact = false,
  allowInherit = false,
  inheritLabel = 'Inherit default',
}: ModelPickerProps): JSX.Element {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [visible, setVisible] = useState(PAGE)
  const [active, setActive] = useState(0)
  const [anchor, setAnchor] = useState<{ left: number; top: number; width: number; above: boolean }>({
    left: 0,
    top: 0,
    width: POPOVER_WIDTH,
    above: false,
  })
  const [refreshing, setRefreshing] = useState(false)

  const triggerRef = useRef<HTMLButtonElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const client = useQueryClient()
  const params = useMemo(() => ({ embedding, limit: 500 }), [embedding])
  const catalog = useCatalog(params)
  const rows = useMemo(() => catalog.data?.models ?? [], [catalog.data])
  const byId = useMemo(() => new Map(rows.map((row) => [row.id, row])), [rows])
  const selected = value ? byId.get(value) : undefined

  const needle = useDebounced(search.trim().toLowerCase(), 140)

  const matches = useMemo(() => {
    if (!needle) return rows
    return rows.filter(
      (row) =>
        row.id.toLowerCase().includes(needle) ||
        row.name.toLowerCase().includes(needle) ||
        row.provider.toLowerCase().includes(needle),
    )
  }, [rows, needle])

  /* The stored model can predate the catalog, or belong to another provider
     entirely, so it is always offered even when the fetch does not carry it. */
  const orphan = value && !byId.has(value) ? value : ''
  const typedId = search.trim()
  const custom =
    typedId && typedId !== value && !byId.has(typedId) && /[a-z0-9]\/[a-z0-9]|-|\./i.test(typedId) ? typedId : ''

  const options = useMemo(() => {
    const list: Option[] = []
    if (allowInherit) list.push({ kind: 'inherit', id: '' })
    if (orphan && (!needle || orphan.toLowerCase().includes(needle))) list.push({ kind: 'orphan', id: orphan })
    if (custom) list.push({ kind: 'custom', id: custom })
    for (const row of matches.slice(0, visible)) list.push({ kind: 'model', id: row.id, row })
    return list
  }, [allowInherit, orphan, custom, matches, visible, needle])

  const groups = useMemo(() => groupByProvider(options), [options])

  useEffect(() => {
    if (!open) return
    // Only re-seeded when the popover opens; typing moves the row explicitly.
    setActive(Math.max(0, options.findIndex((option) => option.id === value)))
  }, [open])

  useEffect(() => setVisible(PAGE), [needle, open])

  const place = () => {
    const rect = triggerRef.current?.getBoundingClientRect()
    if (!rect) return
    const width = Math.max(rect.width, Math.min(POPOVER_WIDTH, window.innerWidth - 24))
    const spaceBelow = window.innerHeight - rect.bottom
    const above = spaceBelow < 320 && rect.top > spaceBelow
    setAnchor({
      left: Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)),
      top: above ? rect.top - 6 : rect.bottom + 6,
      width,
      above,
    })
  }

  useLayoutEffect(() => {
    if (open) place()
  }, [open])

  useEffect(() => {
    if (!open) return
    const close = () => setOpen(false)
    const onPointer = (event: MouseEvent) => {
      const target = event.target as Node
      if (popoverRef.current?.contains(target) || triggerRef.current?.contains(target)) return
      setOpen(false)
    }
    window.addEventListener('mousedown', onPointer)
    window.addEventListener('resize', close)
    // Capture catches scrolling in any ancestor, which would otherwise leave
    // the popover floating away from its trigger.
    window.addEventListener('scroll', close, true)
    return () => {
      window.removeEventListener('mousedown', onPointer)
      window.removeEventListener('resize', close)
      window.removeEventListener('scroll', close, true)
    }
  }, [open])

  const close = () => {
    setOpen(false)
    setSearch('')
    triggerRef.current?.focus()
  }

  const commit = (option: Option) => {
    onChange(option.id)
    close()
  }

  const onKeyDown = (event: ReactKeyboardEvent) => {
    if (event.key === 'Escape') {
      // The dialog behind the popover listens for Escape as well, and it must
      // not close while the picker is only being dismissed.
      event.stopPropagation()
      close()
      return
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      const step = event.key === 'ArrowDown' ? 1 : -1
      setActive((index) => {
        const next = Math.max(0, Math.min(options.length - 1, index + step))
        listRef.current?.querySelector(`[data-index="${next}"]`)?.scrollIntoView({ block: 'nearest' })
        if (next > options.length - 6) setVisible((count) => Math.min(matches.length, count + PAGE))
        return next
      })
      return
    }
    if (event.key === 'Enter') {
      event.preventDefault()
      const option = options[active]
      if (option) commit(option)
    }
  }

  const refresh = async () => {
    setRefreshing(true)
    try {
      const fresh = await api.catalog.models({ ...params, refresh: true })
      client.setQueryData(queryKeys.catalog(params as Record<string, unknown>), fresh)
    } catch {
      /* the callout below already explains an unreachable catalog */
      await catalog.refetch()
    } finally {
      setRefreshing(false)
    }
  }

  const fetchedAt = catalog.data?.fetched_at ? new Date(catalog.data.fetched_at * 1000).toISOString() : null
  const unavailable = catalog.isError || Boolean(catalog.data?.error) || (catalog.data?.stale ?? false)
  const problem = catalog.isError
    ? (catalog.error as Error | undefined)?.message || 'The model catalog could not be reached.'
    : catalog.data?.error || (catalog.data?.stale ? 'Showing a cached copy of the catalog.' : '')

  const triggerLabel = value ? (selected?.name ?? value) : allowInherit ? inheritLabel : placeholder
  const triggerHint = value
    ? selected
      ? `${compactNumber(selected.context_length)} ctx  ${usdPerMillion(selected.prompt_usd_per_mtok)} in  ${usdPerMillion(selected.completion_usd_per_mtok)} out`
      : 'Not in the current catalog'
    : allowInherit
      ? 'Uses the default model'
      : ''

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="select"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((was) => !was)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s-2)',
          textAlign: 'left',
          cursor: 'pointer',
          height: compact ? 'var(--control-h-sm)' : 'var(--control-h)',
          paddingRight: 'var(--s-2)',
          backgroundImage: 'none',
        }}
      >
        <span className="stack grow" style={{ gap: 0, minWidth: 0 }}>
          <span
            className="truncate"
            style={{
              fontSize: compact ? 'var(--t-tiny)' : 'var(--t-small)',
              color: value ? 'var(--text)' : 'var(--text-muted)',
            }}
          >
            {triggerLabel}
          </span>
          {!compact && triggerHint && (
            <span className="t-micro muted truncate tabular">{triggerHint}</span>
          )}
        </span>
        <ChevronsUpDown size={13} style={{ flex: 'none', color: 'var(--text-faint)' }} />
      </button>

      {open &&
        createPortal(
          <div
            ref={popoverRef}
            role="listbox"
            aria-label="Model catalog"
            onKeyDown={onKeyDown}
            style={{
              position: 'fixed',
              left: anchor.left,
              width: anchor.width,
              ...(anchor.above ? { bottom: window.innerHeight - anchor.top } : { top: anchor.top }),
              zIndex: 780,
              background: 'var(--bg-elevated)',
              border: '1px solid var(--line)',
              borderRadius: 'var(--r-lg)',
              boxShadow: 'var(--shadow-lg)',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              maxHeight: 'min(60vh, 460px)',
            }}
          >
            <div
              className="row gap-2"
              style={{ padding: 'var(--s-2)', borderBottom: '1px solid var(--line-faint)' }}
            >
              <Search size={14} style={{ flex: 'none', color: 'var(--text-faint)', marginLeft: 4 }} />
              <Input
                autoFocus
                value={search}
                placeholder={embedding ? 'Search embedding models' : 'Search by model, provider, or id'}
                onChange={(event) => setSearch(event.target.value)}
                style={{ border: 'none', background: 'transparent', height: 'var(--control-h-sm)', paddingLeft: 0 }}
              />
              {search && (
                <Button
                  variant="ghost"
                  size="sm"
                  iconOnly
                  icon={<X size={13} />}
                  aria-label="Clear search"
                  onClick={() => setSearch('')}
                />
              )}
              <Button
                variant="ghost"
                size="sm"
                iconOnly
                icon={refreshing ? <Spinner size={13} /> : <RotateCw size={13} />}
                aria-label="Refresh the catalog"
                title="Refresh the catalog"
                onClick={refresh}
              />
            </div>

            {unavailable && (
              <div style={{ padding: 'var(--s-2) var(--s-3)' }}>
                <Callout tone="caution" icon={<AlertTriangle size={14} />}>
                  <span className="t-tiny">
                    {problem}
                    {fetchedAt ? ` Last fetched ${dateTime(fetchedAt)}.` : ' Nothing has been fetched yet.'} You can
                    still type an exact model id and select it.
                  </span>
                </Callout>
              </div>
            )}

            <div
              ref={listRef}
              style={{ overflowY: 'auto', padding: 'var(--s-1) 0' }}
              onScroll={(event) => {
                const el = event.currentTarget
                if (el.scrollTop + el.clientHeight > el.scrollHeight - 160) {
                  setVisible((count) => (count >= matches.length ? count : count + PAGE))
                }
              }}
            >
              {catalog.isLoading && (
                <div className="row gap-2 center" style={{ padding: 'var(--s-6)' }}>
                  <Spinner /> <span className="t-small muted">Loading the catalog</span>
                </div>
              )}

              {!catalog.isLoading && options.length === 0 && (
                <div className="stack gap-1 center" style={{ padding: 'var(--s-6)', textAlign: 'center' }}>
                  <span className="t-small semibold">No model matches that search</span>
                  <span className="t-tiny muted">
                    {embedding
                      ? 'This provider catalog lists no embedding models. Type the exact id of the one you use.'
                      : 'Try a shorter search, or type an exact model id.'}
                  </span>
                </div>
              )}

              {groups.map((group) => (
                <div key={group.label}>
                  <div
                    className="eyebrow"
                    style={{
                      padding: 'var(--s-2) var(--s-3) var(--s-1)',
                      position: 'sticky',
                      top: 0,
                      background: 'var(--bg-elevated)',
                      zIndex: 1,
                    }}
                  >
                    {group.label}
                  </div>
                  {group.options.map((option) => {
                    const index = options.indexOf(option)
                    return (
                      <Row
                        key={`${option.kind}:${option.id}`}
                        option={option}
                        index={index}
                        active={index === active}
                        chosen={option.id === value}
                        onHover={() => setActive(index)}
                        onPick={() => commit(option)}
                        inheritLabel={inheritLabel}
                      />
                    )
                  })}
                </div>
              ))}
            </div>

            <div
              className="row between gap-2"
              style={{
                padding: 'var(--s-2) var(--s-3)',
                borderTop: '1px solid var(--line-faint)',
                background: 'var(--bg-sunken)',
              }}
            >
              <span className="t-micro muted tabular">
                {matches.length} of {rows.length} models
                {catalog.data?.providers.length ? ` across ${catalog.data.providers.length} providers` : ''}
              </span>
              {fetchedAt && <span className="t-micro faint">Fetched {dateTime(fetchedAt)}</span>}
            </div>
          </div>,
          document.body,
        )}
    </>
  )
}

/* --- rows ----------------------------------------------------------------- */

type Option =
  | { kind: 'model'; id: string; row: ModelRow }
  | { kind: 'inherit'; id: '' }
  | { kind: 'orphan'; id: string }
  | { kind: 'custom'; id: string }

function groupByProvider(options: Option[]): { label: string; options: Option[] }[] {
  const groups: { label: string; options: Option[] }[] = []
  const index = new Map<string, { label: string; options: Option[] }>()
  for (const option of options) {
    const label =
      option.kind === 'model'
        ? option.row.provider
        : option.kind === 'inherit'
          ? 'Default'
          : option.kind === 'orphan'
            ? 'Currently selected'
            : 'Exact id'
    let group = index.get(label)
    if (!group) {
      group = { label, options: [] }
      index.set(label, group)
      groups.push(group)
    }
    group.options.push(option)
  }
  return groups
}

function Row({
  option,
  index,
  active,
  chosen,
  onHover,
  onPick,
  inheritLabel,
}: {
  option: Option
  index: number
  active: boolean
  chosen: boolean
  onHover: () => void
  onPick: () => void
  inheritLabel: string
}) {
  const base: CSSProperties = {
    display: 'flex',
    width: '100%',
    alignItems: 'center',
    gap: 'var(--s-3)',
    padding: 'var(--s-2) var(--s-3)',
    border: 'none',
    background: active ? 'var(--bg-hover)' : 'transparent',
    cursor: 'pointer',
    textAlign: 'left',
  }

  if (option.kind !== 'model') {
    const title =
      option.kind === 'inherit' ? inheritLabel : option.kind === 'orphan' ? option.id : `Use "${option.id}"`
    const note =
      option.kind === 'inherit'
        ? 'Falls back to the default model above'
        : option.kind === 'orphan'
          ? 'Saved earlier and not present in the current catalog'
          : 'Sends this id to the provider exactly as typed'
    return (
      <button type="button" data-index={index} style={base} onMouseEnter={onHover} onClick={onPick} role="option" aria-selected={chosen}>
        <span className="stack grow" style={{ gap: 1, minWidth: 0 }}>
          <span className="t-small semibold truncate">{title}</span>
          <span className="t-micro muted truncate">{note}</span>
        </span>
        {chosen && <Check size={14} style={{ flex: 'none', color: 'var(--accent)' }} />}
      </button>
    )
  }

  const row = option.row
  return (
    <button type="button" data-index={index} style={base} onMouseEnter={onHover} onClick={onPick} role="option" aria-selected={chosen}>
      <span className="stack grow" style={{ gap: 2, minWidth: 0 }}>
        <span className="row gap-2" style={{ minWidth: 0 }}>
          <span className="t-small semibold truncate">{row.name}</span>
          {row.is_free && <Badge tone="positive">free</Badge>}
          {row.supports_structured_output && <Badge tone="info">json</Badge>}
          {row.supports_tools && <Badge tone="neutral">tools</Badge>}
        </span>
        <span className="t-micro faint truncate mono">{row.id}</span>
      </span>
      <span className="stack" style={{ gap: 1, flex: 'none', alignItems: 'flex-end' }}>
        <span className="t-micro muted tabular">{compactNumber(row.context_length)} ctx</span>
        <span className="t-micro tabular" style={{ color: 'var(--text-secondary)' }}>
          {usdPerMillion(row.prompt_usd_per_mtok)} in / {usdPerMillion(row.completion_usd_per_mtok)} out
        </span>
      </span>
      <Check
        size={14}
        className={clsx(!chosen && 'sr-only')}
        style={{ flex: 'none', color: 'var(--accent)' }}
      />
    </button>
  )
}

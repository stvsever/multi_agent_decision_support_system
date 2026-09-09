/**
 * Ontology explorer.
 *
 * One participant, one index, four ways to read it. Selection and filters live
 * here so switching view never loses the reader's place.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type JSX } from 'react'
import { AlertTriangle, Network } from 'lucide-react'
import {
  Callout,
  Card,
  EmptyState,
  InfoDot,
  Segmented,
  Select,
  Skeleton,
} from '@/components/ui/primitives'
import { number, percent, tokens } from '@/lib/format'
import { useOntology, useParticipants, useDebounced } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { Stat } from './atoms'
import { FilterRail } from './FilterRail'
import { IcicleView } from './IcicleView'
import { DomainCoveragePanel, ExtremesStrip } from './Panels'
import { NodeInspector } from './NodeInspector'
import { SunburstView } from './SunburstView'
import { TableView } from './TableView'
import { TreeView } from './TreeView'
import { ROOT_KEY, buildHierarchy, emptyFilters, topExtremes, type Filters } from './model'
import { useOntologyFilter, useOntologyIndex } from './useOntologyIndex'
import './ontology.css'

type ViewName = 'tree' | 'sunburst' | 'icicle' | 'table'

const VIEWS: { value: ViewName; label: string; title: string }[] = [
  { value: 'tree', label: 'Tree', title: 'Indented taxonomy' },
  { value: 'sunburst', label: 'Sunburst', title: 'Radial partition, zoomable' },
  { value: 'icicle', label: 'Icicle', title: 'Rectangular partition, zoomable' },
  { value: 'table', label: 'Table', title: 'Every leaf and feature, sortable' },
]

const ONTOLOGY_EXPLAINER = `The ontology is this participant's evidence linguistified into an IS-A
taxonomy: every measured feature becomes a leaf, and each node above it carries the mean absolute
deviation of everything beneath. Reading down a branch narrows from a domain to a single
measurement; reading a node's colour tells you how far that whole branch sits from the reference.`

export function OntologyPage(): JSX.Element {
  const participants = useParticipants()
  const chosen = useApp((state) => state.selected)
  const density = useApp((state) => state.appearance.density)
  const reducedMotion = useApp((state) => state.appearance.reducedMotion)

  const valid = useMemo(
    () => (participants.data?.participants ?? []).filter((row) => row.valid),
    [participants.data],
  )

  const [directory, setDirectory] = useState('')
  useEffect(() => {
    if (directory && valid.some((row) => row.directory === directory)) return
    const preferred = chosen.find((row) => valid.some((v) => v.directory === row.directory))
    setDirectory(preferred?.directory ?? valid[0]?.directory ?? '')
  }, [valid, chosen, directory])

  const ontology = useOntology(directory || null)
  const index = useOntologyIndex(ontology.data)

  const [view, setView] = useState<ViewName>('tree')
  const [filters, setFilters] = useState<Filters>(emptyFilters)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const [selection, setSelection] = useState<{ key: string | null; feature: string | null }>({
    key: null,
    feature: null,
  })
  const [focusKey, setFocusKey] = useState<string>(ROOT_KEY)

  // A new participant resets the reading position but keeps the chosen view.
  // Keyed on the directory rather than the index so a background refetch does
  // not throw away the reader's expansions.
  const settledFor = useRef<string | null>(null)
  useEffect(() => {
    if (!index || settledFor.current === directory) return
    settledFor.current = directory
    setExpanded(new Set(index.roots))
    setSelection({ key: null, feature: null })
    setFocusKey(ROOT_KEY)
    setFilters(emptyFilters())
  }, [index, directory])

  const debouncedSearch = useDebounced(filters.search, 180)
  const effective = useMemo<Filters>(
    () => ({ ...filters, search: debouncedSearch }),
    [filters, debouncedSearch],
  )
  const filter = useOntologyFilter(index, effective)

  const hierarchyData = useMemo(
    () =>
      index
        ? buildHierarchy(index, filter.keep, ontology.data?.participant_id ?? 'Participant')
        : null,
    [index, filter.keep, ontology.data?.participant_id],
  )

  const extremes = useMemo(
    () => (index && ontology.data ? topExtremes(index, ontology.data.extremes, 12) : []),
    [index, ontology.data],
  )

  const select = useCallback(
    (key: string, feature: string | null = null) => {
      if (!index || key === ROOT_KEY || !index.byKey.has(key)) return
      setSelection({ key, feature })
      setExpanded((previous) => {
        const next = new Set(previous)
        let changed = false
        let cursor = index.byKey.get(key)?.parent ?? null
        while (cursor) {
          if (!next.has(cursor)) {
            next.add(cursor)
            changed = true
          }
          cursor = index.byKey.get(cursor)?.parent ?? null
        }
        return changed ? next : previous
      })
    },
    [index],
  )

  const rowHeight = density === 'compact' ? 26 : 30
  const summary = ontology.data?.summary

  const participantPicker = (
    <Select
      value={directory}
      aria-label="Participant"
      style={{ minWidth: 220 }}
      disabled={valid.length === 0}
      onChange={(event) => setDirectory(event.target.value)}
    >
      {valid.length === 0 && <option value="">No valid participant</option>}
      {valid.map((row) => (
        <option key={row.directory} value={row.directory}>
          {row.name || row.id}
        </option>
      ))}
    </Select>
  )

  return (
    <div className="page onto">
      <header className="onto-head">
        <div className="row between gap-4 wrap">
          <div className="stack gap-1" style={{ minWidth: 0 }}>
            <h1 className="t-h1">Ontology explorer</h1>
            <span className="row gap-2 t-small muted wrap">
              <span>The participant's evidence as a navigable IS-A taxonomy</span>
              <InfoDot label="What the ontology is">{ONTOLOGY_EXPLAINER}</InfoDot>
            </span>
          </div>
          <div className="row gap-2">{participantPicker}</div>
        </div>

        <div className="onto-summary">
          {ontology.isLoading || !summary ? (
            Array.from({ length: 6 }, (_, at) => (
              <div className="stack gap-2" key={at}>
                <Skeleton height={10} width={64} />
                <Skeleton height={20} width={54} />
              </div>
            ))
          ) : (
            <>
              <Stat label="Domains" value={number(summary.domain_count)} />
              <Stat label="Nodes" value={number(summary.node_count)} />
              <Stat
                label="Leaves"
                value={number(summary.leaf_count)}
                meta={`${number(summary.present_leaves)} measured`}
              />
              <Stat label="Max depth" value={number(summary.max_depth)} />
              <Stat
                label="Coverage"
                value={summary.coverage === null ? '-' : percent(summary.coverage, 0)}
              />
              <Stat
                label="Input tokens"
                value={summary.total_tokens === null ? '-' : tokens(summary.total_tokens)}
              />
            </>
          )}
        </div>
      </header>

      {ontology.isError && (
        <Callout
          tone="critical"
          icon={<AlertTriangle size={15} />}
          title="The ontology could not be loaded"
        >
          {(ontology.error as Error)?.message ?? 'The dashboard service returned an error.'}
        </Callout>
      )}

      {!ontology.isLoading && ontology.data && !ontology.data.has_deviation_map && (
        <Callout tone="caution" icon={<AlertTriangle size={15} />} title="No deviation map">
          This participant has no hierarchical_deviation_map.json, so there is no taxonomy to
          explore. Run the engine on this participant first.
        </Callout>
      )}

      {index && ontology.data && (
        <ExtremesStrip
          chips={extremes}
          clamp={index.clamp}
          selectedKey={selection.key}
          selectedFeature={selection.feature}
          onSelect={select}
        />
      )}

      <div className="onto-layout">
        <aside className="onto-rail onto-rail--left">
          {index && ontology.data ? (
            <>
              <FilterRail
                index={index}
                filters={filters}
                onChange={setFilters}
                matchCount={filter.keep ? filter.keptCount : null}
              />
              <DomainCoveragePanel
                coverage={ontology.data.domain_coverage}
                activeDomain={filters.domain}
                onSelectDomain={(domain) => setFilters((prev) => ({ ...prev, domain }))}
              />
            </>
          ) : (
            <Card title="Filter" className="onto-rail__card">
              <div className="stack gap-3">
                <Skeleton height={30} />
                <Skeleton height={140} />
                <Skeleton height={30} />
              </div>
            </Card>
          )}
        </aside>

        <main className="onto-main">
          <Card
            flush
            title="Taxonomy"
            subtitle={
              index
                ? `${number(index.order.length)} nodes, bars clamped at |z| ${index.clamp}`
                : undefined
            }
            actions={<Segmented value={view} options={VIEWS} onChange={setView} size="sm" />}
            className="onto-view"
          >
            {ontology.isLoading ? (
              <div className="stack gap-2" style={{ padding: 'var(--s-4)' }}>
                {Array.from({ length: 12 }, (_, at) => (
                  <Skeleton key={at} height={22} />
                ))}
              </div>
            ) : !index || index.roots.length === 0 || !hierarchyData ? (
              <EmptyState
                icon={<Network size={20} />}
                title="No ontology for this participant"
                body="Pick a participant whose inputs include a hierarchical deviation map."
              />
            ) : view === 'tree' ? (
              <TreeView
                index={index}
                filter={filter}
                query={effective.search}
                expanded={expanded}
                onExpandedChange={setExpanded}
                selected={selection.key}
                onSelect={select}
                rowHeight={rowHeight}
              />
            ) : view === 'sunburst' ? (
              <SunburstView
                index={index}
                data={hierarchyData}
                matchBranch={filter.matchBranch}
                selected={selection.key}
                focusKey={focusKey}
                onSelect={select}
                onFocusChange={setFocusKey}
                reducedMotion={reducedMotion}
              />
            ) : view === 'icicle' ? (
              <IcicleView
                index={index}
                data={hierarchyData}
                matchBranch={filter.matchBranch}
                selected={selection.key}
                focusKey={focusKey}
                onSelect={select}
                onFocusChange={setFocusKey}
                reducedMotion={reducedMotion}
              />
            ) : (
              <TableView
                index={index}
                filters={effective}
                selected={selection.key}
                selectedFeature={selection.feature}
                onSelect={select}
                rowHeight={rowHeight}
              />
            )}
          </Card>
        </main>

        <aside className="onto-rail onto-rail--right">
          {index ? (
            <NodeInspector
              index={index}
              nodeKey={selection.key}
              feature={selection.feature}
              onSelect={select}
            />
          ) : (
            <Card title="Inspector" className="onto-rail__card">
              <Skeleton height={180} />
            </Card>
          )}
        </aside>
      </div>
    </div>
  )
}

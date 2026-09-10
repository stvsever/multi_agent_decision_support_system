/**
 * Ontology explorer.
 *
 * One participant or a merged cohort, one index, five ways to read it.
 * Selection, filters and the graph's own state live here so switching view
 * never loses the reader's place.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type JSX } from 'react'
import { AlertTriangle, Network } from 'lucide-react'
import { Callout, Card, EmptyState, InfoDot, Skeleton, Tabs } from '@/components/ui/primitives'
import { number } from '@/lib/format'
import { useAggregateOntology, useDebounced, useOntology, useParticipants } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { CohortPicker } from './CohortPicker'
import { DistributionPanel } from './DistributionPanel'
import { FilterRail } from './FilterRail'
import { GraphView } from './GraphView'
import { IcicleView } from './IcicleView'
import { DomainCoveragePanel } from './Panels'
import { NodeInspector } from './NodeInspector'
import { SunburstView } from './SunburstView'
import { TableView } from './TableView'
import { TreeView } from './TreeView'
import { ROOT_KEY, buildHierarchy, emptyFilters, prettyLabel, type Filters } from './model'
import { initialCollapsed, type GraphLayoutName } from './graphLayout'
import { useOntologyFilter, useOntologyIndex } from './useOntologyIndex'
import './ontology.css'

type ViewName = 'graph' | 'tree' | 'sunburst' | 'icicle' | 'table'

const VIEWS: { value: ViewName; label: string }[] = [
  { value: 'graph', label: 'Graph' },
  { value: 'tree', label: 'Tree' },
  { value: 'sunburst', label: 'Sunburst' },
  { value: 'icicle', label: 'Icicle' },
  { value: 'table', label: 'Table' },
]

const ONTOLOGY_EXPLAINER = `Each measured feature is a leaf. Above it sit the broader things that
feature belongs to, up to a domain at the top. Every node carries the mean absolute deviation of
everything beneath it, so a node's colour says how far that whole branch sits from the reference,
and reading down a branch narrows from a domain to a single measurement. Select several
participants and the trees are merged: what they share is drawn once, and anything only one of them
carries stays visible and is marked as theirs.`

export function OntologyPage(): JSX.Element {
  const participants = useParticipants()
  const chosen = useApp((state) => state.selected)
  const density = useApp((state) => state.appearance.density)
  const reducedMotion = useApp((state) => state.appearance.reducedMotion)

  const valid = useMemo(
    () => (participants.data?.participants ?? []).filter((row) => row.valid),
    [participants.data],
  )

  const [dirs, setDirs] = useState<string[]>([])
  const [focus, setFocus] = useState('')

  // The page opens on whatever the app already had selected, and re-seeds only
  // when the current choice stops existing.
  useEffect(() => {
    if (dirs.length > 0 && dirs.every((dir) => valid.some((row) => row.directory === dir))) return
    const preferred = chosen
      .filter((row) => valid.some((v) => v.directory === row.directory))
      .map((row) => row.directory)
    const next = preferred.length > 0 ? preferred : valid.slice(0, 1).map((row) => row.directory)
    setDirs(next)
    setFocus(next[0] ?? '')
  }, [valid, chosen, dirs])

  const cohort = dirs.length > 1
  const merged = useAggregateOntology(cohort ? dirs : [])
  // A service without the merge endpoint still has to leave a usable page, so
  // the failure falls back to the marked participant's own tree.
  const mergeFailed = cohort && merged.isError
  const singleDir = !cohort || mergeFailed ? focus || dirs[0] || '' : ''
  const single = useOntology(singleDir || null)

  const active = cohort && !mergeFailed ? merged : single
  const data = active.data
  const index = useOntologyIndex(data)
  // `isLoading` goes false between retries, which would flash an empty state
  // over a request that is still coming. Pending covers the whole wait.
  const waiting = cohort && !mergeFailed ? merged.isPending : Boolean(singleDir) && single.isPending

  const [view, setView] = useState<ViewName>('graph')
  const [graphLayout, setGraphLayout] = useState<GraphLayoutName>('lr')
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())
  const [filters, setFilters] = useState<Filters>(emptyFilters)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const [selection, setSelection] = useState<{ key: string | null; feature: string | null }>({
    key: null,
    feature: null,
  })
  const [focusKey, setFocusKey] = useState<string>(ROOT_KEY)

  // A new selection resets the reading position but keeps the chosen view.
  // Keyed on the participants rather than the index so a background refetch
  // does not throw away the reader's expansions.
  const signature = `${dirs.join('|')}#${mergeFailed ? 'single' : 'merged'}`
  const settledFor = useRef<string | null>(null)
  useEffect(() => {
    if (!index || settledFor.current === signature) return
    settledFor.current = signature
    setExpanded(new Set(index.roots))
    setCollapsed(initialCollapsed(index))
    setSelection({ key: null, feature: null })
    setFocusKey(ROOT_KEY)
    setFilters(emptyFilters())
  }, [index, signature])

  const debouncedSearch = useDebounced(filters.search, 180)
  const effective = useMemo<Filters>(
    () => ({ ...filters, search: debouncedSearch }),
    [filters, debouncedSearch],
  )
  const filter = useOntologyFilter(index, effective)

  const rootLabel = cohort
    ? `${dirs.length} participants`
    : (data?.participant_id ?? 'Participant')

  const hierarchyData = useMemo(
    () => (index ? buildHierarchy(index, filter.keep, rootLabel) : null),
    [index, filter.keep, rootLabel],
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

  const clearSelection = useCallback(() => setSelection({ key: null, feature: null }), [])

  // Only a measurement has a distribution: a branch is an average of several.
  const measurement = useMemo(() => {
    if (!index || !selection.key) return null
    const flat = index.byKey.get(selection.key)
    if (!flat) return null
    if (selection.feature) {
      return {
        path: [...flat.node.path, selection.feature],
        label: prettyLabel(selection.feature),
      }
    }
    if (flat.children.length > 0) return null
    return { path: flat.node.path, label: prettyLabel(flat.node.label) }
  }, [index, selection])

  const focusId = useMemo(
    () => valid.find((row) => row.directory === focus)?.id ?? null,
    [valid, focus],
  )

  const rowHeight = density === 'compact' ? 26 : 30
  const lede = cohort
    ? `${dirs.length} participants merged into one hierarchy, drawn once where they overlap and marked where they do not.`
    : 'Every measurement sits under the broader thing it belongs to, and each level carries how far it deviates from normative expectation.'

  return (
    <div className="page onto" data-debug={JSON.stringify({cohort, mergeFailed, waiting, mStatus: merged.status, mFetch: merged.fetchStatus, mErr: String((merged.error as Error)?.message ?? ''), sStatus: single.status, singleDir: singleDir.slice(-8), focus: focus.slice(-8), dirs: dirs.length})}>
      <header className="onto-head">
        <div className="row between gap-4 wrap">
          <div className="stack gap-1" style={{ minWidth: 0 }}>
            <h1 className="t-h1">Ontology explorer</h1>
            <span className="row gap-2 t-small muted wrap">
              <span>{lede}</span>
              <InfoDot label="What this page shows">{ONTOLOGY_EXPLAINER}</InfoDot>
            </span>
          </div>
          <CohortPicker
            participants={valid}
            selected={dirs}
            onChange={setDirs}
            focus={focus}
            onFocusChange={setFocus}
            reducedMotion={reducedMotion}
          />
        </div>
      </header>

      {active.isError && (
        <Callout
          tone="critical"
          icon={<AlertTriangle size={15} />}
          title="The ontology could not be loaded"
        >
          {(active.error as Error)?.message ?? 'The dashboard service returned an error.'}
        </Callout>
      )}

      {mergeFailed && (
        <Callout
          tone="caution"
          icon={<AlertTriangle size={15} />}
          title="Merged trees are not available"
        >
          This dashboard service cannot merge participants yet, so the page is showing
          {` ${focusId ?? 'one participant'} `}
          on their own. Everything else still works.
        </Callout>
      )}

      {!waiting && data && !data.has_deviation_map && (
        <Callout tone="caution" icon={<AlertTriangle size={15} />} title="No deviation map">
          {cohort
            ? 'None of the selected participants has a hierarchical deviation map, so there is nothing to merge.'
            : 'This participant has no hierarchical_deviation_map.json, so there is no hierarchy to explore. Run the engine on this participant first.'}
        </Callout>
      )}

      <div className="onto-layout">
        <aside className="onto-rail onto-rail--left">
          {index && data ? (
            <>
              <FilterRail
                index={index}
                filters={filters}
                onChange={setFilters}
                matchCount={filter.keep ? filter.keptCount : null}
              />
              <DomainCoveragePanel
                coverage={data.domain_coverage}
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
            title={cohort ? 'Merged hierarchy' : 'Hierarchy'}
            subtitle={
              index
                ? `${number(index.order.length)} nodes, bars clamped at |z| ${index.clamp}`
                : undefined
            }
            className="onto-view"
          >
            <div className="onto-view__tabs">
              <Tabs value={view} options={VIEWS} onChange={setView} spread />
            </div>

            {waiting ? (
              <div className="stack gap-2" style={{ padding: 'var(--s-4)' }}>
                {Array.from({ length: 12 }, (_, at) => (
                  <Skeleton key={at} height={22} />
                ))}
              </div>
            ) : !index || index.roots.length === 0 || !hierarchyData ? (
              <EmptyState
                icon={<Network size={20} />}
                title="No hierarchy for this selection"
                body="Pick a participant whose inputs include a hierarchical deviation map."
              />
            ) : view === 'graph' ? (
              <div className="onto-graph-shell">
                <GraphView
                  index={index}
                  keep={filter.keep}
                  matchBranch={filter.matchBranch}
                  rootLabel={rootLabel}
                  cohort={cohort && !mergeFailed}
                  selected={selection.key}
                  onSelect={select}
                  onClear={clearSelection}
                  collapsed={collapsed}
                  onCollapsedChange={setCollapsed}
                  layout={graphLayout}
                  onLayoutChange={setGraphLayout}
                  density={density}
                  reducedMotion={reducedMotion}
                />
              </div>
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
            <>
              <NodeInspector
                index={index}
                nodeKey={selection.key}
                feature={selection.feature}
                onSelect={select}
              />
              {measurement && (
                <DistributionPanel
                  directories={dirs}
                  path={measurement.path}
                  label={measurement.label}
                  focusId={focusId}
                />
              )}
            </>
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

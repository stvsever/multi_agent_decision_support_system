/**
 * Reports: reading a finished run's deep phenotype output.
 *
 * Mounted at both /reports and /reports/:participant, and it also honours a
 * ?run_id= parameter so a link from the runs screen lands on the exact run
 * rather than on whatever the participant's directory happens to hold now.
 */

import clsx from 'clsx'
import { useMemo, useState, type JSX } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, FileText, FolderOpen, Printer, Sparkles } from 'lucide-react'
import {
  Badge,
  Button,
  Callout,
  CopyButton,
  EmptyState,
  Skeleton,
  Tabs,
  Tooltip,
} from '@/components/ui/primitives'
import { api } from '@/lib/api'
import { dateTime } from '@/lib/format'
import { queryKeys, useParticipants, useRuns } from '@/lib/hooks'
import type { ReportBundle } from '@/lib/types'
import { DocumentTab } from './DocumentTab'
import { ExecutionTab } from './ExecutionTab'
import { FilesTab } from './FilesTab'
import { SummaryTab } from './SummaryTab'
import { SourceRail, participantSourceOf, runSourceOf, type ReportSource } from './SourceRail'
import { buildView } from './model'
import './reports.css'

type TabKey = 'summary' | 'phenotype' | 'clinical' | 'execution' | 'files'

const TABS: { value: TabKey; label: string }[] = [
  { value: 'summary', label: 'Summary' },
  { value: 'phenotype', label: 'Deep phenotype' },
  { value: 'clinical', label: 'Clinical summary' },
  { value: 'execution', label: 'Execution' },
  { value: 'files', label: 'Files' },
]

const FINISHED = new Set(['succeeded', 'failed'])

/** Tabs that read as prose keep a document measure; the tabular ones do not. */
const DOCUMENT_TABS = new Set<TabKey>(['summary', 'phenotype', 'clinical'])

export function ReportsPage(): JSX.Element {
  const { participant } = useParams<{ participant?: string }>()
  const [search] = useSearchParams()
  const navigate = useNavigate()
  const runsQuery = useRuns()
  const participantsQuery = useParticipants()

  const [tab, setTab] = useState<TabKey>('summary')
  const [filter, setFilter] = useState('')

  const runIdParam = search.get('run_id')

  const runs = useMemo(() => {
    const rows = (runsQuery.data?.runs ?? []).filter((run) => FINISHED.has(run.status))
    return [...rows].sort((a, b) => {
      if (a.status !== b.status) return a.status === 'succeeded' ? -1 : 1
      const left = Date.parse(a.finished_at ?? a.created_at ?? '')
      const right = Date.parse(b.finished_at ?? b.created_at ?? '')
      return (Number.isNaN(right) ? 0 : right) - (Number.isNaN(left) ? 0 : left)
    })
  }, [runsQuery.data])

  const participants = useMemo(
    () => participantsQuery.data?.participants ?? [],
    [participantsQuery.data],
  )

  const selected: ReportSource | null = useMemo(() => {
    if (runIdParam) {
      const run = runs.find((entry) => entry.id === runIdParam)
      if (run) return runSourceOf(run)
      // The run list may still be loading, or the run may have been pruned.
      // Its id alone is enough for the service to resolve the bundle.
      return {
        key: `run:${runIdParam}`,
        kind: 'run',
        runId: runIdParam,
        participantId: participant ?? 'Unknown participant',
        participantDir: '',
        label: participant ?? runIdParam,
      }
    }
    if (participant) {
      const run = runs.find((entry) => entry.participant_id === participant)
      if (run) return runSourceOf(run)
      const match = participants.find((entry) => entry.id === participant)
      if (match) return participantSourceOf(match)
      return null
    }
    if (runs.length) return runSourceOf(runs[0])
    if (participants.length) return participantSourceOf(participants[0])
    return null
  }, [runIdParam, participant, runs, participants])

  const bundleParams = useMemo(() => {
    if (!selected) return null
    if (selected.kind === 'run' && selected.runId) return { run_id: selected.runId }
    if (selected.participantDir) return { participant_dir: selected.participantDir }
    return null
  }, [selected])

  const bundleQuery = useQuery<ReportBundle>({
    queryKey: queryKeys.report(bundleParams ?? {}),
    queryFn: () => api.reports.bundle(bundleParams ?? {}),
    enabled: Boolean(bundleParams),
    staleTime: 30_000,
    retry: false,
  })

  const bundle = bundleQuery.data
  const view = useMemo(() => (bundle ? buildView(bundle) : null), [bundle])

  const openSource = (source: ReportSource) => {
    const path = `/reports/${encodeURIComponent(source.participantId)}`
    navigate(source.kind === 'run' && source.runId ? `${path}?run_id=${encodeURIComponent(source.runId)}` : path)
  }

  // The print rules are written for the summary, so printing shows it first.
  const printSummary = () => {
    setTab('summary')
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => window.print()))
  }

  const deepMarkdown = bundle?.deep_phenotype_markdown ?? ''
  const sourcesLoading = runsQuery.isLoading || participantsQuery.isLoading
  const noSources = !sourcesLoading && runs.length === 0 && participants.length === 0

  return (
    <div className="page page--flush reports">
      <SourceRail
        runs={runs}
        participants={participants}
        selectedKey={selected?.key ?? null}
        onSelect={openSource}
        query={filter}
        onQuery={setFilter}
        loading={runsQuery.isLoading || participantsQuery.isLoading}
      />

      <div className="reports__main">
        <header className="reports__toolbar">
          <div className="stack gap-1" style={{ minWidth: 0 }}>
            <div className="row gap-2 wrap">
              <span className="t-h2">{view?.participantId ?? selected?.participantId ?? 'Reports'}</span>
              {selected?.kind === 'run' && selected.runId && (
                <Badge tone="accent" mono>
                  run {selected.runId.split('_').pop()}
                </Badge>
              )}
              {bundle?.run?.status && bundle.run.status !== 'succeeded' && (
                <Badge tone="critical">{bundle.run.status}</Badge>
              )}
              {view?.timestamp ? <span className="t-tiny muted">{dateTime(view.timestamp)}</span> : null}
            </div>
            {bundle?.output_dir ? (
              <div className="row gap-2" style={{ minWidth: 0 }}>
                <FolderOpen size={13} className="muted" style={{ flex: 'none' }} />
                <span className="reports__path truncate" title={bundle.output_dir}>
                  {bundle.output_dir}
                </span>
                <CopyButton text={bundle.output_dir} label="Copy path" />
              </div>
            ) : null}
          </div>

          <div className="row gap-2 wrap reports__no-print" style={{ flex: 'none' }}>
            {deepMarkdown ? (
              <CopyButton text={deepMarkdown} label="Copy deep phenotype" />
            ) : (
              <Button size="sm" variant="ghost" disabled>
                Copy deep phenotype
              </Button>
            )}
            <Tooltip content="Sends the summary as it appears here to your printer, in black and white.">
              <Button size="sm" variant="secondary" icon={<Printer size={13} />} onClick={printSummary}>
                Print summary
              </Button>
            </Tooltip>
            {bundleParams ? (
              <Tooltip content="A short PDF the service composes from this run: what was predicted, the evidence for it, and what the run could not see. It is not a copy of this screen.">
                <a
                  className="btn btn--primary btn--sm reports__pdf"
                  href={api.reports.pdfUrl(bundleParams)}
                  download
                  target="_blank"
                  rel="noreferrer"
                >
                  <FileText size={13} />
                  Download PDF
                </a>
              </Tooltip>
            ) : null}
          </div>
        </header>

        <div className="reports__tabs reports__no-print">
          <Tabs value={tab} options={TABS} onChange={setTab} spread />
        </div>

        <div className={clsx('reports__body', DOCUMENT_TABS.has(tab) && 'reports__body--doc')}>
          {noSources && (
            <EmptyState
              title="Nothing to read yet"
              body="No finished runs and no participants are visible to the service. Add a data root or start a run, and the output will appear here."
            />
          )}

          {!noSources && !selected && sourcesLoading && (
            <div className="stack gap-3">
              <Skeleton height={72} />
              <Skeleton height={128} />
            </div>
          )}

          {!noSources && !selected && !sourcesLoading && (
            <EmptyState
              title={participant ? `Nothing here for ${participant}` : 'Pick a source'}
              body={
                participant
                  ? 'No run and no participant with that id is visible to the service. Pick another source on the left.'
                  : 'Choose a finished run or a participant on the left to load its report bundle.'
              }
            />
          )}

          {selected && bundleQuery.isLoading && (
            <div className="stack gap-3">
              <Skeleton height={72} />
              <Skeleton height={128} />
              <Skeleton height={220} />
            </div>
          )}

          {selected && bundleQuery.isError && (
            <Callout tone="critical" icon={<AlertTriangle size={15} />} title="The report could not be loaded">
              {bundleQuery.error instanceof Error
                ? bundleQuery.error.message
                : 'The dashboard service refused the request.'}
            </Callout>
          )}

          {bundle && view && !bundleQuery.isLoading && (
            <>
              {!bundle.exists && (
                <Callout tone="caution" icon={<AlertTriangle size={15} />} title="No output directory">
                  Nothing has been written to {bundle.output_dir || 'the expected output directory'} yet. Run the
                  pipeline for this participant to produce a report.
                </Callout>
              )}

              {bundle.exists && !view.hasPerformance && (
                <Callout tone="caution" icon={<AlertTriangle size={15} />} title="No performance report">
                  The output directory exists, but the performance report was not written. The run either failed before
                  the predictor finished or was cancelled.
                </Callout>
              )}

              {tab === 'summary' &&
                (view.hasPerformance || Object.keys(bundle.patient_report ?? {}).length > 0 ? (
                  <SummaryTab view={view} cost={bundle.cost} />
                ) : (
                  <EmptyState
                    title="No summary to compose"
                    body="Neither the performance report nor the structured report is present in this output directory."
                  />
                ))}

              {tab === 'phenotype' && (
                <DocumentTab
                  source={deepMarkdown}
                  emptyIcon={<Sparkles size={20} />}
                  emptyTitle="No deep phenotype report"
                  emptyBody="This run did not generate one. The Communicator agent writes the deep phenotype narrative when a run requests it, so enable that option and run again to produce this document."
                />
              )}

              {tab === 'clinical' && (
                <DocumentTab
                  source={bundle.report_markdown ?? ''}
                  emptyIcon={<FileText size={20} />}
                  emptyTitle="No clinical summary"
                  emptyBody="The markdown patient report was not written for this run. The structured report may still hold the same content under the Files tab."
                />
              )}

              {tab === 'execution' && <ExecutionTab view={view} cost={bundle.cost} />}

              {tab === 'files' && bundleParams && <FilesTab artifacts={bundle.artifacts ?? []} params={bundleParams} />}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

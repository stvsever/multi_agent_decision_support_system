/**
 * Picking what to read.
 *
 * Finished runs come first because they carry the verdict and the spend.
 * Participants follow: a bundle may exist on disk for any of them, and asking
 * the service one directory at a time would be far more expensive than letting
 * the reader choose.
 */

import { Search } from 'lucide-react'
import { Badge, Input, Spinner } from '@/components/ui/primitives'
import { relativeTime, usd } from '@/lib/format'
import type { Participant, RunSummary } from '@/lib/types'

export interface ReportSource {
  key: string
  kind: 'run' | 'participant'
  runId?: string
  participantId: string
  participantDir: string
  label: string
}

export function runSourceOf(run: RunSummary): ReportSource {
  return {
    key: `run:${run.id}`,
    kind: 'run',
    runId: run.id,
    participantId: run.participant_id,
    participantDir: run.participant_dir,
    label: run.label || run.participant_id,
  }
}

export function participantSourceOf(participant: Participant): ReportSource {
  return {
    key: `participant:${participant.directory}`,
    kind: 'participant',
    participantId: participant.id,
    participantDir: participant.directory,
    label: participant.name || participant.id,
  }
}

export function SourceRail({
  runs,
  participants,
  selectedKey,
  onSelect,
  query,
  onQuery,
  loading,
}: {
  runs: RunSummary[]
  participants: Participant[]
  selectedKey: string | null
  onSelect: (source: ReportSource) => void
  query: string
  onQuery: (next: string) => void
  loading: boolean
}) {
  const needle = query.trim().toLowerCase()
  const matches = (text: string) => !needle || text.toLowerCase().includes(needle)

  const runRows = runs.filter((run) => matches(`${run.participant_id} ${run.label} ${run.verdict ?? ''}`))
  const participantRows = participants.filter((entry) => matches(`${entry.id} ${entry.name}`))

  return (
    <aside className="reports__rail reports__no-print">
      <div className="reports__rail-head">
        <span className="eyebrow">Source</span>
        <div className="row gap-2">
          <Search size={14} className="muted" />
          <Input
            value={query}
            placeholder="Filter runs and participants"
            aria-label="Filter sources"
            onChange={(event) => onQuery(event.target.value)}
          />
        </div>
      </div>

      <div className="reports__rail-list">
        {loading && (
          <div className="row gap-2 muted t-tiny" style={{ padding: 'var(--s-3)' }}>
            <Spinner size={13} /> Loading sources
          </div>
        )}

        {runRows.length > 0 && <span className="reports__group">Finished runs</span>}
        {runRows.map((run) => {
          const source = runSourceOf(run)
          return (
            <button
              key={source.key}
              type="button"
              className="reports__source"
              aria-current={selectedKey === source.key}
              onClick={() => onSelect(source)}
            >
              <span className="reports__source-title truncate">{run.participant_id}</span>
              <span className="reports__source-meta">
                {run.verdict ? (
                  <Badge tone={run.verdict === 'SATISFACTORY' ? 'positive' : 'caution'}>{run.verdict}</Badge>
                ) : null}
                {run.status !== 'succeeded' ? <Badge tone="critical">{run.status}</Badge> : null}
                <span>{relativeTime(run.finished_at ?? run.created_at)}</span>
                <span>{run.cost?.usd === null || run.cost?.usd === undefined ? 'unpriced' : usd(run.cost.usd)}</span>
              </span>
            </button>
          )
        })}

        {participantRows.length > 0 && <span className="reports__group">Participants</span>}
        {participantRows.map((participant) => {
          const source = participantSourceOf(participant)
          return (
            <button
              key={source.key}
              type="button"
              className="reports__source"
              aria-current={selectedKey === source.key}
              onClick={() => onSelect(source)}
            >
              <span className="reports__source-title truncate">{participant.id}</span>
              <span className="reports__source-meta">
                <span>{participant.domains.length} domains</span>
                {!participant.valid && <Badge tone="caution">input incomplete</Badge>}
              </span>
            </button>
          )
        })}

        {!loading && runRows.length === 0 && participantRows.length === 0 && (
          <span className="t-tiny muted" style={{ padding: 'var(--s-3)' }}>
            Nothing matches this filter.
          </span>
        )}
      </div>
    </aside>
  )
}

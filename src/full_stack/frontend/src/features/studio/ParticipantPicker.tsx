/**
 * Choosing whose data the run reads.
 *
 * Discovery is automatic and runs on mount, so most of this screen is about
 * making the scan legible: what was looked at, what qualified, and what came
 * close. A folder that yields nothing gets a sentence saying so rather than
 * silence that looks identical to success.
 */

import clsx from 'clsx'
import {
  AlertTriangle,
  Check,
  Database,
  FolderSearch,
  Plus,
  RefreshCw,
  Search,
  Trash2,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api, ApiError } from '@/lib/api'
import { compactNumber, relativeTime, tokens } from '@/lib/format'
import { queryKeys, useParticipants } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { Participant, ParticipantsResponse, ScannedRoot } from '@/lib/types'
import {
  Badge,
  Button,
  Callout,
  Disclosure,
  EmptyState,
  InfoDot,
  Input,
  Skeleton,
  Tooltip,
} from '@/components/ui/primitives'
import { FolderBrowser } from './FolderBrowser'
import { REQUIRED_FILES, describeMissing, summariseAdd, type AddOutcome } from './datasetScan'

export function ParticipantPicker({ multiple = false }: { multiple?: boolean }) {
  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } = useParticipants()
  const { selected, setSelected, toggleSelected, notify } = useApp()
  const [search, setSearch] = useState('')
  const [rootDraft, setRootDraft] = useState('')
  const [adding, setAdding] = useState(false)
  const [browsing, setBrowsing] = useState(false)
  const [outcome, setOutcome] = useState<AddOutcome | null>(null)
  const client = useQueryClient()

  const participants = useMemo(() => {
    const rows = data?.participants ?? []
    const needle = search.trim().toLowerCase()
    if (!needle) return rows
    return rows.filter((p) => p.id.toLowerCase().includes(needle) || p.directory.toLowerCase().includes(needle))
  }, [data, search])

  const valid = useMemo(() => participants.filter((p) => p.valid), [participants])
  const isSelected = (participant: Participant) => selected.some((s) => s.directory === participant.directory)

  const choose = (participant: Participant) => {
    if (!participant.valid) return
    if (multiple) toggleSelected(participant)
    else setSelected(isSelected(participant) && selected.length === 1 ? [] : [participant])
  }

  const addRoot = async (raw: string) => {
    const path = raw.trim()
    if (!path) return
    setAdding(true)
    try {
      const before = client.getQueryData<ParticipantsResponse>(queryKeys.participants) ?? data
      const after = await api.datasets.addRoot(path)
      client.setQueryData(queryKeys.participants, after)
      const result = summariseAdd(before, after, path)
      setOutcome(result)
      notify({ tone: result.tone, title: result.title, body: result.body })
      setRootDraft('')
    } catch (error) {
      const message = error instanceof ApiError ? error.message : String(error)
      setOutcome({ tone: 'critical', title: 'That folder could not be added', body: message, nearMisses: [] })
      notify({ tone: 'critical', title: 'That folder could not be added', body: message })
    } finally {
      setAdding(false)
    }
  }

  const removeRoot = async (root: ScannedRoot) => {
    try {
      const after = await api.datasets.removeRoot(root.root)
      client.setQueryData(queryKeys.participants, after)
      setOutcome(null)
      notify({ tone: 'info', title: 'Folder removed', body: `${root.label} is no longer scanned.` })
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'That folder could not be removed',
        body: error instanceof ApiError ? error.message : String(error),
      })
    }
  }

  return (
    <div className="stack gap-4">
      <div className="row gap-2 wrap">
        <div className="picker__search grow">
          <Search size={14} className="faint" />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Filter by id or path"
            aria-label="Filter participants"
          />
        </div>
        <Button
          size="sm"
          variant="ghost"
          icon={<RefreshCw size={13} className={isFetching ? 'spin' : undefined} />}
          onClick={() => refetch()}
        >
          {isFetching ? 'Scanning' : 'Rescan'}
        </Button>
        {multiple && valid.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setSelected(selected.length === valid.length ? [] : valid)}
          >
            {selected.length === valid.length ? 'Clear' : 'Select all'}
          </Button>
        )}
      </div>

      <span className="t-micro faint">
        {isFetching
          ? 'Scanning the folders below.'
          : dataUpdatedAt
            ? `Scanned ${relativeTime(new Date(dataUpdatedAt).toISOString())}. ${data?.count ?? 0} participant folder${(data?.count ?? 0) === 1 ? '' : 's'}.${isError ? ' The last rescan did not complete.' : ''}`
            : isError
              ? 'The last scan did not complete.'
              : 'Not scanned yet.'}
      </span>

      {isLoading ? (
        <div className="stack gap-2">
          {[0, 1, 2].map((index) => (
            <Skeleton key={index} height={54} radius={9} />
          ))}
        </div>
      ) : isError && !data ? (
        /* A failed scan is not an empty one. Saying "no participant folders
           found" here would blame the data for a connection problem. */
        <Callout
          tone="critical"
          icon={<AlertTriangle size={15} />}
          title="The folders could not be scanned"
          action={
            <Button size="sm" onClick={() => void refetch()} loading={isFetching}>
              Try again
            </Button>
          }
        >
          {error instanceof Error && error.message ? error.message : 'The service did not answer.'} Nothing has been
          lost: the folders below are still configured, and a retry rescans them.
        </Callout>
      ) : participants.length === 0 ? (
        <EmptyState
          icon={<Database size={20} />}
          title={search ? 'Nothing matches that filter' : 'No participant folders found yet'}
          body={
            search ? (
              'Clear the filter to see every folder the scan found.'
            ) : (
              <span className="stack gap-2">
                <span>A participant folder is one that holds all four of these files:</span>
                <span className="stack gap-1">
                  {REQUIRED_FILES.map((entry) => (
                    <span key={entry.file} className="t-tiny">
                      <span className="mono">{entry.file}</span>, {entry.why}
                    </span>
                  ))}
                </span>
              </span>
            )
          }
          action={
            !search && (
              <Button variant="primary" icon={<FolderSearch size={14} />} onClick={() => setBrowsing(true)}>
                Browse for a folder
              </Button>
            )
          }
        />
      ) : (
        <ul className="picker">
          {participants.map((participant) => {
            const chosen = isSelected(participant)
            return (
              <li key={participant.directory}>
                <button
                  type="button"
                  className={clsx('picker__row', chosen && 'picker__row--on', !participant.valid && 'picker__row--bad')}
                  onClick={() => choose(participant)}
                  disabled={!participant.valid}
                  aria-pressed={chosen}
                >
                  <span className="picker__mark" aria-hidden>
                    {chosen ? <Check size={13} /> : participant.valid ? null : <AlertTriangle size={13} />}
                  </span>
                  <span className="stack grow" style={{ gap: 2, minWidth: 0 }}>
                    <span className="row gap-2" style={{ minWidth: 0 }}>
                      <span className="semibold truncate">{participant.id}</span>
                      {participant.valid ? (
                        <Badge tone="neutral" mono>
                          {compactNumber(participant.input_tokens)} tok
                        </Badge>
                      ) : (
                        <Badge tone="critical">Incomplete</Badge>
                      )}
                    </span>
                    <span className="t-tiny muted truncate">{participant.directory}</span>
                    {participant.valid ? (
                      <span className="row gap-1 wrap" style={{ marginTop: 2 }}>
                        {participant.domains.slice(0, 6).map((domain) => (
                          <span key={domain} className="picker__domain">
                            {domain.replace(/_/g, ' ').toLowerCase()}
                          </span>
                        ))}
                        {participant.domains.length > 6 && (
                          <span className="picker__domain">+{participant.domains.length - 6}</span>
                        )}
                      </span>
                    ) : (
                      <span className="t-tiny" style={{ color: 'var(--critical)' }}>
                        {describeMissing(participant.missing)}
                      </span>
                    )}
                  </span>
                  {participant.valid && (
                    <Tooltip
                      content={
                        <div className="stack gap-1">
                          {Object.entries(participant.domain_coverage).map(([domain, coverage]) => (
                            <div key={domain}>
                              {domain}: {coverage.present_leaves}/{coverage.total_leaves} leaves,{' '}
                              {tokens(coverage.total_tokens)} tokens
                            </div>
                          ))}
                        </div>
                      }
                    >
                      <span className="picker__cov">{Object.keys(participant.domain_coverage).length} domains</span>
                    </Tooltip>
                  )}
                </button>
              </li>
            )
          })}
        </ul>
      )}

      {outcome && (
        <Callout
          tone={outcome.tone}
          icon={outcome.tone === 'positive' ? <Check size={15} /> : <AlertTriangle size={15} />}
          title={outcome.title}
          action={
            <Button size="sm" variant="ghost" onClick={() => setOutcome(null)}>
              Dismiss
            </Button>
          }
        >
          <div className="stack gap-2">
            <span>{outcome.body}</span>
            {outcome.nearMisses.length > 0 && <NearMissList outcome={outcome} />}
          </div>
        </Callout>
      )}

      <div className="stack gap-2">
        <span className="eyebrow row gap-2">
          Scanned folders
          <InfoDot>
            <p>
              Each folder here is searched up to four levels deep for directories that hold all four required
              files. The sample participants ship with the engine and are always scanned.
            </p>
            <p>
              A folder that holds some of the four is reported as a near miss with the filenames it lacks, which is
              usually a naming difference rather than missing data.
            </p>
          </InfoDot>
        </span>

        <div className="roots">
          {(data?.roots ?? []).map((root) => (
            <RootRow key={root.root} root={root} onRemove={() => removeRoot(root)} />
          ))}
          {(data?.roots ?? []).length === 0 && (
            <span className="t-tiny muted" style={{ padding: 'var(--s-3)' }}>
              No folders are being scanned yet.
            </span>
          )}
        </div>

        <div className="row gap-2 wrap">
          <Button icon={<FolderSearch size={14} />} onClick={() => setBrowsing(true)} style={{ flex: 'none' }}>
            Browse for a folder
          </Button>
          <Input
            className="grow"
            value={rootDraft}
            onChange={(event) => setRootDraft(event.target.value)}
            placeholder="or paste an absolute path"
            onKeyDown={(event) => event.key === 'Enter' && addRoot(rootDraft)}
          />
          <Button
            icon={<Plus size={14} />}
            onClick={() => addRoot(rootDraft)}
            loading={adding}
            disabled={!rootDraft.trim()}
            style={{ flex: 'none' }}
          >
            Add
          </Button>
        </div>
      </div>

      {participants.some((participant) => !participant.valid) && (
        <Callout tone="caution" icon={<AlertTriangle size={15} />} title="Some folders are incomplete">
          A folder needs all four input files before it can run. Each incomplete row lists what it lacks.
        </Callout>
      )}

      <FolderBrowser
        open={browsing}
        onClose={() => setBrowsing(false)}
        onAdd={(path) => void addRoot(path)}
        adding={adding}
        outcome={outcome}
      />
    </div>
  )
}

function RootRow({ root, onRemove }: { root: ScannedRoot; onRemove: () => void }) {
  const nearMisses = root.near_misses ?? []
  return (
    <div className="roots__row">
      <div className="row gap-3">
        <span className="stack grow" style={{ gap: 1, minWidth: 0 }}>
          <span className="t-small semibold truncate">{root.label}</span>
          <span className="t-micro faint truncate mono">{root.root}</span>
        </span>
        <Badge tone={root.found > 0 ? 'positive' : 'neutral'}>
          {root.found} found
        </Badge>
        {root.scanned_dir_count !== undefined && (
          <span className="t-micro faint tabular" style={{ flex: 'none' }}>
            {root.scanned_dir_count} scanned
          </span>
        )}
        {root.bundled ? (
          <Tooltip content="Ships with the engine and is always scanned">
            <Badge outline>bundled</Badge>
          </Tooltip>
        ) : (
          <Button
            size="sm"
            variant="ghost"
            iconOnly
            icon={<Trash2 size={13} />}
            aria-label={`Stop scanning ${root.label}`}
            onClick={onRemove}
          />
        )}
      </div>
      {nearMisses.length > 0 && (
        <Disclosure
          title={`${nearMisses.length} folder${nearMisses.length === 1 ? '' : 's'} came close`}
          subtitle="Some of the four files, but not all"
        >
          <div className="stack gap-1">
            {nearMisses.slice(0, 12).map((miss) => (
              <span key={miss.directory} className="t-micro">
                <span className="mono">{miss.directory}</span>, {describeMissing(miss.missing)}
              </span>
            ))}
            {nearMisses.length > 12 && <span className="t-micro faint">and {nearMisses.length - 12} more</span>}
          </div>
        </Disclosure>
      )}
    </div>
  )
}

function NearMissList({ outcome }: { outcome: AddOutcome }) {
  return (
    <div className="stack gap-1">
      <span className="t-tiny semibold">These came close:</span>
      {outcome.nearMisses.slice(0, 5).map((miss) => (
        <span key={miss.directory} className="t-micro">
          <span className="mono">{miss.directory}</span>, {describeMissing(miss.missing)}
        </span>
      ))}
      {outcome.nearMisses.length > 5 && (
        <span className="t-micro faint">and {outcome.nearMisses.length - 5} more</span>
      )}
    </div>
  )
}

/** Choosing whose data the run reads, with validation surfaced up front. */

import clsx from 'clsx'
import { AlertTriangle, Check, Database, FolderPlus, RefreshCw, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { compactNumber, tokens } from '@/lib/format'
import { queryKeys, useParticipants } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { Participant } from '@/lib/types'
import { Badge, Button, Callout, EmptyState, InfoDot, Input, Skeleton, Tooltip } from '@/components/ui/primitives'

export function ParticipantPicker({ multiple = false }: { multiple?: boolean }) {
  const { data, isLoading, refetch, isFetching } = useParticipants()
  const { selected, setSelected, toggleSelected, notify } = useApp()
  const [search, setSearch] = useState('')
  const [rootDraft, setRootDraft] = useState('')
  const [adding, setAdding] = useState(false)
  const client = useQueryClient()

  const participants = useMemo(() => {
    const rows = data?.participants ?? []
    const needle = search.trim().toLowerCase()
    if (!needle) return rows
    return rows.filter(
      (p) => p.id.toLowerCase().includes(needle) || p.directory.toLowerCase().includes(needle),
    )
  }, [data, search])

  const isSelected = (participant: Participant) =>
    selected.some((s) => s.directory === participant.directory)

  const choose = (participant: Participant) => {
    if (!participant.valid) return
    if (multiple) toggleSelected(participant)
    else setSelected(isSelected(participant) && selected.length === 1 ? [] : [participant])
  }

  const addRoot = async () => {
    const path = rootDraft.trim()
    if (!path) return
    setAdding(true)
    try {
      await api.datasets.addRoot(path)
      await client.invalidateQueries({ queryKey: queryKeys.participants })
      setRootDraft('')
      notify({ tone: 'positive', title: 'Folder added', body: 'Scanned for participant directories.' })
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'Could not add that folder',
        body: error instanceof Error ? error.message : String(error),
      })
    } finally {
      setAdding(false)
    }
  }

  return (
    <div className="stack gap-4">
      <div className="row gap-2 wrap">
        <div className="picker__search grow">
          <Search size={14} className="faint" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
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
          Rescan
        </Button>
        {multiple && participants.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              setSelected(
                selected.length === participants.filter((p) => p.valid).length
                  ? []
                  : participants.filter((p) => p.valid),
              )
            }
          >
            {selected.length === participants.filter((p) => p.valid).length ? 'Clear' : 'Select all'}
          </Button>
        )}
      </div>

      {isLoading ? (
        <div className="stack gap-2">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} height={54} radius={9} />
          ))}
        </div>
      ) : participants.length === 0 ? (
        <EmptyState
          icon={<Database size={20} />}
          title="No participant folders found"
          body="A participant folder holds four files: data_overview.json, hierarchical_deviation_map.json, multimodal_data.json, and non_numerical_data.txt. Point the dashboard at a folder that contains them."
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
                        Missing: {participant.missing.join(', ')}
                      </span>
                    )}
                  </span>
                  {participant.valid && (
                    <Tooltip
                      content={
                        <div className="stack gap-1">
                          {Object.entries(participant.domain_coverage).map(([domain, cov]) => (
                            <div key={domain}>
                              {domain}: {cov.present_leaves}/{cov.total_leaves} leaves, {tokens(cov.total_tokens)} tokens
                            </div>
                          ))}
                        </div>
                      }
                    >
                      <span className="picker__cov">
                        {Object.keys(participant.domain_coverage).length} domains
                      </span>
                    </Tooltip>
                  )}
                </button>
              </li>
            )
          })}
        </ul>
      )}

      {data && data.roots.length > 0 && (
        <div className="stack gap-2">
          <span className="eyebrow row gap-2">
            Scanned folders
            <InfoDot>
              <p>
                The dashboard scans these folders up to four levels deep for directories containing all four
                required files. Sample participants ship with the engine and are always scanned.
              </p>
            </InfoDot>
          </span>
          <div className="row gap-2 wrap">
            {data.roots.map((root) => (
              <Tooltip key={root.root} content={root.root}>
                <span className="badge badge--outline">
                  {root.label}
                  <span className="faint"> · {root.found}</span>
                </span>
              </Tooltip>
            ))}
          </div>
          <div className="row gap-2">
            <Input
              value={rootDraft}
              onChange={(e) => setRootDraft(e.target.value)}
              placeholder="/absolute/path/to/your/participants"
              onKeyDown={(e) => e.key === 'Enter' && addRoot()}
            />
            <Button icon={<FolderPlus size={14} />} onClick={addRoot} loading={adding} style={{ flex: 'none' }}>
              Add folder
            </Button>
          </div>
        </div>
      )}

      {participants.some((p) => !p.valid) && (
        <Callout tone="caution" icon={<AlertTriangle size={15} />} title="Some folders are incomplete">
          A folder needs all four input files before it can run. The missing filenames are listed on each row.
        </Callout>
      )}
    </div>
  )
}

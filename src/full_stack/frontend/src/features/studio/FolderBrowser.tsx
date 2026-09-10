/**
 * Walking the filesystem to find participant folders.
 *
 * Typing an absolute path from memory is the part people get wrong, so this
 * lists real directories, marks the ones that already qualify, and says what a
 * folder is missing while it is still on screen. Nothing here opens a file: the
 * listing is names and structure only.
 */

import clsx from 'clsx'
import {
  AlertTriangle,
  ArrowUp,
  Check,
  ChevronRight,
  Folder,
  FolderPlus,
  Search,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { ApiError } from '@/lib/api'
import { useBrowse } from '@/lib/hooks'
import type { BrowseEntry, BrowseResponse } from '@/lib/types'
import { Badge, Button, Callout, Modal, Skeleton, Spinner } from '@/components/ui/primitives'
import { REQUIRED_FILE_NAMES, describeMissing, type AddOutcome } from './datasetScan'

/** The service also reports whether the listed folder is itself an input set. */
type BrowseDetail = BrowseResponse & {
  is_participant?: boolean
  missing?: string[]
  truncated?: boolean
}

export function FolderBrowser({
  open,
  onClose,
  onAdd,
  adding,
  outcome,
}: {
  open: boolean
  onClose: () => void
  onAdd: (path: string) => void
  adding: boolean
  outcome: AddOutcome | null
}) {
  const [path, setPath] = useState('')
  const [filter, setFilter] = useState('')
  const [lastGood, setLastGood] = useState('')

  const browse = useBrowse(open ? path : null)
  const data = browse.data as BrowseDetail | undefined

  useEffect(() => {
    if (data?.path) setLastGood(data.path)
  }, [data?.path])

  const entries = useMemo(() => {
    const rows = data?.entries ?? []
    const needle = filter.trim().toLowerCase()
    const dirs = rows.filter((entry) => entry.is_dir)
    return needle ? dirs.filter((entry) => entry.name.toLowerCase().includes(needle)) : dirs
  }, [data, filter])

  const fileCount = (data?.entries ?? []).filter((entry) => !entry.is_dir).length
  const missing = data?.missing ?? REQUIRED_FILE_NAMES.filter(
    (name) => !(data?.entries ?? []).some((entry) => !entry.is_dir && entry.name === name),
  )
  const isParticipant = data?.is_participant ?? missing.length === 0

  const go = (next: string) => {
    setFilter('')
    setPath(next)
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Find a folder to scan"
      subtitle="Point at a folder that holds participant inputs, or at a single participant folder."
      width={860}
      footer={
        <div className="row between gap-3 wrap">
          <span className="t-tiny muted mono truncate" style={{ minWidth: 0 }}>
            {data?.path ?? (browse.isLoading ? 'reading the folder' : '')}
          </span>
          <span className="row gap-2">
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
            <Button
              variant="primary"
              icon={<FolderPlus size={14} />}
              loading={adding}
              disabled={!data?.path}
              onClick={() => data?.path && onAdd(data.path)}
            >
              Scan this folder
            </Button>
          </span>
        </div>
      }
    >
      <div className="fb">
        <aside className="fb__roots">
          <span className="eyebrow">Start from</span>
          {(data?.roots ?? []).map((root) => (
            <button
              key={root.path}
              type="button"
              className={clsx('fb__root', data?.path === root.path && 'fb__root--on')}
              onClick={() => go(root.path)}
            >
              <span className="truncate">{root.label}</span>
            </button>
          ))}
          <span className="t-micro faint">
            A participant folder holds {REQUIRED_FILE_NAMES.length} files. The scan looks four levels deep.
          </span>
        </aside>

        <div className="fb__main">
          <div className="row gap-2">
            <Button
              size="sm"
              variant="ghost"
              iconOnly
              icon={<ArrowUp size={14} />}
              aria-label="Go up one folder"
              disabled={!data?.parent}
              onClick={() => data?.parent && go(data.parent)}
            />
            <div className="picker__search grow">
              <Search size={13} className="faint" />
              <input
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                placeholder="Filter folders in here"
                aria-label="Filter folders"
              />
            </div>
            {browse.isFetching && <Spinner size={14} />}
          </div>

          {browse.isError && <BrowseError error={browse.error} lastGood={lastGood} onBack={go} />}

          {browse.isLoading && (
            <div className="stack gap-2">
              <Skeleton height={30} />
              <Skeleton height={30} />
              <Skeleton height={30} />
            </div>
          )}

          {data && (
            <>
              <div className={clsx('fb__status', isParticipant && 'fb__status--ok')}>
                {isParticipant ? (
                  <>
                    <Check size={14} />
                    <span className="t-tiny">
                      This folder is itself a participant input. Scanning it adds exactly one participant.
                    </span>
                  </>
                ) : missing.length === REQUIRED_FILE_NAMES.length ? (
                  <span className="t-tiny muted">
                    {entries.length} folder{entries.length === 1 ? '' : 's'} here, {fileCount} file
                    {fileCount === 1 ? '' : 's'}. Open a folder to look inside, or scan this one to search it.
                  </span>
                ) : (
                  <>
                    <AlertTriangle size={14} />
                    <span className="t-tiny">
                      Close: this folder is {describeMissing(missing)}.
                    </span>
                  </>
                )}
              </div>

              <div className="fb__list">
                {entries.length === 0 ? (
                  <span className="t-small muted" style={{ padding: 'var(--s-4)' }}>
                    {filter ? 'No folder here matches that filter.' : 'No folders in here.'}
                  </span>
                ) : (
                  entries.map((entry) => (
                    <FolderRow key={entry.path} entry={entry} onOpen={() => go(entry.path)} onAdd={() => onAdd(entry.path)} />
                  ))
                )}
              </div>

              {data.truncated && (
                <span className="t-micro faint">
                  This folder holds more entries than the listing shows. Use the filter, or paste the path directly.
                </span>
              )}
            </>
          )}

          {outcome && (
            <Callout
              tone={outcome.tone === 'positive' ? 'positive' : outcome.tone === 'info' ? 'info' : 'caution'}
              icon={outcome.tone === 'positive' ? <Check size={15} /> : <AlertTriangle size={15} />}
              title={outcome.title}
            >
              <div className="stack gap-2">
                <span>{outcome.body}</span>
                {outcome.nearMisses.length > 0 && (
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
                )}
              </div>
            </Callout>
          )}
        </div>
      </div>
    </Modal>
  )
}

function FolderRow({
  entry,
  onOpen,
  onAdd,
}: {
  entry: BrowseEntry
  onOpen: () => void
  onAdd: () => void
}) {
  return (
    <div className={clsx('fb__row', entry.is_participant && 'fb__row--participant')}>
      <button type="button" className="fb__open" onClick={onOpen}>
        <Folder size={14} className={entry.is_participant ? undefined : 'faint'} />
        <span className="truncate grow">{entry.name}</span>
        {entry.is_participant ? (
          <Badge tone="positive">participant</Badge>
        ) : entry.child_dir_count > 0 ? (
          <span className="t-micro faint tabular">
            {entry.child_dir_count} folder{entry.child_dir_count === 1 ? '' : 's'}
          </span>
        ) : (
          <span className="t-micro faint">empty</span>
        )}
        <ChevronRight size={13} className="faint" />
      </button>
      <Button size="sm" variant="ghost" onClick={onAdd}>
        Scan
      </Button>
    </div>
  )
}

function BrowseError({
  error,
  lastGood,
  onBack,
}: {
  error: unknown
  lastGood: string
  onBack: (path: string) => void
}) {
  const failure = error instanceof ApiError ? error : null
  const message = failure?.message || (error instanceof Error ? error.message : '')
  const goBack = lastGood ? (
    <Button size="sm" onClick={() => onBack(lastGood)}>
      Go back
    </Button>
  ) : undefined

  // The service answers a missing folder with "No directory at ...", so any
  // other 404 means this build simply does not serve the browser yet.
  if (failure?.status === 404 && !/no directory at/i.test(message)) {
    return (
      <Callout tone="caution" icon={<AlertTriangle size={15} />} title="This service build has no folder browser">
        Close this and paste an absolute path in the field below instead. Everything else works the same.
      </Callout>
    )
  }

  if (failure?.status === 404) {
    return (
      <Callout tone="caution" icon={<AlertTriangle size={15} />} title="That folder is no longer there" action={goBack}>
        {message}
      </Callout>
    )
  }

  if (failure?.status === 400) {
    return (
      <Callout tone="caution" icon={<AlertTriangle size={15} />} title="That path cannot be used" action={goBack}>
        {message}
      </Callout>
    )
  }

  return (
    <Callout tone="caution" icon={<AlertTriangle size={15} />} title="That folder could not be listed" action={goBack}>
      {message || 'The service did not answer.'} This is usually a permission the service process does not have.
    </Callout>
  )
}

/**
 * What COMPASS has written on this machine, and how to remove it.
 *
 * Every row is a real path with a real size. Nothing is deleted without a
 * second press, and the sentence between the two presses says what is lost
 * rather than what is stored.
 */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { RotateCcw, Trash2, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button, Callout, Spinner } from '@/components/ui/primitives'
import { bytes } from '@/lib/format'
import { queryKeys } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { Group } from '../controls'
import { RESET_LOSS, useSettingsReset } from '../reset'
import { LOSS, StorageUnavailable, readStorage, removeStorage, type StorageEntry } from '../storage'

const STORAGE_KEY = ['system-storage'] as const

export function StoragePanel() {
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)
  // The same reset About offers. Nothing about it depends on this listing.
  const { reset, busy: resetting } = useSettingsReset()

  const storage = useQuery({ queryKey: STORAGE_KEY, queryFn: readStorage, retry: false, staleTime: 30_000 })
  const [armed, setArmed] = useState('')
  const [busy, setBusy] = useState('')

  const unavailable = storage.error instanceof StorageUnavailable

  const afterChange = async (key: string) => {
    await client.invalidateQueries({ queryKey: STORAGE_KEY })
    if (key === 'runs') await client.invalidateQueries({ queryKey: queryKeys.runs })
    if (key === 'prompts') await client.invalidateQueries({ queryKey: queryKeys.prompts })
    if (key === 'model_catalog') await client.invalidateQueries({ queryKey: ['catalog'] })
    if (key === 'outputs') await client.invalidateQueries({ queryKey: ['report'] })
  }

  const remove = async (entry: StorageEntry) => {
    setBusy(entry.key)
    try {
      const result = await removeStorage(entry.key)
      setArmed('')
      await afterChange(entry.key)
      notify({
        tone: 'positive',
        title: `${entry.label} cleared`,
        body: result.freed_bytes ? `${bytes(result.freed_bytes)} freed on disk.` : 'There was nothing left to remove.',
      })
    } catch (error) {
      notify({
        tone: 'critical',
        title: `${entry.label} was not cleared`,
        body: error instanceof Error ? error.message : 'The dashboard service rejected the request.',
      })
    } finally {
      setBusy('')
    }
  }

  const resetConfig = async () => {
    if (await reset()) setArmed('')
  }

  return (
    <Group title="Storage on this machine">
      {storage.isLoading && (
        <div className="row gap-2">
          <Spinner size={13} /> <span className="t-tiny muted">Measuring what is on disk</span>
        </div>
      )}

      {unavailable && (
        <span className="t-small muted">
          This build of the dashboard service does not report local storage, so nothing can be listed or cleared from
          here.
        </span>
      )}

      {storage.isError && !unavailable && (
        <Callout
          tone="critical"
          icon={<TriangleAlert size={15} />}
          action={
            <Button size="sm" onClick={() => void storage.refetch()}>
              Try again
            </Button>
          }
        >
          {(storage.error as Error).message}
        </Callout>
      )}

      {storage.data && (
        <>
          <div className="settings__roots">
            {storage.data.entries.map((entry) => (
              <StorageRow
                key={entry.key}
                entry={entry}
                armed={armed === entry.key}
                busy={entry.key === 'config' ? resetting : busy === entry.key}
                onArm={() => setArmed(entry.key)}
                onCancel={() => setArmed('')}
                onConfirm={() => void (entry.key === 'config' ? resetConfig() : remove(entry))}
              />
            ))}
          </div>
          <span className="t-tiny muted tabular">{bytes(storage.data.total_bytes)} in total.</span>
        </>
      )}
    </Group>
  )
}

function StorageRow({
  entry,
  armed,
  busy,
  onArm,
  onCancel,
  onConfirm,
}: {
  entry: StorageEntry
  armed: boolean
  busy: boolean
  onArm: () => void
  onCancel: () => void
  onConfirm: () => void
}) {
  const isConfig = entry.key === 'config'
  const empty = !entry.exists || entry.bytes === 0
  // The configuration row performs the same reset About offers, so it says the
  // same sentence rather than a second wording of it.
  const loss = isConfig
    ? RESET_LOSS
    : (LOSS[entry.key] ?? `Everything under ${entry.path} is deleted. This cannot be undone.`)

  return (
    <div className="settings__store" data-armed={armed}>
      <div className="row gap-3 wrap">
        <span className="stack grow" style={{ gap: 2, minWidth: 0 }}>
          <span className="row gap-2 wrap">
            <span className="t-small semibold">{entry.label}</span>
            <span className="t-tiny muted tabular">{empty ? 'empty' : bytes(entry.bytes)}</span>
            {entry.item_count !== null && entry.item_count > 0 && (
              <Badge outline>
                {entry.item_count} {entry.item_count === 1 ? 'item' : 'items'}
              </Badge>
            )}
          </span>
          {entry.description && <span className="t-tiny muted">{entry.description}</span>}
          <span className="mono t-micro faint truncate" title={entry.path}>
            {entry.path}
          </span>
        </span>
        {!armed && entry.removable && (
          <Button
            size="sm"
            variant={isConfig ? 'secondary' : 'ghost'}
            icon={isConfig ? <RotateCcw size={13} /> : <Trash2 size={13} />}
            disabled={empty && !isConfig}
            onClick={onArm}
          >
            {isConfig ? 'Reset to defaults' : 'Delete'}
          </Button>
        )}
      </div>

      {armed && (
        <div className="stack gap-2">
          <span className="t-tiny" style={{ color: 'var(--critical)' }}>
            {loss}
          </span>
          <div className="row gap-2">
            <Button size="sm" onClick={onCancel}>
              Cancel
            </Button>
            <Button variant="danger" size="sm" loading={busy} onClick={onConfirm}>
              {isConfig ? 'Reset everything' : `Delete ${entry.label.toLowerCase()}`}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

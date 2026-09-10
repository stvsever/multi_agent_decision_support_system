/** Where participants are read from, where output is written, and what is kept. */

import { useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button, Field, Input } from '@/components/ui/primitives'
import { ApiError, api } from '@/lib/api'
import { queryKeys, useParticipants } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { Group, SectionHead, SwitchRow, TextSetting } from '../controls'
import { useSettingsController } from '../state'
import { StoragePanel } from './StoragePanel'

export function DataSection() {
  const { config, update } = useSettingsController()
  const workspace = config.workspace
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)
  const participants = useParticipants()
  const [path, setPath] = useState('')
  const [busy, setBusy] = useState('')

  const refresh = async () => {
    await client.invalidateQueries({ queryKey: queryKeys.settings })
    await client.invalidateQueries({ queryKey: queryKeys.participants })
  }

  const add = async () => {
    const trimmed = path.trim()
    if (!trimmed) return
    setBusy('add')
    try {
      await api.datasets.addRoot(trimmed)
      setPath('')
      await refresh()
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'That folder was not added',
        body: error instanceof ApiError ? error.message : 'The path could not be read.',
      })
    } finally {
      setBusy('')
    }
  }

  const remove = async (root: string) => {
    setBusy(root)
    try {
      await api.datasets.removeRoot(root)
      await refresh()
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'That folder was not removed',
        body: error instanceof ApiError ? error.message : 'The dashboard service rejected the change.',
      })
    } finally {
      setBusy('')
    }
  }

  const rootStatus = (root: string) => participants.data?.roots.find((entry) => entry.root === root)

  return (
    <>
      <SectionHead
        title="Data"
        description="Folders scanned for participants, where a finished report is written, and what COMPASS keeps on disk."
      />

      <Group title="Data roots">
        {workspace.data_roots.length > 0 && (
          <div className="settings__roots">
            {workspace.data_roots.map((root) => {
              const status = rootStatus(root)
              const near = status?.near_misses?.length ?? 0
              return (
                <div key={root} className="settings__root">
                  <span className="grow truncate" title={root}>
                    {root}
                  </span>
                  {near > 0 && (
                    <span
                      className="t-micro muted"
                      title="Directories that hold some of the four required input files"
                    >
                      {near} incomplete
                    </span>
                  )}
                  {status?.found !== undefined && (
                    <Badge tone={status.found ? 'positive' : 'caution'}>{status.found} found</Badge>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    icon={<Trash2 size={13} />}
                    aria-label={`Remove ${root}`}
                    loading={busy === root}
                    onClick={() => void remove(root)}
                  />
                </div>
              )
            })}
          </div>
        )}

        <Field
          label="Add a folder"
          hint="An absolute path. Every participant directory inside it is discovered and validated. The bundled sample participants are always available as well."
        >
          <div className="row gap-2">
            <Input
              mono
              value={path}
              placeholder="/data/cohort/participants"
              aria-label="Data root to add"
              onChange={(event) => setPath(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') void add()
              }}
            />
            <Button
              variant="secondary"
              icon={<Plus size={13} />}
              loading={busy === 'add'}
              disabled={!path.trim()}
              onClick={() => void add()}
            >
              Add
            </Button>
          </div>
        </Field>
      </Group>

      <Group title="Output">
        <TextSetting
          label="Output directory"
          section="workspace"
          field="output_dir"
          mono
          placeholder="Leave empty for the default output folder"
          hint="Reports, execution logs, and audit artifacts are written here, one folder per participant."
        />
        <SwitchRow
          label="Open the report when a run finishes"
          hint="Moves straight to the report screen as soon as the communicator is done."
          checked={workspace.auto_open_report}
          onChange={(next) => update('workspace', { auto_open_report: next })}
        />
      </Group>

      <StoragePanel />
    </>
  )
}

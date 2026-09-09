/** Where participants are read from and where output is written. */

import { useQueryClient } from '@tanstack/react-query'
import { FolderTree, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button, Callout, Field, Input } from '@/components/ui/primitives'
import { ApiError, api } from '@/lib/api'
import { queryKeys, useParticipants } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { Group, SectionHead, SwitchRow, TextSetting } from '../controls'
import { useSettingsController } from '../state'

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

  const found = (root: string) => participants.data?.roots.find((entry) => entry.root === root)?.found

  return (
    <>
      <SectionHead
        title="Data"
        description="Folders scanned for participants, and where a finished report is written."
      />

      <Group title="Data roots">
        <Callout tone="neutral" icon={<FolderTree size={15} />}>
          The sample participants bundled with COMPASS are always available, so you can try a run before pointing the
          engine at your own data. Add a folder here and every participant directory inside it is discovered and
          validated.
        </Callout>

        {workspace.data_roots.length > 0 && (
          <div className="settings__roots">
            {workspace.data_roots.map((root) => (
              <div key={root} className="settings__root">
                <span className="grow truncate" title={root}>
                  {root}
                </span>
                {found(root) !== undefined && (
                  <Badge tone={found(root) ? 'positive' : 'caution'}>{found(root)} found</Badge>
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
            ))}
          </div>
        )}

        <Field label="Add a folder" hint="An absolute path to a directory holding participant folders.">
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
    </>
  )
}

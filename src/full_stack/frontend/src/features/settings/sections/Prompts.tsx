/**
 * The engine's real system prompts.
 *
 * These are the files each agent and tool loads when it is constructed, so an
 * edit lands on the next run with no restart. The original is snapshotted the
 * first time a prompt is saved, which is what makes restore trustworthy.
 */

import { useQueryClient } from '@tanstack/react-query'
import { Undo2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Badge, Button, EmptyState, Spinner, Tabs, Textarea } from '@/components/ui/primitives'
import { ApiError, api } from '@/lib/api'
import { queryKeys, usePrompts } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { PromptRow } from '@/lib/types'
import { SectionHead } from '../controls'

export function PromptsSection() {
  const prompts = usePrompts()
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)

  const rows = prompts.data?.prompts ?? []
  const [scope, setScope] = useState<'agent' | 'tool'>('agent')
  const [chosen, setChosen] = useState<{ scope: 'agent' | 'tool'; name: string } | null>(null)
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [original, setOriginal] = useState('')

  // The row is looked up every render so "modified" and the snapshot flag stay
  // current after a save refreshes the listing.
  const selected = chosen
    ? (rows.find((row) => row.scope === chosen.scope && row.name === chosen.name) ?? null)
    : null

  useEffect(() => {
    if (!chosen) return
    let cancelled = false
    setLoading(true)
    api.prompts
      .get(chosen.scope, chosen.name)
      .then((row) => {
        if (cancelled) return
        setContent(row.content)
        setOriginal(row.content)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        notify({
          tone: 'critical',
          title: 'That prompt could not be read',
          body: error instanceof ApiError ? error.message : 'The file is missing or unreadable.',
        })
        setChosen(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [chosen, notify])

  const save = async () => {
    if (!selected) return
    setSaving(true)
    try {
      await api.prompts.save(selected.scope, selected.name, content)
      setOriginal(content)
      await client.invalidateQueries({ queryKey: queryKeys.prompts })
      notify({ tone: 'positive', title: `${selected.title} saved`, body: 'It takes effect on the next run.' })
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'The prompt was not saved',
        body: error instanceof ApiError ? error.message : 'The dashboard service rejected the change.',
      })
    } finally {
      setSaving(false)
    }
  }

  const restore = async () => {
    if (!selected) return
    setSaving(true)
    try {
      const restored = await api.prompts.restore(selected.scope, selected.name)
      setContent(restored.content)
      setOriginal(restored.content)
      await client.invalidateQueries({ queryKey: queryKeys.prompts })
      notify({ tone: 'positive', title: `${selected.title} restored`, body: 'The shipped text is back in place.' })
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'The prompt was not restored',
        body: error instanceof ApiError ? error.message : 'No snapshot of the original is available.',
      })
    } finally {
      setSaving(false)
    }
  }

  const agents = rows.filter((row) => row.scope === 'agent')
  const tools = rows.filter((row) => row.scope === 'tool')
  const listed: PromptRow[] = scope === 'agent' ? agents : tools

  const dirty = content !== original

  return (
    <>
      <SectionHead
        title="Prompts"
        description="The system prompts the agents and tools actually load. A saved edit applies to the next run."
      />

      <Tabs
        spread
        value={scope}
        options={[
          { value: 'agent', label: `Agents (${agents.length})` },
          { value: 'tool', label: `Tools (${tools.length})` },
        ]}
        onChange={setScope}
      />

      <div className="settings__prompts">
        <div className="settings__promptlist">
          {prompts.isLoading && (
            <div className="row gap-2" style={{ padding: 'var(--s-3)' }}>
              <Spinner /> <span className="t-tiny muted">Loading</span>
            </div>
          )}
          {listed.map((row) => (
            <button
              key={`${row.scope}/${row.name}`}
              type="button"
              className="settings__promptitem"
              aria-current={chosen?.name === row.name && chosen.scope === row.scope}
              onClick={() => setChosen({ scope: row.scope, name: row.name })}
            >
              <span className="stack grow" style={{ gap: 1, minWidth: 0 }}>
                <span className="truncate semibold">{row.title}</span>
                <span className="t-micro muted tabular">{row.characters.toLocaleString()} characters</span>
              </span>
              {row.modified && <Badge tone="caution">modified</Badge>}
            </button>
          ))}
        </div>

        <div className="stack gap-3" style={{ minWidth: 0 }}>
          {!selected && (
            <EmptyState
              title="Select a prompt"
              body="Each entry is the system prompt for one agent or one tool. Editing changes how a role reasons; study conventions belong in Instructions."
            />
          )}

          {selected && (
            <>
              <div className="row between gap-3 wrap">
                <div className="stack" style={{ gap: 1, minWidth: 0 }}>
                  <span className="t-small semibold truncate">{selected.title}</span>
                  <span className="t-micro muted mono truncate">
                    {selected.scope}/{selected.name}
                  </span>
                </div>
                <div className="row gap-2">
                  <Button
                    size="sm"
                    icon={<Undo2 size={13} />}
                    disabled={!selected.has_snapshot || saving}
                    title={
                      selected.has_snapshot
                        ? 'Put the shipped text back'
                        : 'There is no snapshot yet, because this prompt has never been saved'
                    }
                    onClick={() => void restore()}
                  >
                    Restore default
                  </Button>
                  <Button variant="primary" size="sm" loading={saving} disabled={!dirty} onClick={() => void save()}>
                    Save
                  </Button>
                </div>
              </div>

              <span className="t-tiny muted">
                {selected.description ? `${selected.description} ` : ''}A prompt that contradicts the output schema
                will make this agent fail.
              </span>

              {loading ? (
                <div className="row gap-2 center" style={{ padding: 'var(--s-8)' }}>
                  <Spinner /> <span className="t-small muted">Reading the prompt</span>
                </div>
              ) : (
                <Textarea
                  mono
                  className="settings__editor"
                  aria-label={`${selected.title} system prompt`}
                  spellCheck={false}
                  value={content}
                  onChange={(event) => setContent(event.target.value)}
                />
              )}

              <div className="row between gap-2">
                <span className="t-micro faint tabular">{content.length.toLocaleString()} characters</span>
                {dirty && <span className="t-micro" style={{ color: 'var(--caution)' }}>Unsaved changes</span>}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  )
}

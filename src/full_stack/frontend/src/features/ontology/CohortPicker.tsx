/**
 * Who the page is about.
 *
 * One participant reads their own tree. Two or more merge into one, so the
 * control has to make the difference obvious before the tree changes under the
 * reader, and it has to say which participant is being marked inside the
 * cohort.
 */

import { AnimatePresence, motion } from 'framer-motion'
import { Check, Crosshair, Search, Users } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Badge, Button, Tooltip } from '@/components/ui/primitives'
import { compactNumber } from '@/lib/format'
import type { Participant } from '@/lib/types'

export interface CohortPickerProps {
  participants: Participant[]
  /** Selected directories, which is what every ontology query is keyed on. */
  selected: string[]
  onChange: (next: string[]) => void
  focus: string
  onFocusChange: (next: string) => void
  reducedMotion: boolean
}

export function CohortPicker({
  participants,
  selected,
  onChange,
  focus,
  onFocusChange,
  reducedMotion,
}: CohortPickerProps) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onPointer = (event: MouseEvent) => {
      if (!wrapRef.current?.contains(event.target as globalThis.Node)) setOpen(false)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('mousedown', onPointer)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onPointer)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase()
    if (!needle) return participants
    return participants.filter(
      (row) =>
        row.id.toLowerCase().includes(needle) ||
        row.name.toLowerCase().includes(needle) ||
        row.directory.toLowerCase().includes(needle),
    )
  }, [participants, search])

  const chosen = useMemo(() => new Set(selected), [selected])
  const focused = participants.find((row) => row.directory === focus)
  const label =
    selected.length === 0
      ? 'No participant'
      : selected.length === 1
        ? (participants.find((row) => row.directory === selected[0])?.id ?? '1 participant')
        : `${selected.length} participants`

  const toggle = (directory: string) => {
    if (chosen.has(directory)) {
      const next = selected.filter((item) => item !== directory)
      onChange(next)
      if (focus === directory && next.length > 0) onFocusChange(next[0])
    } else {
      onChange([...selected, directory])
    }
  }

  return (
    <div className="onto-cohort" ref={wrapRef}>
      <Button
        icon={<Users size={14} />}
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen((value) => !value)}
      >
        {label}
        {selected.length > 1 && (
          <Badge tone="accent" className="onto-cohort__tag">
            cohort
          </Badge>
        )}
      </Button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="onto-cohort__panel"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: reducedMotion ? 0 : 0.16, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="onto-cohort__head">
              <span className="onto-cohort__search">
                <Search size={13} className="faint" />
                <input
                  value={search}
                  autoFocus
                  placeholder="Filter participants"
                  aria-label="Filter participants"
                  onChange={(event) => setSearch(event.target.value)}
                />
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  onChange(
                    selected.length === participants.length
                      ? participants.slice(0, 1).map((row) => row.directory)
                      : participants.map((row) => row.directory),
                  )
                }
              >
                {selected.length === participants.length ? 'Just one' : 'Select all'}
              </Button>
            </div>

            <div className="onto-cohort__list">
              {rows.length === 0 && <span className="t-tiny muted">Nothing matches that filter.</span>}
              {rows.map((row) => {
                const on = chosen.has(row.directory)
                return (
                  <div key={row.directory} className="onto-cohort__row" data-on={on || undefined}>
                    <button
                      type="button"
                      className="onto-cohort__pick"
                      aria-pressed={on}
                      onClick={() => toggle(row.directory)}
                    >
                      <span className="onto-cohort__mark" aria-hidden="true">
                        {on && <Check size={11} />}
                      </span>
                      <span className="stack grow" style={{ gap: 1, minWidth: 0 }}>
                        <span className="t-small semibold truncate">{row.id}</span>
                        <span className="t-micro muted truncate">{row.directory}</span>
                      </span>
                      <span className="t-micro faint tabular">{compactNumber(row.input_tokens)}</span>
                    </button>
                    {selected.length > 1 && on && (
                      <Tooltip content="Mark this one inside the cohort">
                        <button
                          type="button"
                          className="onto-cohort__focus"
                          aria-pressed={focus === row.directory}
                          aria-label={`Mark ${row.id} inside the cohort`}
                          onClick={() => onFocusChange(row.directory)}
                        >
                          <Crosshair size={12} />
                        </button>
                      </Tooltip>
                    )}
                  </div>
                )
              })}
            </div>

            <div className="onto-cohort__foot t-micro muted">
              {selected.length > 1 ? (
                <span>
                  Merged tree over {selected.length} participants
                  {focused ? `, marking ${focused.id}` : ''}. Leaves only one participant carries
                  stay visible and stay marked.
                </span>
              ) : (
                <span>Add a second participant to merge the trees and see distributions.</span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

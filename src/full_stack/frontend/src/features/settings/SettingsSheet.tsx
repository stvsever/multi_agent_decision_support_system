/**
 * The settings surface.
 *
 * One wide dialog with a section list on the left and a single panel on the
 * right. Every control in it writes straight through to the service; there is
 * no Save button and no draft to lose.
 */

import {
  Coins,
  Cpu,
  Database,
  Gauge,
  Info,
  MessageSquareText,
  Palette,
  SlidersHorizontal,
} from 'lucide-react'
import { useEffect, useMemo, useRef, type ComponentType, type JSX } from 'react'
import { Modal, Spinner } from '@/components/ui/primitives'
import { useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { SaveIndicator } from './controls'
import { AboutSection } from './sections/About'
import { AppearanceSection } from './sections/Appearance'
import { ComputeSection } from './sections/Compute'
import { CostSection } from './sections/Cost'
import { DataSection } from './sections/Data'
import { EngineSection } from './sections/Engine'
import { InstructionsSection } from './sections/Instructions'
import { PromptsSection } from './sections/Prompts'
import { SettingsProvider, useSettingsController } from './state'
import './settings.css'

interface SectionDef {
  id: string
  label: string
  icon: ComponentType<{ size?: number }>
  render: () => JSX.Element
}

/* Ordered the way the questions actually arrive: what is this, how should it
   look, where does it think, how hard does it work, what may it spend, what
   does it read and write, then the two prompt surfaces. */
const SECTIONS: SectionDef[] = [
  { id: 'about', label: 'About', icon: Info, render: () => <AboutSection /> },
  { id: 'appearance', label: 'Appearance', icon: Palette, render: () => <AppearanceSection /> },
  { id: 'compute', label: 'Models and compute', icon: Cpu, render: () => <ComputeSection /> },
  { id: 'engine', label: 'Engine', icon: Gauge, render: () => <EngineSection /> },
  { id: 'cost', label: 'Cost guardrails', icon: Coins, render: () => <CostSection /> },
  { id: 'workspace', label: 'Data', icon: Database, render: () => <DataSection /> },
  { id: 'instructions', label: 'Instructions', icon: MessageSquareText, render: () => <InstructionsSection /> },
  { id: 'prompts', label: 'Prompts', icon: SlidersHorizontal, render: () => <PromptsSection /> },
]

/* Sections that were folded into a larger one still answer to their old name,
   so every deep link written before the merge lands somewhere sensible. */
const ALIASES: Record<string, string> = {
  connection: 'compute',
  models: 'compute',
  local: 'compute',
  budgets: 'engine',
  batch: 'engine',
  data: 'workspace',
}

const resolveSection = (requested: string): string => {
  const target = ALIASES[requested] ?? requested
  return SECTIONS.find((entry) => entry.id === target)?.id ?? SECTIONS[0].id
}

const isKnown = (requested: string) => Boolean(ALIASES[requested]) || SECTIONS.some((e) => e.id === requested)

const HASH_PREFIX = '#settings='

export function SettingsSheet(): JSX.Element | null {
  const settingsOpen = useApp((s) => s.settingsOpen)
  const openSettings = useApp((s) => s.openSettings)
  const closeSettings = useApp((s) => s.closeSettings)
  const settings = useSettings()
  const container = useRef<HTMLDivElement>(null)

  const open = settingsOpen !== false
  const section = useMemo(
    () => resolveSection(typeof settingsOpen === 'string' ? settingsOpen : ''),
    [settingsOpen],
  )

  /* A pasted URL can name a section. The hash is written while the sheet is
     open and cleared on close, so it never lingers in the address bar. */
  useEffect(() => {
    const requested = window.location.hash.startsWith(HASH_PREFIX)
      ? window.location.hash.slice(HASH_PREFIX.length)
      : ''
    if (requested && isKnown(requested)) openSettings(requested)
  }, [openSettings])

  useEffect(() => {
    try {
      const target = open ? `${HASH_PREFIX}${section}` : ''
      const url = `${window.location.pathname}${window.location.search}${target}`
      window.history.replaceState(window.history.state, '', url)
    } catch {
      /* an unavailable history API is not worth failing the dialog over */
    }
  }, [open, section])

  /* Focus starts on the section list and stays inside the dialog while it is
     open, which is what makes Escape the only way out for a keyboard user. */
  useEffect(() => {
    if (!open) return
    const modal = container.current?.closest<HTMLElement>('.modal')
    if (!modal) return
    container.current?.querySelector<HTMLButtonElement>('.settings__navitem[aria-selected="true"]')?.focus()

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return
      const focusable = [
        ...modal.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ].filter((element) => element.offsetParent !== null)
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const activeElement = document.activeElement as HTMLElement | null
      if (event.shiftKey && (activeElement === first || !modal.contains(activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    modal.addEventListener('keydown', onKeyDown)
    return () => modal.removeEventListener('keydown', onKeyDown)
  }, [open])

  if (!open) return null

  const active = SECTIONS.find((entry) => entry.id === section) ?? SECTIONS[0]

  return (
    <Modal
      open={open}
      onClose={closeSettings}
      width={940}
      title="Settings"
      subtitle="Everything the engine can be told, in one place. Changes save as you make them."
    >
      <div className="settings" ref={container}>
        <div
          className="settings__nav"
          role="tablist"
          aria-orientation="vertical"
          aria-label="Settings sections"
        >
          {SECTIONS.map((entry) => (
            <button
              key={entry.id}
              id={`settings-tab-${entry.id}`}
              type="button"
              role="tab"
              className="settings__navitem"
              aria-selected={entry.id === active.id}
              aria-controls="settings-panel"
              tabIndex={entry.id === active.id ? 0 : -1}
              onKeyDown={(event) => {
                const step = event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0
                if (!step) return
                event.preventDefault()
                const index = SECTIONS.findIndex((item) => item.id === active.id)
                const next = SECTIONS[(index + step + SECTIONS.length) % SECTIONS.length]
                openSettings(next.id)
                container.current
                  ?.querySelector<HTMLButtonElement>(`#settings-tab-${next.id}`)
                  ?.focus()
              }}
              onClick={() => openSettings(entry.id)}
            >
              <entry.icon size={15} />
              <span className="truncate">{entry.label}</span>
            </button>
          ))}
        </div>

        {settings.data ? (
          /* One provider for the whole dialog, so a pending save survives a
             move between sections. */
          <SettingsProvider config={settings.data.config}>
            <div
              className="settings__panel"
              key={active.id}
              id="settings-panel"
              role="tabpanel"
              aria-labelledby={`settings-tab-${active.id}`}
            >
              {settings.data.notices && settings.data.notices.length > 0 && (
                <div className="callout callout--caution">
                  <div className="grow stack gap-1">
                    {settings.data.notices.map((notice) => (
                      <span key={notice}>{notice}</span>
                    ))}
                  </div>
                </div>
              )}
              {active.render()}
            </div>
            <SaveBadge />
          </SettingsProvider>
        ) : (
          <div className="settings__panel">
            {settings.isLoading && (
              <div className="row gap-2 center" style={{ padding: 'var(--s-12)' }}>
                <Spinner /> <span className="t-small muted">Loading settings</span>
              </div>
            )}
            {settings.isError && (
              <div className="callout callout--critical">
                The dashboard service is not reachable, so no setting can be read or changed. Confirm it is running,
                then reopen this dialog.
              </div>
            )}
          </div>
        )}
      </div>
    </Modal>
  )
}

function SaveBadge() {
  const { status } = useSettingsController()
  return <SaveIndicator status={status} />
}

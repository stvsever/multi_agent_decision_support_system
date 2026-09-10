/**
 * First-run welcome.
 *
 * Shown once, on the very first visit, and then only when it is asked for
 * again from the compass mark in the rail. It says what the engine does and
 * what it needs, and offers the tour rather than insisting on it.
 *
 * Asking for it again is transient state of its own. The persisted first-run
 * marker is written once and never cleared, so reopening About cannot make the
 * dialog reappear by itself on the next reload.
 *
 * Entry is a CSS animation rather than an animated mount, so the dialog leaves
 * the moment it is dismissed. An exit animation can outlive its own trigger
 * and leave a scrim sitting over the tour it just started.
 */

import { ArrowRight, Compass } from 'lucide-react'
import { useCallback, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { Button } from '@/components/ui/primitives'
import { useApp } from '@/lib/store'
import { closeWelcome, useWelcomeReopened } from './welcomeOpen'
import './welcome.css'

const LINES = [
  "COMPASS reads one participant's multi-modal data and answers the question you set about it.",
  'A team of agents plans the work, predicts, criticises the prediction, then writes it up. Every claim in the report points back at the evidence behind it.',
  'Two things are needed before a first run: a model connection, and at least one participant folder.',
  'A structural audit walks the whole path without calling a model, so the first attempt costs nothing.',
]

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'

export function WelcomeDialog() {
  const welcomeSeen = useApp((s) => s.welcomeSeen)
  const tourActive = useApp((s) => s.tourActive)
  const dismissWelcome = useApp((s) => s.dismissWelcome)
  const startTour = useApp((s) => s.startTour)
  const reopened = useWelcomeReopened()

  const open = (!welcomeSeen || reopened) && !tourActive
  const cardRef = useRef<HTMLDivElement>(null)
  const restoreTo = useRef<HTMLElement | null>(null)

  const close = useCallback(() => {
    closeWelcome()
    dismissWelcome()
  }, [dismissWelcome])

  // Clearing the reopen flag here too, so the dialog does not come back when
  // the tour it started ends.
  const takeTour = useCallback(() => {
    closeWelcome()
    startTour()
  }, [startTour])

  useEffect(() => {
    if (!open) return
    restoreTo.current = document.activeElement as HTMLElement | null
    const card = cardRef.current
    card?.querySelector<HTMLElement>('[data-autofocus]')?.focus()

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        close()
        return
      }
      if (event.key !== 'Tab' || !card) return
      const items = Array.from(card.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => !el.hasAttribute('disabled'))
      if (items.length === 0) return
      const first = items[0]
      const last = items[items.length - 1]
      const active = document.activeElement
      if (event.shiftKey && (active === first || !card.contains(active))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (active === last || !card.contains(active))) {
        event.preventDefault()
        first.focus()
      }
    }

    window.addEventListener('keydown', onKey, true)
    return () => {
      window.removeEventListener('keydown', onKey, true)
      restoreTo.current?.focus?.()
    }
  }, [open, close])

  if (!open) return null

  return createPortal(
    <div
      className="welcome"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close()
      }}
    >
      {/* Centring is the grid parent's job, so nothing here depends on a
          transform that an animation could overwrite. */}
      <div className="welcome__card" ref={cardRef} role="dialog" aria-modal="true" aria-labelledby="welcome-title">
        <span className="welcome__mark" aria-hidden>
          <Compass size={20} />
        </span>
        <h2 className="welcome__title" id="welcome-title">
          Welcome to COMPASS
        </h2>
        <div className="welcome__lines">
          {LINES.map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
        <div className="welcome__actions">
          <Button variant="primary" size="lg" icon={<ArrowRight size={14} />} data-autofocus onClick={takeTour}>
            Take the tour
          </Button>
          <Button variant="ghost" size="lg" onClick={close}>
            Look around
          </Button>
        </div>
        <p className="welcome__foot">The tour takes under a minute. Reopen this from the compass mark in the rail.</p>
      </div>
    </div>,
    document.body,
  )
}

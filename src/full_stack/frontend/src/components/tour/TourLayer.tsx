/**
 * Guided walkthrough.
 *
 * The tour dims everything except the element being described and cuts a hole
 * in the scrim over it, so attention lands on the real interface rather than on
 * a description of it. Three things matter here and are easy to get wrong:
 *
 *  - the card is placed with top and left only, never a transform, because
 *    framer-motion owns the transform on that element and would overwrite it;
 *  - the target is scrolled into view and then left to settle before its rect
 *    is committed, otherwise the hole is measured mid-scroll and lands short;
 *  - a step whose target does not exist is skipped rather than shown centred,
 *    because a centred card silently describes something that is not there.
 *
 * The placement is arithmetic on the viewport, so the viewport is held in state
 * and the arithmetic is redone on resize. The hole is a real hole: the scrim
 * paints but takes no clicks, and four shields around the spot stop everything
 * else, so the highlighted control can still be used.
 */

import { AnimatePresence, motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Compass, X } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useLocation, useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/primitives'
import { useApp } from '@/lib/store'
import './tour.css'

type Placement = 'auto' | 'right' | 'bottom' | 'left' | 'top'

interface TourStep {
  id: string
  title: string
  body: string
  /** `data-tour` value of the element to spotlight. Absent means centre screen. */
  target?: string
  /** Route the tour moves to before showing this step. */
  route?: string
  placement?: Placement
}

const STEPS: TourStep[] = [
  {
    id: 'welcome',
    title: 'A quick orientation',
    body: 'COMPASS answers a question about one participant by reading all of their data and writing up what it found. This takes under a minute and you can leave at any point.',
    route: '/',
  },
  {
    id: 'participants',
    title: 'Choose whose data to read',
    body: 'Pick the participant folders to work from. Samples ship with the engine, so you can try the whole flow before bringing your own.',
    target: 'studio-participants',
    route: '/studio',
    placement: 'right',
  },
  {
    id: 'task',
    title: 'State the question',
    body: 'Name the outcome you want answered. The task family fixes what the Predictor has to return and what the Critic scores it against.',
    target: 'studio-task',
    route: '/studio',
    placement: 'right',
  },
  {
    id: 'cost',
    title: 'See the cost before you spend it',
    body: 'The projection is priced from the live catalog and the size of the evidence. A structural audit walks the same path with no model calls at all.',
    target: 'studio-launch',
    route: '/studio',
    placement: 'left',
  },
  {
    id: 'runs',
    title: 'Watch the plan execute',
    body: 'A run is drawn as a live graph. Parallel steps share a column, and the critic loop is drawn as a return path.',
    target: 'nav-runs',
    route: '/runs',
    placement: 'right',
  },
  {
    id: 'ontology',
    title: 'Follow the evidence hierarchy',
    body: 'Each measurement sits under a broader category, and every level carries the deviation beneath it. Read it as a tree, a sunburst, an icicle or a table.',
    target: 'nav-ontology',
    route: '/ontology',
    placement: 'right',
  },
  {
    id: 'reports',
    title: 'Read the report',
    body: 'The Communicator writes the answer grounded in that evidence and marks what was missing. The whole thing exports as a PDF.',
    target: 'nav-reports',
    route: '/reports',
    placement: 'right',
  },
  {
    id: 'settings',
    title: 'Everything else is in settings',
    body: 'The model, per-role overrides, budgets and the prompts themselves all live here. Change any of it before a run.',
    target: 'settings-button',
    placement: 'bottom',
  },
]

interface Spot {
  top: number
  left: number
  width: number
  height: number
  radius: number
}

const CARD_W = 380
const GAP = 14
/** Tab is kept inside the card, so the card needs to know what is focusable. */
const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
/** How long a target gets to mount before the step is skipped. */
const RESOLVE_MS = 1200
/** Upper bound on waiting for a smooth scroll to settle. */
const SETTLE_MS = 700

export function TourLayer() {
  const { tourActive, tourStep, setTourStep, endTour } = useApp()
  const reducedMotion = useApp((s) => s.appearance.reducedMotion)
  const navigate = useNavigate()
  const location = useLocation()
  const [spot, setSpot] = useState<Spot | null>(null)
  const [cardH, setCardH] = useState(196)
  const [viewport, setViewport] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }))
  const cardRef = useRef<HTMLDivElement | null>(null)
  /** Which way the user is moving, so a skipped step skips the same way. */
  const direction = useRef(1)
  /** Where focus was before the tour took it. */
  const restoreTo = useRef<HTMLElement | null>(null)

  const step = STEPS[tourStep]

  const finish = useCallback(() => {
    endTour()
    setSpot(null)
    // Back to whatever opened the tour, or to the gear the last step describes
    // when that control has since been unmounted by a route change. The body is
    // not a restore point: focusing it is a no-op that would strand the caret.
    const previous = restoreTo.current
    const usable = previous && previous.isConnected && previous !== document.body ? previous : null
    const target = usable ?? document.querySelector<HTMLElement>('[data-tour="settings-button"]')
    target?.focus({ preventScroll: true })
  }, [endTour])

  const go = useCallback(
    (next: number) => {
      direction.current = next >= tourStep ? 1 : -1
      if (next < 0 || next >= STEPS.length) finish()
      else setTourStep(next)
    },
    [tourStep, setTourStep, finish],
  )

  // Move to the step's route before measuring, so the target can exist.
  useEffect(() => {
    if (!tourActive || !step?.route) return
    if (location.pathname !== step.route) navigate(step.route)
  }, [tourActive, step?.route, location.pathname, navigate])

  useEffect(() => {
    if (!tourActive || !step) return
    if (!step.target) {
      setSpot(null)
      return
    }
    // Wait for the route change; this effect re-runs when it lands.
    if (step.route && location.pathname !== step.route) return

    let cancelled = false
    let frame = 0
    let observer: ResizeObserver | null = null
    let deadline = performance.now() + RESOLVE_MS

    const commit = (element: HTMLElement, radius: number) => {
      const next = geometry(element, radius)
      setSpot((prev) => (same(prev, next) ? prev : next))
    }

    // A target can be replaced under us: a poll that remounts the panel the
    // step is pointing at leaves a detached node measuring 0x0. Committing that
    // would collapse the hole into the top-left corner, so the search starts
    // again instead, and the step is skipped if nothing comes back.
    const gone = (element: HTMLElement) =>
      !element.isConnected || element.getBoundingClientRect().width === 0
    const restart = () => {
      observer?.disconnect()
      observer = null
      deadline = performance.now() + RESOLVE_MS
      frame = requestAnimationFrame(find)
    }

    // Third: stay glued to the element for as long as the step is showing.
    const track = (element: HTMLElement) => {
      let radius = cornerRadius(element)
      const loop = () => {
        if (cancelled) return
        if (gone(element)) {
          restart()
          return
        }
        commit(element, radius)
        frame = requestAnimationFrame(loop)
      }
      frame = requestAnimationFrame(loop)
      observer = new ResizeObserver(() => {
        if (gone(element)) return
        radius = cornerRadius(element)
        commit(element, radius)
      })
      observer.observe(element)
    }

    // Second: let the scroll finish. Reading the rect straight after asking for
    // a smooth scroll measures the element half way through the animation.
    const settle = (element: HTMLElement) => {
      element.scrollIntoView({
        block: 'center',
        inline: 'nearest',
        behavior: reducedMotion ? 'auto' : 'smooth',
      })
      const until = performance.now() + SETTLE_MS
      let previous: Spot | null = null
      let steady = 0
      const poll = () => {
        if (cancelled) return
        if (gone(element)) {
          restart()
          return
        }
        const radius = cornerRadius(element)
        const now = geometry(element, radius)
        steady = previous && same(previous, now) ? steady + 1 : 0
        previous = now
        if (steady >= 2 || performance.now() > until) {
          setSpot(now)
          track(element)
          return
        }
        frame = requestAnimationFrame(poll)
      }
      frame = requestAnimationFrame(poll)
    }

    // First: find the target, allowing for a screen that is still mounting. A
    // hidden or zero-sized element counts as absent: there is nothing to point
    // at, so the step is skipped rather than shown against empty space.
    const find = () => {
      if (cancelled) return
      const element = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`)
      if (element && element.getBoundingClientRect().width > 0) {
        settle(element)
        return
      }
      if (performance.now() > deadline) {
        go(tourStep + direction.current)
        return
      }
      frame = requestAnimationFrame(find)
    }

    frame = requestAnimationFrame(find)
    return () => {
      cancelled = true
      cancelAnimationFrame(frame)
      observer?.disconnect()
    }
  }, [tourActive, tourStep, step, location.pathname, reducedMotion, go])

  // Measured before paint, so the centred opener is centred on its first frame.
  useLayoutEffect(() => {
    const node = cardRef.current
    if (!node) return
    const read = () => {
      const height = node.getBoundingClientRect().height
      setCardH((prev) => (Math.abs(prev - height) < 1 ? prev : height))
    }
    read()
    const observer = new ResizeObserver(read)
    observer.observe(node)
    return () => observer.disconnect()
  }, [tourActive, tourStep])

  // The centred opener is placed by arithmetic on the viewport, so the
  // arithmetic has to be redone when the viewport changes. Steps with a target
  // are re-measured by the tracking loop; this one has no loop to ride on.
  useEffect(() => {
    if (!tourActive) return
    const read = () =>
      setViewport((prev) =>
        prev.width === window.innerWidth && prev.height === window.innerHeight
          ? prev
          : { width: window.innerWidth, height: window.innerHeight },
      )
    read()
    window.addEventListener('resize', read)
    window.addEventListener('orientationchange', read)
    return () => {
      window.removeEventListener('resize', read)
      window.removeEventListener('orientationchange', read)
    }
  }, [tourActive])

  // Focus follows the card. Without this the focused control is whatever the
  // welcome dialog handed back, which is behind the shields and cannot be used.
  useEffect(() => {
    if (!tourActive) return
    if (restoreTo.current === null) {
      const active = document.activeElement as HTMLElement | null
      restoreTo.current = active && active !== document.body ? active : null
    }
    const card = cardRef.current
    if (!card) return
    const target = card.querySelector<HTMLElement>('[data-autofocus]') ?? card
    target.focus({ preventScroll: true })
  }, [tourActive, tourStep])

  useEffect(() => {
    if (tourActive) return
    restoreTo.current = null
  }, [tourActive])

  useEffect(() => {
    if (!tourActive) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        finish()
        return
      }
      if (event.key === 'ArrowRight') {
        go(tourStep + 1)
        return
      }
      if (event.key === 'ArrowLeft') {
        go(Math.max(0, tourStep - 1))
        return
      }
      if (event.key !== 'Tab') return
      // Tab stays in the card: everything else on the screen is either behind a
      // shield or is the one control the step is pointing at.
      const card = cardRef.current
      if (!card) return
      const items = Array.from(card.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
        (element) => !element.hasAttribute('disabled'),
      )
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
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [tourActive, tourStep, go, finish])

  if (!tourActive || !step) return null

  const last = tourStep === STEPS.length - 1
  // A card wider than the window is not a card, so it gives way to the window.
  const cardW = Math.min(CARD_W, Math.max(200, viewport.width - GAP * 2))
  const place = cardPlacement(spot, step.placement ?? 'auto', cardH, cardW, viewport)
  const duration = reducedMotion ? 0 : 0.28
  const shields = shieldRects(spot, viewport)

  return createPortal(
    <AnimatePresence>
      <motion.div
        key="tour"
        className="tour"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: reducedMotion ? 0 : 0.18 }}
      >
        {/* One scrim with a mask, so the target stays fully lit while
            everything around it dims. It paints only: what it dims is put out
            of reach by the shields below, which leave the lit hole alone. */}
        <svg className="tour__scrim" aria-hidden>
          <defs>
            <mask id="tour-hole">
              <rect x="0" y="0" width="100%" height="100%" fill="white" />
              {spot && (
                <motion.rect
                  initial={false}
                  animate={{ x: spot.left, y: spot.top, width: spot.width, height: spot.height }}
                  transition={{ duration, ease: [0.22, 1, 0.36, 1] }}
                  rx={spot.radius}
                  fill="black"
                />
              )}
            </mask>
          </defs>
          <rect x="0" y="0" width="100%" height="100%" mask="url(#tour-hole)" className="tour__scrim-fill" />
        </svg>

        {/* Clicks land on the interface only inside the hole. Nothing here
            dismisses the tour: a stray click on the control being described
            used to end the walkthrough instead of doing what it looked like. */}
        {shields.map((shield) => (
          <div key={shield.key} className="tour__shield" style={{ top: shield.top, left: shield.left, width: shield.width, height: shield.height }} aria-hidden />
        ))}

        {spot && (
          <motion.div
            className="tour__ring"
            style={{ borderRadius: spot.radius }}
            initial={false}
            animate={{ top: spot.top, left: spot.left, width: spot.width, height: spot.height }}
            transition={{ duration, ease: [0.22, 1, 0.36, 1] }}
          />
        )}

        {/* Position is top and left only. framer-motion writes the transform on
            this element, so any centring done with translate is lost. */}
        <motion.div
          ref={cardRef}
          className="tour__card"
          style={{ top: place.top, left: place.left, width: cardW }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: reducedMotion ? 0 : 0.2 }}
          role="dialog"
          aria-modal="true"
          aria-label={step.title}
          tabIndex={-1}
        >
          <div className="tour__card-head">
            <span className="tour__badge">
              <Compass size={13} />
            </span>
            <span className="grow t-small semibold">{step.title}</span>
            <Button variant="ghost" size="sm" iconOnly icon={<X size={14} />} onClick={finish} aria-label="End the tour" />
          </div>
          <p className="tour__body">{step.body}</p>
          <div className="tour__foot">
            <div className="tour__dots">
              {STEPS.map((s, index) => (
                <button
                  key={s.id}
                  type="button"
                  aria-label={`Step ${index + 1}: ${s.title}`}
                  className="tour__dot"
                  data-on={index === tourStep}
                  data-seen={index < tourStep}
                  onClick={() => go(index)}
                />
              ))}
            </div>
            <div className="row gap-2">
              <Button size="sm" variant="ghost" icon={<ArrowLeft size={13} />} disabled={tourStep === 0} onClick={() => go(tourStep - 1)}>
                Back
              </Button>
              <Button
                size="sm"
                variant="primary"
                icon={last ? undefined : <ArrowRight size={13} />}
                data-autofocus
                onClick={() => (last ? finish() : go(tourStep + 1))}
              >
                {last ? 'Done' : 'Next'}
              </Button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body,
  )
}

/** The element's own corner radius, in pixels, clamped to what it can show. */
function cornerRadius(element: HTMLElement): number {
  const raw = getComputedStyle(element).borderTopLeftRadius.trim()
  const value = Number.parseFloat(raw)
  if (!Number.isFinite(value)) return 0
  const box = element.getBoundingClientRect()
  const px = raw.endsWith('%') ? (value / 100) * box.width : value
  return Math.min(px, Math.min(box.width, box.height) / 2)
}

/**
 * The hole around the element. Padding scales with the element so a control
 * gets a tight cut-out and a card gets a looser one, and the corner stays
 * concentric with the element's own corner instead of a flat 12px guess.
 */
function geometry(element: HTMLElement, radius: number): Spot {
  const box = element.getBoundingClientRect()
  const shortest = Math.min(box.width, box.height)
  const pad = Math.max(3, Math.min(12, shortest * 0.09))
  const height = box.height + pad * 2
  const width = box.width + pad * 2
  return {
    top: box.top - pad,
    left: box.left - pad,
    width,
    height,
    radius: Math.min(radius + pad, Math.min(width, height) / 2),
  }
}

/** Sub-pixel churn is not worth a render. */
function same(a: Spot | null, b: Spot | null): boolean {
  if (!a || !b) return a === b
  return (
    Math.abs(a.top - b.top) < 0.5 &&
    Math.abs(a.left - b.left) < 0.5 &&
    Math.abs(a.width - b.width) < 0.5 &&
    Math.abs(a.height - b.height) < 0.5 &&
    Math.abs(a.radius - b.radius) < 0.5
  )
}

const SIDES: Placement[] = ['right', 'left', 'bottom', 'top']

interface Viewport {
  width: number
  height: number
}

interface Shield {
  key: string
  top: number
  left: number
  width: number
  height: number
}

/**
 * The four bands around the spotlight, which is to say the whole screen minus
 * the hole. They are what stops a click, so the lit element keeps its own hit
 * area and everything else is out of reach. With no spot the screen is covered.
 */
function shieldRects(spot: Spot | null, view: Viewport): Shield[] {
  if (!spot) return [{ key: 'all', top: 0, left: 0, width: view.width, height: view.height }]
  const right = spot.left + spot.width
  const bottom = spot.top + spot.height
  const top = clamp(spot.top, 0, view.height)
  const height = clamp(bottom, 0, view.height) - top
  return [
    { key: 'top', top: 0, left: 0, width: view.width, height: Math.max(0, top) },
    { key: 'bottom', top: Math.max(0, bottom), left: 0, width: view.width, height: Math.max(0, view.height - bottom) },
    { key: 'left', top, left: 0, width: Math.max(0, spot.left), height: Math.max(0, height) },
    { key: 'right', top, left: Math.max(0, right), width: Math.max(0, view.width - right), height: Math.max(0, height) },
  ]
}

/**
 * Place the card beside the spotlight, or dead centre when there is none.
 * Centring is arithmetic on top and left rather than a translate, because the
 * card's transform belongs to the animation. The viewport is passed in rather
 * than read from `window` so that a resize re-runs this: read straight from
 * `window` during render, the centred card kept its first position for ever and
 * ended up off screen. A side that has no room is dropped for one that does,
 * rather than clamped into the element it is supposed to be pointing at.
 */
function cardPlacement(
  spot: Spot | null,
  placement: Placement,
  cardH: number,
  cardW: number,
  view: Viewport,
): { top: number; left: number } {
  const vw = view.width
  const vh = view.height
  if (!spot) {
    return { top: Math.max(GAP, (vh - cardH) / 2), left: Math.max(GAP, (vw - cardW) / 2) }
  }

  const order = placement === 'auto' ? SIDES : [placement, ...SIDES.filter((s) => s !== placement)]
  const fits = (side: Placement) => {
    if (side === 'right') return spot.left + spot.width + GAP + cardW <= vw - GAP
    if (side === 'left') return spot.left - GAP - cardW >= GAP
    if (side === 'bottom') return spot.top + spot.height + GAP + cardH <= vh - GAP
    return spot.top - GAP - cardH >= GAP
  }
  const side = order.find(fits) ?? order[0]

  let top: number
  let left: number
  if (side === 'right' || side === 'left') {
    left = side === 'right' ? spot.left + spot.width + GAP : spot.left - cardW - GAP
    top = spot.top + spot.height / 2 - cardH / 2
  } else {
    left = spot.left + spot.width / 2 - cardW / 2
    top = side === 'bottom' ? spot.top + spot.height + GAP : spot.top - cardH - GAP
  }

  return {
    top: clamp(top, GAP, Math.max(GAP, vh - cardH - GAP)),
    left: clamp(left, GAP, Math.max(GAP, vw - cardW - GAP)),
  }
}

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

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
/**
 * Uniform breathing room around the highlighted element. It is a constant on
 * purpose: scaling it with the element made a small control and a tall panel
 * carry visibly different frames, which read as the highlight not fitting what
 * it was pointing at.
 */
const SPOT_PAD = 6
/** Clearance kept above a target that is taller than the viewport. */
const SCROLL_MARGIN = 24
/** How long the tour takes to bring a target into place. */
const SCROLL_MS = 380
/** Tab is kept inside the card, so the card needs to know what is focusable. */
const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
/** How long a target gets to mount before the step is skipped. */
const RESOLVE_MS = 1200

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

  /**
   * Resolve the step's target, bring it to rest, and pin the highlight to it.
   *
   * There is deliberately no per-frame chase loop. Scrolling is locked for the
   * duration (see below), so once the target is in place nothing can move it
   * except a layout change, and those announce themselves: a ResizeObserver on
   * the element and a window resize listener are enough. The old loop measured
   * on every frame and re-committed, which meant the highlight was continuously
   * catching up to its target rather than sitting on it.
   *
   * Nothing here rides on requestAnimationFrame either. rAF stops entirely on a
   * hidden tab, which left the walkthrough resolving nothing at all until the
   * tab came back; timers keep running, so the step still lands.
   */
  useEffect(() => {
    if (!tourActive || !step) return
    if (!step.target) {
      setSpot(null)
      return
    }
    // Wait for the route change; this effect re-runs when it lands.
    if (step.route && location.pathname !== step.route) return

    let cancelled = false
    let timer = 0
    let observer: ResizeObserver | null = null
    const later = (fn: () => void, ms = 16) => {
      timer = window.setTimeout(fn, ms)
    }

    const usable = (element: HTMLElement) =>
      element.isConnected && element.getBoundingClientRect().width > 0

    /** Pin the highlight and keep it pinned against layout changes only. */
    const pin = (element: HTMLElement) => {
      const measure = () => {
        if (cancelled || !usable(element)) return
        const next = geometry(element, cornerRadius(element))
        setSpot((prev) => (same(prev, next) ? prev : next))
      }
      measure()
      observer = new ResizeObserver(measure)
      observer.observe(element)
      // A panel elsewhere growing or shrinking moves this one without resizing
      // it, so the frame it sits in is watched too.
      const host = element.closest('.shell__content') ?? document.body
      observer.observe(host)
      window.addEventListener('resize', measure)
      cleanupResize = () => window.removeEventListener('resize', measure)
    }
    let cleanupResize: (() => void) | null = null

    /** Move the target to its resting place, then pin. */
    const settle = (element: HTMLElement) => {
      const container = scrollParent(element)
      if (!container) {
        pin(element)
        return
      }
      const from = container.scrollTop
      const to = restingScrollTop(container, element)
      if (reducedMotion || Math.abs(to - from) < 1) {
        container.scrollTop = to
        pin(element)
        return
      }
      // The exact resting position is known before the move starts, so the
      // highlight is committed on arrival rather than guessed at from a rect
      // that is still changing.
      const started = performance.now()
      const glide = () => {
        if (cancelled) return
        const progress = Math.min(1, (performance.now() - started) / SCROLL_MS)
        container.scrollTop = from + (to - from) * (1 - Math.pow(1 - progress, 3))
        if (progress < 1) {
          later(glide)
          return
        }
        container.scrollTop = to
        if (usable(element)) pin(element)
      }
      later(glide)
    }

    // A screen may still be mounting, so the target is polled for rather than
    // demanded at once. A step whose target never appears is skipped in the
    // direction of travel: a centred card would silently describe something
    // that is not on the screen.
    const deadline = performance.now() + RESOLVE_MS
    const find = () => {
      if (cancelled) return
      const element = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`)
      if (element && usable(element)) {
        settle(element)
        return
      }
      if (performance.now() > deadline) {
        go(tourStep + direction.current)
        return
      }
      later(find)
    }

    find()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
      observer?.disconnect()
      cleanupResize?.()
    }
  }, [tourActive, tourStep, step, location.pathname, reducedMotion, go])

  /**
   * Nothing moves during the walkthrough except by pressing Back or Next.
   *
   * A scroll the tour did not ask for slides the interface out from under the
   * highlight, and the tracking loop then chases it, which is what made the
   * spotlight look like it was drifting off its target. Every scroller is
   * frozen for the duration and the tour moves them itself. The lock is
   * overflow rather than a wheel handler alone, so a trackpad, a scrollbar
   * drag, and a page key are all covered by the same mechanism; the handlers
   * below only stop the browser scrolling something the lock cannot reach.
   */
  useEffect(() => {
    if (!tourActive) return
    const frozen: { node: HTMLElement; overflow: string }[] = []
    const freeze = (node: HTMLElement | null) => {
      if (!node) return
      frozen.push({ node, overflow: node.style.overflow })
      node.style.overflow = 'hidden'
    }
    freeze(document.documentElement)
    freeze(document.body)
    document.querySelectorAll<HTMLElement>('.shell__content').forEach(freeze)

    const swallow = (event: Event) => event.preventDefault()
    const KEYS = new Set([' ', 'PageUp', 'PageDown', 'Home', 'End', 'ArrowUp', 'ArrowDown'])
    const keys = (event: KeyboardEvent) => {
      // Typing inside the card is not scrolling, so a field keeps its keys.
      const target = event.target as HTMLElement | null
      if (target && target.closest('input, textarea, [contenteditable="true"]')) return
      if (KEYS.has(event.key)) event.preventDefault()
    }
    window.addEventListener('wheel', swallow, { passive: false })
    window.addEventListener('touchmove', swallow, { passive: false })
    window.addEventListener('keydown', keys)
    return () => {
      window.removeEventListener('wheel', swallow)
      window.removeEventListener('touchmove', swallow)
      window.removeEventListener('keydown', keys)
      for (const entry of frozen) entry.node.style.overflow = entry.overflow
    }
  }, [tourActive])

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
        {/* The dim, the hole and the ring are one shape drawn three times, so
            they are all read from `spot` in this render and moved by CSS.
            Animating the mask and the ring through framer while the shields
            stayed put meant three different ideas of where the hole was: for a
            few frames one of them dimmed what another had lit, which is what
            made a step look half highlighted. */}
        <svg className="tour__scrim" aria-hidden>
          <defs>
            <mask id="tour-hole" maskUnits="userSpaceOnUse">
              <rect x="0" y="0" width={viewport.width} height={viewport.height} fill="white" />
              {spot && (
                <rect
                  className="tour__hole"
                  x={spot.left}
                  y={spot.top}
                  width={spot.width}
                  height={spot.height}
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
          <div
            className="tour__ring"
            style={{
              borderRadius: spot.radius,
              top: spot.top,
              left: spot.left,
              width: spot.width,
              height: spot.height,
            }}
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
 * The hole around the element: its own box, grown by one constant on every
 * side, with the corner kept concentric with the element's own corner. Every
 * highlight therefore traces its target at the same offset, which is what makes
 * the frame read as belonging to the thing underneath it.
 */
function geometry(element: HTMLElement, radius: number): Spot {
  const box = element.getBoundingClientRect()
  const height = box.height + SPOT_PAD * 2
  const width = box.width + SPOT_PAD * 2
  return {
    top: box.top - SPOT_PAD,
    left: box.left - SPOT_PAD,
    width,
    height,
    radius: Math.min(radius + SPOT_PAD, Math.min(width, height) / 2),
  }
}

/**
 * The nearest ancestor that actually scrolls.
 *
 * `hidden` counts: the tour locks its scrollers by setting overflow hidden, and
 * a locked container is still the one whose scrollTop has to move to bring the
 * next target into view.
 */
function scrollParent(element: HTMLElement): HTMLElement | null {
  let node: HTMLElement | null = element.parentElement
  while (node && node !== document.body) {
    const overflow = getComputedStyle(node).overflowY
    const scrolls = overflow === 'auto' || overflow === 'scroll' || overflow === 'hidden'
    if (scrolls && node.scrollHeight > node.clientHeight + 1) return node
    node = node.parentElement
  }
  return null
}

/** Where the container has to sit for `element` to be properly in view. */
function restingScrollTop(container: HTMLElement, element: HTMLElement): number {
  const view = container.clientHeight
  const top =
    element.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop
  const height = element.getBoundingClientRect().height
  // Centred whenever it fits at all, even without the usual margin around it:
  // a panel that is a little shorter than the frame should be shown whole, and
  // insisting on the margin pushed its lower edge off the screen instead. Only
  // a target that genuinely cannot fit falls back to anchoring its top, since
  // centring that one would take its heading off the top.
  const wanted = height <= view ? top - (view - height) / 2 : top - SCROLL_MARGIN
  return clamp(wanted, 0, Math.max(0, container.scrollHeight - view))
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
  const room = (side: Placement) => {
    if (side === 'right') return vw - GAP - (spot.left + spot.width + GAP)
    if (side === 'left') return spot.left - GAP * 2
    if (side === 'bottom') return vh - GAP - (spot.top + spot.height + GAP)
    return spot.top - GAP * 2
  }
  const need = (side: Placement) => (side === 'right' || side === 'left' ? cardW : cardH)
  // A spot that fills the screen leaves no side with room, and the card then
  // has to sit over it. Falling back to the requested side put it over the
  // middle of the panel; the roomiest side keeps it against an edge, where it
  // covers the least.
  const side = order.find((option) => room(option) >= need(option)) ??
    [...order].sort((a, b) => room(b) - room(a))[0]

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

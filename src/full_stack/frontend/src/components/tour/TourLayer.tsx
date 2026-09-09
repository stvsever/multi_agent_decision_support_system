/**
 * Guided walkthrough.
 *
 * The tour dims everything except the element being described and cuts a hole
 * in the scrim over it, so attention lands on the real interface rather than on
 * a screenshot of it. Steps that target a missing element are skipped rather
 * than left pointing at nothing.
 */

import { AnimatePresence, motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Compass, X } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { useLocation, useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/primitives'
import { useApp } from '@/lib/store'
import './tour.css'

interface TourStep {
  id: string
  title: string
  body: string
  /** `data-tour` value of the element to spotlight. Absent means centre screen. */
  target?: string
  /** Route the tour moves to before showing this step. */
  route?: string
  placement?: 'auto' | 'right' | 'bottom' | 'left' | 'top'
}

const STEPS: TourStep[] = [
  {
    id: 'welcome',
    title: 'A quick orientation',
    body: "COMPASS reads a participant's multi-modal evidence and answers a phenotype question through an orchestrated actor-critic workflow. This takes about a minute and you can leave at any point.",
    route: '/',
  },
  {
    id: 'connect',
    title: 'Connect a provider first',
    body: 'The engine calls models through OpenRouter. Add your key once in settings and the dashboard verifies it, reads live pricing, and shows your remaining credit.',
    target: 'settings-button',
    placement: 'bottom',
  },
  {
    id: 'participants',
    title: 'Choose whose data to read',
    body: 'A participant folder holds four files: a coverage overview, a hierarchical deviation map, the multi-modal feature payload, and free-text health information. Sample participants ship with the engine so you can try the whole flow first.',
    target: 'studio-participants',
    route: '/studio',
    placement: 'right',
  },
  {
    id: 'task',
    title: 'State the question',
    body: 'Five task families are supported: binary, multiclass, univariate regression, multivariate regression, and a hierarchical tree that mixes them. The family fixes the contract the Predictor must satisfy and the checklist the Critic scores against.',
    target: 'studio-task',
    route: '/studio',
    placement: 'right',
  },
  {
    id: 'model',
    title: 'Confirm the model',
    body: 'One model drives every agent by default. Per-role models, token ceilings, temperatures, and the system prompts themselves are all adjustable in settings when you need that resolution.',
    target: 'studio-model',
    route: '/studio',
    placement: 'right',
  },
  {
    id: 'launch',
    title: 'See the cost before you spend it',
    body: "The projection is fitted against real run ledgers and priced from the live catalog, so it moves with the size of the participant's evidence. A structural audit runs the whole data path with no provider calls at all.",
    target: 'studio-launch',
    route: '/studio',
    placement: 'left',
  },
  {
    id: 'runs',
    title: 'Watch the workflow execute',
    body: "The run console renders the orchestrator's plan as a live graph: parallel steps sit in the same column, dependencies carry a travelling marker showing direction, and the critic loop is drawn as a return path.",
    target: 'nav-runs',
    route: '/runs',
    placement: 'right',
  },
  {
    id: 'ontology',
    title: 'Navigate the evidence taxonomy',
    body: 'Every measurement is a leaf in an IS-A ontology, and each node carries the mean absolute deviation beneath it. Explore it as a tree, a sunburst, an icicle, or a ranked table.',
    target: 'nav-ontology',
    route: '/ontology',
    placement: 'right',
  },
  {
    id: 'reports',
    title: 'Read the deep phenotype report',
    body: 'The Communicator writes an evidence-grounded report and marks missing information explicitly. Everything is also exportable as a standardized PDF.',
    target: 'nav-reports',
    route: '/reports',
    placement: 'right',
  },
]

interface Rect {
  top: number
  left: number
  width: number
  height: number
}

const PAD = 8

export function TourLayer() {
  const { tourActive, tourStep, setTourStep, endTour } = useApp()
  const navigate = useNavigate()
  const location = useLocation()
  const [rect, setRect] = useState<Rect | null>(null)

  const step = STEPS[tourStep]

  const finish = useCallback(() => {
    endTour()
    setRect(null)
  }, [endTour])

  // Move to the step's route before measuring, so the target exists.
  useEffect(() => {
    if (!tourActive || !step?.route) return
    if (location.pathname !== step.route) navigate(step.route)
  }, [tourActive, step?.route, location.pathname, navigate])

  useLayoutEffect(() => {
    if (!tourActive || !step) return
    let frame = 0
    let attempts = 0

    const measure = () => {
      if (!step.target) {
        setRect(null)
        return
      }
      const element = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`)
      if (!element) {
        // The route may still be mounting; retry briefly, then skip the step.
        attempts += 1
        if (attempts > 30) {
          setRect(null)
          return
        }
        frame = requestAnimationFrame(measure)
        return
      }
      element.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      const box = element.getBoundingClientRect()
      setRect({ top: box.top - PAD, left: box.left - PAD, width: box.width + PAD * 2, height: box.height + PAD * 2 })
    }

    frame = requestAnimationFrame(measure)
    const onChange = () => requestAnimationFrame(measure)
    window.addEventListener('resize', onChange)
    window.addEventListener('scroll', onChange, true)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', onChange)
      window.removeEventListener('scroll', onChange, true)
    }
  }, [tourActive, tourStep, step, location.pathname])

  useEffect(() => {
    if (!tourActive) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') finish()
      if (event.key === 'ArrowRight') setTourStep(Math.min(STEPS.length - 1, tourStep + 1))
      if (event.key === 'ArrowLeft') setTourStep(Math.max(0, tourStep - 1))
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [tourActive, tourStep, setTourStep, finish])

  if (!tourActive || !step) return null

  const last = tourStep === STEPS.length - 1
  const card = cardPosition(rect, step.placement ?? 'auto')

  return createPortal(
    <AnimatePresence>
      <motion.div
        key="tour"
        className="tour"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
      >
        {/* The scrim is a single element with an even-odd mask so the target
            stays fully interactive and fully lit while everything else dims. */}
        <svg className="tour__scrim" onClick={finish} aria-hidden>
          <defs>
            <mask id="tour-hole">
              <rect x="0" y="0" width="100%" height="100%" fill="white" />
              {rect && (
                <motion.rect
                  initial={false}
                  animate={{ x: rect.left, y: rect.top, width: rect.width, height: rect.height }}
                  transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
                  rx={12}
                  fill="black"
                />
              )}
            </mask>
          </defs>
          <rect x="0" y="0" width="100%" height="100%" mask="url(#tour-hole)" className="tour__scrim-fill" />
        </svg>

        {rect && (
          <motion.div
            className="tour__ring"
            initial={false}
            animate={{ top: rect.top, left: rect.left, width: rect.width, height: rect.height }}
            transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
          />
        )}

        <motion.div
          className="tour__card"
          style={card}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
          role="dialog"
          aria-label={step.title}
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
                  onClick={() => setTourStep(index)}
                />
              ))}
            </div>
            <div className="row gap-2">
              <Button
                size="sm"
                variant="ghost"
                icon={<ArrowLeft size={13} />}
                disabled={tourStep === 0}
                onClick={() => setTourStep(tourStep - 1)}
              >
                Back
              </Button>
              <Button
                size="sm"
                variant="primary"
                icon={last ? undefined : <ArrowRight size={13} />}
                onClick={() => (last ? finish() : setTourStep(tourStep + 1))}
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

const CARD_W = 340
const CARD_H = 190
const GAP = 14

/** Place the card beside the spotlight, keeping it inside the viewport. */
function cardPosition(rect: Rect | null, placement: string): React.CSSProperties {
  if (!rect) {
    return { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }
  }
  const vw = window.innerWidth
  const vh = window.innerHeight

  let side = placement
  if (side === 'auto') {
    if (rect.left + rect.width + GAP + CARD_W < vw) side = 'right'
    else if (rect.left - GAP - CARD_W > 0) side = 'left'
    else if (rect.top + rect.height + GAP + CARD_H < vh) side = 'bottom'
    else side = 'top'
  }

  let top = rect.top
  let left = rect.left
  if (side === 'right') {
    left = rect.left + rect.width + GAP
    top = rect.top
  } else if (side === 'left') {
    left = rect.left - CARD_W - GAP
    top = rect.top
  } else if (side === 'bottom') {
    left = rect.left + rect.width / 2 - CARD_W / 2
    top = rect.top + rect.height + GAP
  } else {
    left = rect.left + rect.width / 2 - CARD_W / 2
    top = rect.top - CARD_H - GAP
  }

  return {
    top: Math.max(GAP, Math.min(vh - CARD_H - GAP, top)),
    left: Math.max(GAP, Math.min(vw - CARD_W - GAP, left)),
    width: CARD_W,
  }
}

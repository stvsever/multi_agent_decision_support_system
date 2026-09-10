/**
 * The methodology film.
 *
 * A self-driving explanation of how a run answers one phenotype question, from
 * the shape of the question to the written report. One clock drives it: a
 * single animation frame loop advances a normalised time, acts occupy ranges of
 * that time, and every scene is a pure function of the act's progress. That is
 * what makes the rail scrubbable, and what makes a reduced-motion reader see a
 * correct still frame rather than a frozen transition.
 *
 * It reads `/capabilities` for its vocabulary and nothing else. No participant
 * is read and no run is made.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type JSX, type PointerEvent as ReactPointerEvent } from 'react'
import { motion } from 'framer-motion'
import { Pause, Play, RotateCcw } from 'lucide-react'
import { Badge, Button, Callout, Card, Skeleton } from '@/components/ui/primitives'
import { useCapabilities } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import {
  ACTS,
  ACT_STARTS,
  TOTAL_SECONDS,
  actIndexAt,
  actProgressAt,
  buildVocabulary,
  layoutFor,
} from './filmScript'
import {
  SceneCommunication,
  SceneEvaluation,
  SceneEvidence,
  SceneExecution,
  SceneIntegration,
  SceneOrchestration,
  ScenePrediction,
  SceneQuestion,
  StageHud,
  type SceneProps,
} from './filmScenes'
import './film.css'

const SCENES: ((props: SceneProps) => JSX.Element)[] = [
  SceneQuestion,
  SceneEvidence,
  SceneOrchestration,
  SceneExecution,
  SceneIntegration,
  ScenePrediction,
  SceneEvaluation,
  SceneCommunication,
]

/** The still frame a reduced-motion reader sees: the act, fully composed. */
const actEnd = (index: number): number => ACT_STARTS[index] + ACTS[index].seconds * 0.999

export function MethodologyFilm(): JSX.Element {
  const { data, isLoading, error } = useCapabilities()
  /* The store mirrors `data-motion` on the document root, so this is the same
     preference the stylesheet reads. */
  const reducedMotion = useApp((state) => state.appearance.reducedMotion)

  const stageRef = useRef<HTMLDivElement>(null)
  const width = useElementWidth(stageRef)
  const active = useStageActive(stageRef)

  /* Held on the composed first frame rather than at zero, because zero is the
     frame where nothing has entered yet: a reader arriving on a background tab,
     or scrolled past the stage, would otherwise meet an empty rectangle. The
     film rewinds to zero the moment it actually starts. */
  const [time, setTime] = useState(() => actEnd(0))
  const [playing, setPlaying] = useState(false)
  const timeRef = useRef(time)
  const started = useRef(reducedMotion)

  const seek = useCallback((next: number) => {
    const clamped = Math.max(0, Math.min(TOTAL_SECONDS, next))
    timeRef.current = clamped
    setTime(clamped)
  }, [])

  /* The first time the stage is genuinely on screen, rewind and roll. The
     document is re-read here rather than trusted from `active`, which is still
     the mount-time optimistic value on this pass. */
  useEffect(() => {
    if (started.current || !active || reducedMotion) return
    if (document.visibilityState === 'hidden') return
    started.current = true
    seek(0)
    setPlaying(true)
  }, [active, reducedMotion, seek])

  useEffect(() => {
    if (!playing || !active || reducedMotion) return
    let frame = 0
    let last = performance.now()
    const step = (now: number) => {
      // Clamped so a backgrounded tab cannot jump the film forward on return.
      const delta = Math.min(0.06, (now - last) / 1000)
      last = now
      const next = timeRef.current + delta
      if (next >= TOTAL_SECONDS) {
        timeRef.current = TOTAL_SECONDS
        setTime(TOTAL_SECONDS)
        setPlaying(false)
        return
      }
      timeRef.current = next
      setTime(next)
      frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [playing, active, reducedMotion])

  // Switching the preference on mid-film should land on a composed frame.
  useEffect(() => {
    if (!reducedMotion) return
    setPlaying(false)
    seek(actEnd(actIndexAt(timeRef.current)))
  }, [reducedMotion, seek])

  const index = actIndexAt(time)
  const act = ACTS[index]
  const progress = actProgressAt(time, index)
  const finished = time >= TOTAL_SECONDS

  const vocabulary = useMemo(() => (data ? buildVocabulary(data) : null), [data])
  const layout = useMemo(() => layoutFor(Math.max(320, Math.min(1500, width || 960))), [width])

  /* A jump while the film is running lands at the act's start so the reader
     watches it assemble. A jump while it is paused lands on the composed frame
     instead, because the start of an act is the frame where nothing has
     entered yet and a still of it says nothing. */
  const jump = useCallback(
    (next: number) => seek(reducedMotion || !playing ? actEnd(next) : ACT_STARTS[next]),
    [playing, reducedMotion, seek],
  )

  const toggle = useCallback(() => {
    if (!playing && timeRef.current >= TOTAL_SECONDS - 0.01) seek(0)
    setPlaying(!playing)
  }, [playing, seek])

  const restart = useCallback(() => {
    seek(reducedMotion ? actEnd(0) : 0)
    if (!reducedMotion) setPlaying(true)
  }, [reducedMotion, seek])

  const Scene = SCENES[index] ?? SCENES[0]
  const stageName = data?.stages?.[act.stage]

  return (
    <Card
      title="How a run answers one question"
      subtitle="An illustration of the method from end to end. No participant data is read and no model is called."
      actions={
        stageName ? (
          <Badge tone="accent">
            Stage {act.stage + 1} of {data?.stages.length ?? 0}, {stageName}
          </Badge>
        ) : undefined
      }
      flush
    >
      <div className="film">
        <div className="film__caption">
          <span className="eyebrow">
            Act {index + 1} of {ACTS.length}
          </span>
          <span className="t-h3">{act.title}</span>
        </div>

        <div className="film__stage" ref={stageRef}>
          {isLoading && <Skeleton height={layout.height} radius={10} />}

          {error && !data && (
            <div className="film__fallback">
              <Callout tone="caution" title="The engine description could not be read">
                The walkthrough names its stages, agents and tools from the running engine, so it stays blank until
                that call succeeds. The rest of this screen still works.
              </Callout>
            </div>
          )}

          {vocabulary && width > 0 && (
            <svg
              className="film__svg"
              viewBox={`0 0 ${layout.width} ${layout.height}`}
              width={layout.width}
              height={layout.height}
              aria-hidden="true"
            >
              <StageHud L={layout} act={index} p={progress} v={vocabulary} />
              <Scene p={progress} L={layout} v={vocabulary} />
            </svg>
          )}
        </div>

        {/* Keyed rather than wrapped in AnimatePresence: the line must be
            correct on the frame the act changes, including a scrub. */}
        <motion.p
          key={act.key}
          className="film__prose"
          initial={reducedMotion ? false : { opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reducedMotion ? 0 : 0.26, ease: [0.22, 1, 0.36, 1] }}
        >
          {act.prose}
        </motion.p>

        <Transport
          index={index}
          time={time}
          playing={playing}
          finished={finished}
          reducedMotion={reducedMotion}
          onToggle={toggle}
          onRestart={restart}
          onJump={jump}
          onScrub={seek}
        />
      </div>
    </Card>
  )
}

/* --- Transport ------------------------------------------------------------- */

function Transport({
  index,
  time,
  playing,
  finished,
  reducedMotion,
  onToggle,
  onRestart,
  onJump,
  onScrub,
}: {
  index: number
  time: number
  playing: boolean
  finished: boolean
  reducedMotion: boolean
  onToggle: () => void
  onRestart: () => void
  onJump: (act: number) => void
  onScrub: (time: number) => void
}) {
  const railRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const moved = useRef(false)

  const timeAt = (clientX: number): number => {
    const rect = railRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return 0
    return ((clientX - rect.left) / rect.width) * TOTAL_SECONDS
  }

  // The rail is linear in time because each act is sized by its own duration:
  // a press jumps to the act, a drag scrubs inside it.
  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (reducedMotion) return
    dragging.current = true
    moved.current = false
    try {
      railRef.current?.setPointerCapture(event.pointerId)
    } catch {
      // Not every pointer can be captured; the drag still tracks without it.
    }
    onJump(actIndexAt(timeAt(event.clientX)))
  }

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return
    moved.current = true
    onScrub(timeAt(event.clientX))
  }

  const endDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return
    dragging.current = false
    if (railRef.current?.hasPointerCapture(event.pointerId)) railRef.current.releasePointerCapture(event.pointerId)
  }

  return (
    <div className="film__transport">
      <div className="film__buttons">
        {!reducedMotion && (
          <Button
            size="sm"
            variant="secondary"
            iconOnly
            icon={playing ? <Pause size={13} /> : <Play size={13} />}
            onClick={onToggle}
            aria-label={playing ? 'Pause the walkthrough' : finished ? 'Play the walkthrough again' : 'Play the walkthrough'}
          />
        )}
        <Button
          size="sm"
          variant="ghost"
          iconOnly
          icon={<RotateCcw size={13} />}
          onClick={onRestart}
          aria-label="Back to the first act"
        />
      </div>

      <div
        className="film__rail"
        ref={railRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {ACTS.map((act, actIndex) => {
          const fill = Math.max(0, Math.min(1, (time - ACT_STARTS[actIndex]) / act.seconds))
          const current = actIndex === index
          return (
            <button
              key={act.key}
              type="button"
              className="film__rail-item"
              style={{ flexGrow: act.seconds }}
              data-current={current || undefined}
              aria-current={current ? 'step' : undefined}
              aria-label={`Act ${actIndex + 1}, ${act.title}`}
              onClick={() => {
                if (moved.current) return
                onJump(actIndex)
              }}
            >
              <span className="film__rail-track">
                <span className="film__rail-fill" style={{ transform: `scaleX(${fill})` }} />
              </span>
              <span className="film__rail-label">
                <span className="film__rail-index">{actIndex + 1}</span>
                <span className="truncate">{act.title}</span>
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/* --- Measurement ----------------------------------------------------------- */

/** Measured width, so the stage can lay itself out in real pixels. */
function useElementWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width ?? 0
      setWidth((prev) => (Math.abs(prev - next) < 1 ? prev : next))
    })
    observer.observe(element)
    setWidth(element.getBoundingClientRect().width)
    return () => observer.disconnect()
  }, [ref])
  return width
}

/** True only while the stage is on screen and the tab is in front. */
function useStageActive(ref: React.RefObject<HTMLElement | null>): boolean {
  const [visible, setVisible] = useState(true)
  const [inView, setInView] = useState(true)

  useEffect(() => {
    const handler = () => setVisible(document.visibilityState !== 'hidden')
    handler()
    document.addEventListener('visibilitychange', handler)
    return () => document.removeEventListener('visibilitychange', handler)
  }, [])

  useEffect(() => {
    const element = ref.current
    if (!element) return
    const observer = new IntersectionObserver((entries) => setInView(entries[0]?.isIntersecting ?? true), {
      threshold: 0.2,
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref])

  return visible && inView
}

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

import { useCallback, useEffect, useMemo, useRef, useState, type JSX } from 'react'
import { useReducedMotion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Pause, Play, RotateCcw } from 'lucide-react'
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
  FilmDefs,
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
  const { data, isLoading, error, refetch } = useCapabilities()
  /* The store mirrors `data-motion` on the document root, so this is the same
     preference the stylesheet reads. */
  const appReducedMotion = useApp((state) => state.appearance.reducedMotion)
  const systemReducedMotion = useReducedMotion()
  const reducedMotion = appReducedMotion || !!systemReducedMotion

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
    if (started.current || !active || reducedMotion || !data) return
    if (document.visibilityState === 'hidden') return
    started.current = true
    seek(0)
    setPlaying(true)
  }, [active, reducedMotion, seek, data])

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
  const scrollDiagram = index >= 1 && index <= 6
  const diagramWidth = scrollDiagram ? Math.max(960, width) : Math.max(280, width || 960)
  const layout = useMemo(() => layoutFor(diagramWidth, index), [diagramWidth, index])

  // A chapter selection pauses on its composed frame. Play then animates that
  // chapter from its beginning. Manual navigation always takes over autoplay.
  const jump = useCallback(
    (next: number) => {
      started.current = true
      setPlaying(false)
      seek(actEnd(Math.max(0, Math.min(ACTS.length - 1, next))))
    },
    [seek],
  )

  const toggle = useCallback(() => {
    started.current = true
    if (!playing) {
      const current = actIndexAt(timeRef.current)
      if (timeRef.current >= TOTAL_SECONDS - 0.01) seek(0)
      else if (actProgressAt(timeRef.current, current) > 0.98) seek(ACT_STARTS[current])
    }
    setPlaying(!playing)
  }, [playing, seek])

  const restart = useCallback(() => {
    started.current = true
    seek(reducedMotion ? actEnd(0) : 0)
    if (!reducedMotion) setPlaying(true)
  }, [reducedMotion, seek])


  const Scene = SCENES[index] ?? SCENES[0]
  const stageName = data?.stages?.[act.stage]

  return (
    <Card
      title="How the multi-agent inference system works"
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
      <div className="film" data-act={act.key} data-reduced={reducedMotion} data-playing={playing && active} data-scroll-diagram={scrollDiagram}>
        {/* The line about the act reads before the picture it describes, not
            after it. The act's own name is carried by the rail below, where it
            is already marked as the current one, so a heading here would say it
            twice. */}
        <p key={act.key} className="film__prose">{act.prose}</p>

        {scrollDiagram && width > 0 && width < 960 && <span className="film__pan-hint">Scroll horizontally to explore the diagram <ArrowRight size={12} /></span>}
        <div className="film__stage" ref={stageRef} tabIndex={scrollDiagram && width < 960 ? 0 : undefined}
          role={scrollDiagram && width < 960 ? 'region' : undefined} aria-label={scrollDiagram && width < 960 ? `Scrollable ${act.title} diagram` : undefined}>
          {isLoading && <Skeleton height={layout.height} radius={10} />}

          {error && !data && (
            <div className="film__fallback">
              <Callout tone="caution" title="The engine description could not be read">
                The walkthrough names its stages, agents and tools from the running engine, so it stays blank until
                that call succeeds. The rest of this screen still works.
                <Button size="sm" variant="secondary" onClick={() => void refetch()}>Retry</Button>
              </Callout>
            </div>
          )}

          {vocabulary && width > 0 && (
            <svg
              className="film__svg"
              role="img"
              aria-label={`${act.title}. ${act.prose}`}
              viewBox={`0 0 ${layout.width} ${layout.height}`}
              width={layout.width}
              height={layout.height}
            >
              <FilmDefs />
              <StageHud L={layout} act={index} p={progress} v={vocabulary} />
              <Scene p={progress} L={layout} v={vocabulary} />
            </svg>
          )}
        </div>

        <Transport
          index={index}
          time={time}
          playing={playing}
          finished={finished}
          reducedMotion={reducedMotion}
          onToggle={toggle}
          onRestart={restart}
          onJump={jump}
          onScrub={(next) => { started.current = true; setPlaying(false); seek(next) }}
          disabled={!vocabulary}
        />
      </div>
    </Card>
  )
}

/* Chapter controls and the timeline have separate hit areas. */
function Transport({ index, time, playing, finished, reducedMotion, onToggle, onRestart, onJump, onScrub, disabled }: {
  index: number; time: number; playing: boolean; finished: boolean; reducedMotion: boolean
  onToggle: () => void; onRestart: () => void; onJump: (act: number) => void; onScrub: (time: number) => void; disabled: boolean
}) {
  return <div className="film__transport">
    <div className="film__controls">
      <div className="film__buttons">
        {!reducedMotion && <Button size="sm" variant="secondary" disabled={disabled}
          icon={playing ? <Pause size={13} /> : <Play size={13} />} onClick={onToggle}
          aria-label={playing ? 'Pause the walkthrough' : finished ? 'Play the walkthrough again' : 'Play the walkthrough'}>
          {playing ? 'Pause' : finished ? 'Replay' : 'Play'}
        </Button>}
        <Button size="sm" variant="ghost" iconOnly icon={<RotateCcw size={13} />} onClick={onRestart} disabled={disabled} aria-label="Back to the first act" />
      </div>
      {!reducedMotion && <input className="film__scrubber" type="range" min={0} max={TOTAL_SECONDS} step={0.05}
        value={time} disabled={disabled} aria-label="Walkthrough position" aria-valuetext={`${ACTS[index].title}, ${Math.round(actProgressAt(time, index) * 100)} percent`}
        onChange={(event) => onScrub(Number(event.target.value))} />}
      <div className="film__buttons">
        <Button size="sm" variant="ghost" iconOnly icon={<ArrowLeft size={14} />} disabled={disabled || index === 0} onClick={() => onJump(index - 1)} aria-label="Previous act" />
        <Button size="sm" variant="ghost" iconOnly icon={<ArrowRight size={14} />} disabled={disabled || index === ACTS.length - 1} onClick={() => onJump(index + 1)} aria-label="Next act" />
      </div>
    </div>
    <nav className="film__rail" aria-label="Methodology walkthrough">
      {ACTS.map((act, actIndex) => {
        const fill = Math.max(0, Math.min(1, (time - ACT_STARTS[actIndex]) / act.seconds))
        const current = actIndex === index
        return <button key={act.key} type="button" className="film__rail-item" disabled={disabled}
          data-current={current || undefined} aria-current={current ? 'step' : undefined}
          aria-label={`Act ${actIndex + 1}, ${act.title}`} onClick={() => onJump(actIndex)}>
          <span className="film__rail-label"><span className="film__rail-index">{actIndex + 1}</span><span>{act.title}</span></span>
          <span className="film__rail-track"><span className="film__rail-fill" style={{ transform: `scaleX(${fill})` }} /></span>
        </button>
      })}
    </nav>
  </div>
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

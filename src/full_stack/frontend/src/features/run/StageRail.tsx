/** The seven pipeline stages with the time each one has consumed. */

import clsx from 'clsx'
import { Check, X } from 'lucide-react'
import { memo, useMemo } from 'react'
import { Badge } from '@/components/ui/primitives'
import { duration as formatDuration } from '@/lib/format'
import { useNow } from '@/lib/hooks'
import type { RunEvent } from '@/lib/types'
import { stageTimings } from './runUtils'

export interface StageRailProps {
  stages: string[]
  currentStage: number
  events: RunEvent[]
  running: boolean
  /** Marks the stage the run stopped in rather than leaving it looking current. */
  failed?: boolean
  iteration: number
  maxIterations?: number
  /** ISO start of the run, so Initialization is timed from the run itself. */
  startedAt?: string | null
}

export const StageRail = memo(function StageRail({
  stages,
  currentStage,
  events,
  running,
  failed = false,
  iteration,
  maxIterations,
  startedAt,
}: StageRailProps) {
  const now = useNow(running)
  const startedMs = useMemo(() => {
    if (!startedAt) return null
    const parsed = Date.parse(startedAt)
    return Number.isNaN(parsed) ? null : parsed
  }, [startedAt])
  const timings = useMemo(
    () => stageTimings(events, stages.length, running ? now : null, startedMs),
    [events, stages.length, running, now, startedMs],
  )

  const total = timings.totals.reduce((sum, value) => sum + value, 0)
  const timed = timings.totals.filter((value) => value > 0).length
  // With a single timed stage there is nothing to be slower than.
  const slowest =
    timed > 1 ? timings.totals.reduce((best, value, index) => (value > timings.totals[best] ? index : best), 0) : -1

  return (
    <section className="run-rail" aria-label="Pipeline stages">
      <ol className="run-rail__track">
        {stages.map((stage, index) => {
          const seconds = timings.totals[index] ?? 0
          const isCurrent = index === currentStage
          const isDone = currentStage > index || (!running && timings.visited[index] && !isCurrent)
          const share = total > 0 ? seconds / total : 0
          return (
            <li
              key={stage}
              className={clsx(
                'run-stage',
                isCurrent && !failed && 'run-stage--current',
                isCurrent && failed && 'run-stage--failed',
                isDone && 'run-stage--done',
                !isCurrent && !isDone && 'run-stage--pending',
              )}
            >
              <span className="run-stage__marker" aria-hidden>
                {isDone ? (
                  <Check size={12} />
                ) : isCurrent && failed ? (
                  <X size={12} />
                ) : (
                  <span className="run-stage__index">{index + 1}</span>
                )}
              </span>
              <span className="run-stage__body">
                <span className="run-stage__name truncate">{stage}</span>
                <span className="run-stage__time tabular">
                  {seconds > 0
                    ? formatDuration(seconds)
                    : isCurrent
                      ? failed
                        ? 'stopped here'
                        : 'starting'
                      : '-'}
                  {seconds > 0 && index === slowest && <span className="run-stage__flag"> slowest</span>}
                </span>
                <span className="run-stage__bar" aria-hidden>
                  <span className="run-stage__bar-fill" style={{ width: `${Math.round(share * 100)}%` }} />
                </span>
              </span>
            </li>
          )
        })}
      </ol>
      <div className="run-rail__aside">
        <Badge tone={running ? 'accent' : 'neutral'}>
          {maxIterations && maxIterations > 1 ? `Iteration ${iteration} of ${maxIterations}` : `Iteration ${iteration}`}
        </Badge>
        <span className="t-tiny muted tabular">{total > 0 ? `${formatDuration(total)} tracked` : 'no timing yet'}</span>
      </div>
    </section>
  )
})

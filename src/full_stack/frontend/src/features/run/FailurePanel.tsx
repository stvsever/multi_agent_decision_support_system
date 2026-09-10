/**
 * What a failed run tells the reader.
 *
 * The worker reports an exception and a stack. The stack answers "where in the
 * code", which is rarely the first question, so the plain facts lead and the
 * trace stays one click away.
 */

import { AlertTriangle } from 'lucide-react'
import { memo, useMemo } from 'react'
import { Badge, CopyButton, Disclosure } from '@/components/ui/primitives'
import type { RunDetail } from '@/lib/types'
import { describeFailure } from './runUtils'

export const FailurePanel = memo(function FailurePanel({ detail }: { detail: RunDetail }) {
  const failure = useMemo(() => describeFailure(detail), [detail])
  const stepError = failure.step?.error && failure.step.error !== detail.error ? failure.step.error : ''

  return (
    <section className="run-failure" aria-label="Failure">
      <div className="run-failure__head">
        <span className="run-failure__icon" aria-hidden>
          <AlertTriangle size={16} />
        </span>
        <div className="stack gap-2 grow">
          <span className="t-lead semibold">{failure.lead}</span>
          <p className="run-failure__message">{failure.message}</p>
          <div className="row gap-2 wrap">
            {failure.exception && (
              <Badge tone="critical" mono>
                {failure.exception}
              </Badge>
            )}
            {failure.stage && <Badge outline>{failure.stage}</Badge>}
            {failure.step && (
              <Badge outline mono>
                step #{failure.step.id} {failure.step.tool}
              </Badge>
            )}
          </div>
          {failure.hint && <p className="t-small muted" style={{ maxWidth: '68ch' }}>{failure.hint}</p>}
        </div>
      </div>

      {stepError && (
        <div className="run-block">
          <span className="run-block__label">The step that failed reported</span>
          <pre className="run-pre run-pre--critical">{stepError}</pre>
        </div>
      )}

      {detail.traceback && (
        <Disclosure title="Traceback" subtitle="The worker stack trace, for a bug report">
          <div className="stack gap-2">
            <div className="row">
              <div className="grow" />
              <CopyButton text={detail.traceback} label="Copy traceback" />
            </div>
            <pre className="run-pre run-pre--tall run-pre--critical">{detail.traceback}</pre>
          </div>
        </Disclosure>
      )}
    </section>
  )
})

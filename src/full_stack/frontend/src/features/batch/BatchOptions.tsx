/**
 * The controls that turn a selection into a queue.
 *
 * Configure shows these inline as soon as more than one participant is picked,
 * and the batch console shows the same component, so there is one definition of
 * what a batch launch does rather than two that drift.
 */

import { useState } from 'react'
import { Button, Field, Select, SliderField, Toggle } from '@/components/ui/primitives'
import { ApiError } from '@/lib/api'
import { number } from '@/lib/format'
import { useConfigPatch, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { describeQueue, TIMEOUT_CHOICES, timeoutLabel } from './util'

export interface BatchSettings {
  concurrency: number
  continueOnError: boolean
}

export function BatchOptions({
  value,
  onChange,
  participantCount,
  deepReport,
  onDeepReportChange,
}: {
  value: BatchSettings
  onChange: (next: BatchSettings) => void
  participantCount: number
  /** Omitted where the page already carries the report toggle elsewhere. */
  deepReport?: boolean
  onDeepReportChange?: (next: boolean) => void
}) {
  const { data: settings } = useSettings()
  const patch = useConfigPatch()
  const openSettings = useApp((state) => state.openSettings)
  const notify = useApp((state) => state.notify)
  // The worker count is a per-run override on the same screen as this panel,
  // so read it from the store rather than the saved config: quoting the saved
  // number here would contradict the slider a few centimetres away.
  const overrides = useApp((state) => state.overrides)
  const [saving, setSaving] = useState(false)

  const config = settings?.config
  const savedWorkers = config?.engine.executor_max_workers ?? 1
  const workerOverride = overrides.engine?.executor_max_workers as number | undefined
  const workers = workerOverride ?? savedWorkers
  const timeout = config?.batch.run_timeout_seconds ?? 3600
  const savedConcurrency = config?.batch.concurrency ?? 2
  const choices = TIMEOUT_CHOICES.includes(timeout) ? TIMEOUT_CHOICES : [...TIMEOUT_CHOICES, timeout].sort((a, b) => a - b)

  const saveTimeout = async (seconds: number) => {
    setSaving(true)
    try {
      await patch({ batch: { run_timeout_seconds: seconds } })
    } catch (error) {
      // The select reads from the saved config, so a rejected patch leaves it
      // on the last known-good value on its own. Only the telling is missing.
      notify({
        tone: 'critical',
        title: 'The run timeout was not saved',
        body: error instanceof ApiError ? error.message : String(error),
      })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="stack gap-5">
      <Field
        label="Participants in parallel"
        hint={`About ${number(value.concurrency * workers)} model calls in flight across the batch.${
          value.concurrency === savedConcurrency ? '' : ` The saved default is ${savedConcurrency}.`
        }`}
        info={
          <>
            <p>
              This sets how many participants run at the same time. It is not the only source of parallelism:
              inside each participant the executor fans its plan steps out across up to {number(workers)} workers.
            </p>
            <p>
              What the provider sees is roughly the product of the two, so {value.concurrency} participants times{' '}
              {number(workers)} workers is about {number(value.concurrency * workers)} requests in flight. Raise it
              only if the rate limit has room.
            </p>
            {workerOverride !== undefined && (
              <p>
                This run overrides the worker count: {number(workerOverride)} in place of the saved{' '}
                {number(savedWorkers)}. The figures above already use the override.
              </p>
            )}
          </>
        }
      >
        <SliderField
          value={value.concurrency}
          min={1}
          max={16}
          onChange={(concurrency) => onChange({ ...value, concurrency })}
        />
      </Field>

      <div className="row gap-3 between">
        <div className="stack" style={{ gap: 1 }}>
          <span className="t-small semibold">Continue on error</span>
          <span className="t-tiny muted">
            {value.continueOnError
              ? 'A failed participant does not stop the rest.'
              : 'The first failure cancels everything still queued.'}
          </span>
        </div>
        <Toggle
          checked={value.continueOnError}
          onChange={(continueOnError) => onChange({ ...value, continueOnError })}
          label="Continue on error"
        />
      </div>

      {onDeepReportChange && (
        <div className="row gap-3 between">
          <div className="stack" style={{ gap: 1 }}>
            <span className="t-small semibold">Deep phenotype report</span>
            <span className="t-tiny muted">
              {deepReport
                ? 'The Communicator writes a full report for each participant.'
                : 'Skipped, which removes the largest single cost per participant.'}
            </span>
          </div>
          <Toggle checked={Boolean(deepReport)} onChange={onDeepReportChange} label="Deep phenotype report" />
        </div>
      )}

      <Field
        label="Run timeout"
        hint={
          saving
            ? 'Saving.'
            : 'A saved setting: it applies to every run, here and elsewhere, and is stored as soon as you change it.'
        }
        info={
          <p>
            A participant that has not finished within this window is stopped and recorded as failed, so one stuck
            run cannot hold a cohort open indefinitely.
          </p>
        }
      >
        <Select
          value={String(timeout)}
          disabled={saving || !config}
          onChange={(event) => void saveTimeout(Number(event.target.value))}
        >
          {choices.map((seconds) => (
            <option key={seconds} value={seconds}>
              {timeoutLabel(seconds)}
            </option>
          ))}
        </Select>
      </Field>

      <div className="row between gap-3 wrap">
        <p className="t-small secondary grow" style={{ margin: 0 }}>
          {describeQueue(participantCount, value.concurrency, value.continueOnError, timeout)}
        </p>
        <Button size="sm" variant="ghost" onClick={() => openSettings('engine')} style={{ flex: 'none' }}>
          Batch defaults
        </Button>
      </div>
    </div>
  )
}

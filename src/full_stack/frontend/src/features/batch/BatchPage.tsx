/**
 * Batch console.
 *
 * Two views over the same cohort: compose one, then watch it. Launching moves
 * the screen to the monitor with the new batch already selected, because the
 * next thing anyone wants after pressing launch is to see it start.
 */

import { useState, type JSX } from 'react'
import { Badge, Tabs } from '@/components/ui/primitives'
import { useBatches } from '@/lib/hooks'
import { BatchComposer } from './BatchComposer'
import { BatchMonitor } from './BatchMonitor'
import './batch.css'

type View = 'compose' | 'monitor'

export function BatchPage(): JSX.Element {
  const [view, setView] = useState<View>('compose')
  const [activeBatch, setActiveBatch] = useState<string | null>(null)
  const { data } = useBatches(true)

  const batches = data?.batches ?? []
  const running = batches.filter((b) => b.status === 'running').length

  const launched = (batchId: string) => {
    setActiveBatch(batchId)
    setView('monitor')
  }

  return (
    <div className="page">
      <div className="page__head">
        <div className="stack gap-2">
          <h1 className="page__title">Batch</h1>
          <p className="page__lede">
            One task across a cohort. Each participant becomes its own run with its own report, and the batch tracks the
            cohort as a whole.
          </p>
        </div>
      </div>

      <Tabs
        value={view}
        onChange={setView}
        options={[
          { value: 'compose', label: 'Compose' },
          {
            value: 'monitor',
            label: 'Monitor',
            badge:
              running > 0 ? (
                <Badge tone="accent">{running} running</Badge>
              ) : batches.length > 0 ? (
                <Badge>{batches.length}</Badge>
              ) : undefined,
          },
        ]}
      />

      {view === 'compose' ? (
        <BatchComposer onLaunched={launched} />
      ) : (
        <BatchMonitor activeId={activeBatch} onSelect={setActiveBatch} />
      )}
    </div>
  )
}

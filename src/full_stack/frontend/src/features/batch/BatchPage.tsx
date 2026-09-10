/**
 * Batch console.
 *
 * Composing a cohort now happens in Configure, which is where the participants
 * and the task already are. This screen is where a launched batch is watched,
 * and it opens on the monitor because that is what someone arriving from a
 * launch came to see. The composer stays available for anyone who lands here
 * directly.
 */

import { useEffect, useState, type JSX } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Badge, Tabs } from '@/components/ui/primitives'
import { useBatches } from '@/lib/hooks'
import { BatchComposer } from './BatchComposer'
import { BatchMonitor } from './BatchMonitor'
import './batch.css'

type View = 'monitor' | 'compose'

export function BatchPage(): JSX.Element {
  const location = useLocation()
  const arrivedWith = (location.state as { batchId?: string } | null)?.batchId ?? null

  const [view, setView] = useState<View>('monitor')
  const [activeBatch, setActiveBatch] = useState<string | null>(arrivedWith)
  const { data } = useBatches(true)

  useEffect(() => {
    if (!arrivedWith) return
    setActiveBatch(arrivedWith)
    setView('monitor')
  }, [arrivedWith])

  const batches = data?.batches ?? []
  const running = batches.filter((batch) => batch.status === 'running').length

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
            One task across a cohort. Each participant becomes its own run with its own report, and the batch tracks
            the cohort as a whole. To start one, select more than one participant in{' '}
            <Link to="/studio">Configure</Link> and the batch controls appear there.
          </p>
        </div>
      </div>

      <Tabs
        spread
        value={view}
        onChange={setView}
        options={[
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
          { value: 'compose', label: 'Compose here instead' },
        ]}
      />

      {view === 'monitor' ? (
        <BatchMonitor activeId={activeBatch} onSelect={setActiveBatch} onCompose={() => setView('compose')} />
      ) : (
        <BatchComposer onLaunched={launched} />
      )}
    </div>
  )
}

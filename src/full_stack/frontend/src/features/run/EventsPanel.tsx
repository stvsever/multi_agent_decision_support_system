/** The raw event feed, oldest first, following the tail unless the reader scrolls away. */

import clsx from 'clsx'
import { ArrowDown } from 'lucide-react'
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Badge, Button, EmptyState } from '@/components/ui/primitives'
import type { RunEvent } from '@/lib/types'
import { eventTone, summariseEvent } from './runUtils'

export interface EventsPanelProps {
  events: RunEvent[]
}

const BOTTOM_SLACK = 56

export const EventsPanel = memo(function EventsPanel({ events }: EventsPanelProps) {
  const [types, setTypes] = useState<string[]>([])
  const [pinned, setPinned] = useState(true)
  const scroller = useRef<HTMLDivElement>(null)

  const present = useMemo(() => {
    const seen: string[] = []
    for (const event of events) if (!seen.includes(event.type)) seen.push(event.type)
    return seen.sort()
  }, [events])

  const visible = useMemo(
    () => (types.length === 0 ? events : events.filter((event) => types.includes(event.type))),
    [events, types],
  )

  // Following the tail is the default; scrolling up releases it until the reader
  // returns to the bottom or presses the jump control.
  useEffect(() => {
    if (!pinned) return
    const node = scroller.current
    if (node) node.scrollTop = node.scrollHeight
  }, [visible.length, pinned])

  const onScroll = useCallback(() => {
    const node = scroller.current
    if (!node) return
    const distance = node.scrollHeight - node.scrollTop - node.clientHeight
    setPinned(distance <= BOTTOM_SLACK)
  }, [])

  const jump = useCallback(() => {
    const node = scroller.current
    if (node) node.scrollTop = node.scrollHeight
    setPinned(true)
  }, [])

  const toggleType = (type: string) =>
    setTypes((prev) => (prev.includes(type) ? prev.filter((item) => item !== type) : [...prev, type]))

  return (
    <div className="run-events">
      <div className="run-events__filters">
        <button type="button" className="run-chip" aria-pressed={types.length === 0} onClick={() => setTypes([])}>
          All
        </button>
        {present.map((type) => (
          <button
            key={type}
            type="button"
            className="run-chip run-chip--mono"
            aria-pressed={types.includes(type)}
            onClick={() => toggleType(type)}
          >
            {type}
          </button>
        ))}
      </div>

      <div className="run-events__scroll" ref={scroller} onScroll={onScroll}>
        {visible.length === 0 ? (
          <EmptyState title="No events" body="Events arrive as soon as the engine starts reporting." />
        ) : (
          <ol className="run-events__list">
            {visible.map((event) => (
              <li key={event.id} className="run-event">
                <span className="run-event__time mono tabular">{event.time}</span>
                <Badge tone={eventTone(event.type)} mono>
                  {event.type}
                </Badge>
                <span className="run-event__summary t-tiny secondary">{summariseEvent(event)}</span>
              </li>
            ))}
          </ol>
        )}
      </div>

      <div className={clsx('run-events__jump', pinned && 'run-events__jump--hidden')}>
        <Button size="sm" variant="secondary" icon={<ArrowDown size={13} />} onClick={jump}>
          Jump to latest
        </Button>
      </div>
    </div>
  )
})

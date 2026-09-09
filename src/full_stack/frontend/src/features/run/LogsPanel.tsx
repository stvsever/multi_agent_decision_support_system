/** Worker stdout and stderr, filterable and copyable. */

import { Search } from 'lucide-react'
import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { CopyButton, EmptyState, Input } from '@/components/ui/primitives'

export interface LogsPanelProps {
  logs: string[]
}

export const LogsPanel = memo(function LogsPanel({ logs }: LogsPanelProps) {
  const [filter, setFilter] = useState('')
  const scroller = useRef<HTMLPreElement>(null)

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    return needle ? logs.filter((line) => line.toLowerCase().includes(needle)) : logs
  }, [logs, filter])

  useEffect(() => {
    const node = scroller.current
    if (node) node.scrollTop = node.scrollHeight
  }, [visible.length])

  return (
    <div className="run-logspanel">
      <div className="row gap-2">
        <div className="run-search grow">
          <Search size={14} className="run-search__icon" />
          <Input
            value={filter}
            placeholder="Filter lines"
            aria-label="Filter log lines"
            onChange={(event) => setFilter(event.target.value)}
          />
        </div>
        <span className="t-tiny muted tabular">
          {visible.length} of {logs.length}
        </span>
        <CopyButton text={visible.join('\n')} label="Copy" />
      </div>

      {logs.length === 0 ? (
        <EmptyState title="No log output" body="The worker writes here as it runs. A quiet run is a healthy one." />
      ) : (
        <pre className="run-logs" ref={scroller}>
          {visible.join('\n')}
        </pre>
      )}
    </div>
  )
})

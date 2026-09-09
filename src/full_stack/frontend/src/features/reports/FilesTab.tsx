/** The artifact list and an inline viewer for whichever file is selected. */

import { useMemo, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, FileJson, FileText } from 'lucide-react'
import { Badge, Callout, CopyButton, Card, EmptyState, Skeleton } from '@/components/ui/primitives'
import { Markdown } from '@/components/ui/Markdown'
import { api } from '@/lib/api'
import { bytes } from '@/lib/format'
import type { Artifact } from '@/lib/types'

export type FileParams = { run_id?: string; participant_dir?: string }

const JSON_TOKEN = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g
const HIGHLIGHT_LIMIT = 400_000

/** Minimal JSON colouring, built as elements so nothing is injected as markup. */
function highlightJson(text: string): ReactNode[] {
  if (text.length > HIGHLIGHT_LIMIT) return [text]
  const out: ReactNode[] = []
  let last = 0
  let key = 0
  JSON_TOKEN.lastIndex = 0

  let match: RegExpExecArray | null
  while ((match = JSON_TOKEN.exec(text)) !== null) {
    if (match.index > last) {
      out.push(
        <span key={`p${key}`} className="reports__json-punct">
          {text.slice(last, match.index)}
        </span>,
      )
      key += 1
    }
    if (match[1] !== undefined) {
      const isKey = match[2] !== undefined
      out.push(
        <span key={`t${key}`} className={isKey ? 'reports__json-key' : 'reports__json-string'}>
          {match[1]}
        </span>,
      )
      key += 1
      if (isKey) {
        out.push(
          <span key={`c${key}`} className="reports__json-punct">
            {match[2]}
          </span>,
        )
        key += 1
      }
    } else if (match[3] !== undefined) {
      out.push(
        <span key={`a${key}`} className="reports__json-atom">
          {match[3]}
        </span>,
      )
      key += 1
    } else if (match[4] !== undefined) {
      out.push(
        <span key={`n${key}`} className="reports__json-number">
          {match[4]}
        </span>,
      )
      key += 1
    }
    last = match.index + match[0].length
  }
  if (last < text.length) {
    out.push(
      <span key="tail" className="reports__json-punct">
        {text.slice(last)}
      </span>,
    )
  }
  return out
}

function JsonViewer({ text }: { text: string }) {
  const pretty = useMemo(() => {
    try {
      return JSON.stringify(JSON.parse(text), null, 2)
    } catch {
      return text
    }
  }, [text])
  const nodes = useMemo(() => highlightJson(pretty), [pretty])
  return <div className="reports__viewer">{nodes}</div>
}

export function FilesTab({ artifacts, params }: { artifacts: Artifact[]; params: FileParams }) {
  const [selected, setSelected] = useState<string | null>(null)
  const artifact = artifacts.find((entry) => entry.key === selected) ?? null

  const file = useQuery({
    queryKey: ['report-file', params, selected],
    queryFn: () => api.reports.file({ key: selected as string, ...params }),
    enabled: Boolean(selected && artifact?.exists),
    staleTime: 60_000,
    retry: false,
  })

  if (artifacts.length === 0) {
    return <EmptyState title="No artifacts" body="This output directory holds no files the dashboard knows how to read." />
  }

  const present = artifacts.filter((entry) => entry.exists).length

  return (
    <>
      <Card title="Files" subtitle={`${present} of ${artifacts.length} artifacts present on disk`}>
        <div className="stack gap-2">
          {artifacts.map((entry) => (
            <button
              key={entry.key}
              type="button"
              className="reports__file"
              aria-current={entry.key === selected}
              disabled={!entry.exists}
              onClick={() => setSelected(entry.key === selected ? null : entry.key)}
            >
              <span className="row gap-3" style={{ minWidth: 0 }}>
                {entry.format === 'json' ? <FileJson size={15} /> : <FileText size={15} />}
                <span className="stack" style={{ gap: 1, minWidth: 0 }}>
                  <span className="t-small semibold truncate">{entry.label}</span>
                  <span className="t-micro muted mono truncate">{entry.file}</span>
                </span>
              </span>
              <span className="row gap-2" style={{ flex: 'none' }}>
                {entry.exists ? (
                  <span className="t-tiny muted tabular">{bytes(entry.size)}</span>
                ) : (
                  <Badge tone="neutral">not generated</Badge>
                )}
              </span>
            </button>
          ))}
        </div>
      </Card>

      {artifact && artifact.exists && (
        <Card
          title={artifact.label}
          subtitle={artifact.path}
          actions={file.data ? <CopyButton text={file.data} label="Copy file" /> : undefined}
        >
          {file.isLoading && (
            <div className="stack gap-2">
              <Skeleton height={12} />
              <Skeleton height={12} width="86%" />
              <Skeleton height={12} width="64%" />
            </div>
          )}
          {file.isError && (
            <Callout tone="critical" icon={<AlertTriangle size={15} />} title="The file could not be read">
              {file.error instanceof Error ? file.error.message : 'The service refused the request.'}
            </Callout>
          )}
          {file.data !== undefined && !file.isLoading ? (
            artifact.format === 'markdown' ? (
              <div className="reports__reading">
                <Markdown source={file.data} />
              </div>
            ) : (
              <JsonViewer text={file.data} />
            )
          ) : null}
        </Card>
      )}
    </>
  )
}

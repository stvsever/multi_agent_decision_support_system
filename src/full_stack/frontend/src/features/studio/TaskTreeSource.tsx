/**
 * Getting a tree in and out.
 *
 * Most people already have the shape of their question written down somewhere.
 * A file drop, a file picker, and a paste box all reach the same reader, and
 * the export writes the shape the reader accepts, so the format can be learned
 * by exporting an example and editing it.
 */

import clsx from 'clsx'
import { AlertTriangle, Check, Download, FileJson, Upload, X } from 'lucide-react'
import { useRef, useState } from 'react'
import { Badge, Button, Callout, Disclosure, Textarea } from '@/components/ui/primitives'
import type { TaskNodeInput } from '@/lib/types'
import { TASK_TREE_EXAMPLES, cloneExample, type TaskTreeExample } from './taskExamples'
import { downloadJson, parseTaskTree, serialiseTaskTree, type ImportIssue } from './taskTreeIO'
import { MODE_LABELS, countNodes, walkNodes } from './taskValidation'

const MAX_SHOWN_ISSUES = 8

export function TaskTreeSource({
  root,
  onApply,
}: {
  root: TaskNodeInput | null
  onApply: (root: TaskNodeInput, note: string) => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [issues, setIssues] = useState<ImportIssue[]>([])
  const [applied, setApplied] = useState<string>('')
  const [dragging, setDragging] = useState(false)
  const [pasting, setPasting] = useState(false)
  const [draft, setDraft] = useState('')

  const accept = (text: string, origin: string) => {
    const result = parseTaskTree(text)
    if (!result.ok) {
      setIssues(result.issues)
      setApplied('')
      return
    }
    setIssues([])
    setApplied(`${result.nodeCount} node${result.nodeCount === 1 ? '' : 's'} from ${origin}`)
    setPasting(false)
    setDraft('')
    onApply(result.root, origin)
  }

  const readFile = async (file: File) => {
    const named = file.name.toLowerCase()
    if (!named.endsWith('.json') && file.type !== 'application/json') {
      setIssues([{ path: file.name, message: 'Expected a .json file. This one is something else.' }])
      setApplied('')
      return
    }
    try {
      accept(await file.text(), file.name)
    } catch {
      setIssues([{ path: file.name, message: 'The file could not be read from disk.' }])
      setApplied('')
    }
  }

  return (
    <div
      className={clsx('tio', dragging && 'tio--over')}
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(event) => {
        if (event.currentTarget.contains(event.relatedTarget as Node)) return
        setDragging(false)
      }}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        const file = event.dataTransfer.files?.[0]
        if (file) {
          void readFile(file)
          return
        }
        const text = event.dataTransfer.getData('text')
        if (text) accept(text, 'the dropped text')
      }}
    >
      <div className="row gap-2 wrap between">
        <span className="row gap-2">
          <FileJson size={15} className="faint" />
          <span className="t-small semibold">Drop a .json task tree here</span>
        </span>
        <span className="row gap-2 wrap">
          <Button size="sm" icon={<Upload size={13} />} onClick={() => fileRef.current?.click()}>
            Choose a file
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setPasting((was) => !was)}>
            {pasting ? 'Close paste box' : 'Paste JSON'}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            icon={<Download size={13} />}
            disabled={!root}
            onClick={() => root && downloadJson('compass_task_tree.json', serialiseTaskTree(root))}
          >
            Export
          </Button>
        </span>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="application/json,.json"
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) void readFile(file)
          event.target.value = ''
        }}
      />

      {pasting && (
        <div className="stack gap-2">
          <Textarea
            mono
            rows={7}
            autoFocus
            value={draft}
            placeholder='{ "prediction_type": "hierarchical", "root": { "node_id": "root", ... } }'
            onChange={(event) => setDraft(event.target.value)}
          />
          <div className="row gap-2">
            <Button size="sm" variant="primary" disabled={!draft.trim()} onClick={() => accept(draft, 'the paste box')}>
              Apply
            </Button>
            <Button size="sm" variant="ghost" onClick={() => { setPasting(false); setDraft('') }}>
              Cancel
            </Button>
          </div>
        </div>
      )}

      {issues.length > 0 && (
        <Callout
          tone="critical"
          icon={<AlertTriangle size={15} />}
          title={`That file was not applied: ${issues.length} problem${issues.length === 1 ? '' : 's'}`}
          action={
            <Button size="sm" variant="ghost" iconOnly icon={<X size={13} />} aria-label="Dismiss" onClick={() => setIssues([])} />
          }
        >
          <ul className="tio__issues">
            {issues.slice(0, MAX_SHOWN_ISSUES).map((issue, index) => (
              <li key={`${issue.path}-${index}`}>
                <span className="mono t-micro">{issue.path}</span>
                <span>{issue.message}</span>
              </li>
            ))}
            {issues.length > MAX_SHOWN_ISSUES && (
              <li className="muted">and {issues.length - MAX_SHOWN_ISSUES} more</li>
            )}
          </ul>
        </Callout>
      )}

      {applied && issues.length === 0 && (
        <span className="row gap-2 t-tiny" style={{ color: 'var(--positive)' }}>
          <Check size={13} />
          Applied {applied}
        </span>
      )}

      <Disclosure
        title="Start from an example"
        subtitle="Three shapes phenotyping questions usually take"
        defaultOpen={!root}
      >
        <div className="tio__examples">
          {TASK_TREE_EXAMPLES.map((example) => (
            <ExampleCard
              key={example.id}
              example={example}
              onUse={() => {
                setIssues([])
                setApplied(`${countNodes(example.root)} nodes from the ${example.title.toLowerCase()} example`)
                onApply(cloneExample(example), example.title)
              }}
            />
          ))}
        </div>
      </Disclosure>
    </div>
  )
}

function ExampleCard({ example, onUse }: { example: TaskTreeExample; onUse: () => void }) {
  return (
    <div className="tio__example">
      <div className="stack gap-1">
        <span className="t-small semibold">{example.title}</span>
        <span className="t-tiny muted">{example.summary}</span>
      </div>
      <TreePreview root={example.root} />
      <Button size="sm" block onClick={onUse}>
        Use this tree
      </Button>
    </div>
  )
}

/** The structure, before anything is applied. */
export function TreePreview({ root }: { root: TaskNodeInput }) {
  const rows: { key: string; node: TaskNodeInput; depth: number }[] = []
  walkNodes(root, (node, key, depth) => rows.push({ key, node, depth }))
  return (
    <div className="tio__preview">
      {rows.map(({ key, node, depth }) => (
        <div key={key} className="tio__preview-row" style={{ paddingLeft: depth * 14 }}>
          <span className="t-tiny truncate grow">
            {depth > 0 && <span className="faint">└ </span>}
            {node.display_name}
          </span>
          <Badge outline>{MODE_LABELS[node.mode]}</Badge>
        </div>
      ))}
    </div>
  )
}

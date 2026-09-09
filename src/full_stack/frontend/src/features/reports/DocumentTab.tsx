/** A generated markdown document in a reading column, with its own contents list. */

import { useMemo } from 'react'
import type { ReactNode } from 'react'
import { EmptyState } from '@/components/ui/primitives'
import { Markdown, extractHeadings } from '@/components/ui/Markdown'

export function DocumentTab({
  source,
  emptyIcon,
  emptyTitle,
  emptyBody,
}: {
  source: string
  emptyIcon?: ReactNode
  emptyTitle: string
  emptyBody: ReactNode
}) {
  const text = source ?? ''
  const headings = useMemo(() => extractHeadings(text), [text])

  if (!text.trim()) {
    return <EmptyState icon={emptyIcon} title={emptyTitle} body={emptyBody} />
  }

  return (
    <div className="reports__doc">
      <article className="reports__reading">
        <Markdown source={text} />
      </article>
      {headings.length > 1 && (
        <nav className="reports__toc reports__no-print" aria-label="Contents">
          <span className="reports__toc-title">Contents</span>
          {headings.map((heading) => (
            <button
              key={heading.id}
              type="button"
              className="reports__toc-link truncate"
              data-level={heading.level}
              title={heading.text}
              onClick={() =>
                document.getElementById(heading.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
              }
            >
              {heading.text}
            </button>
          ))}
        </nav>
      )}
    </div>
  )
}

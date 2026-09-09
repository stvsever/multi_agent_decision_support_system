/**
 * Dependency-free renderer for the GitHub-flavoured markdown subset the engine
 * emits.
 *
 * Everything becomes React elements. No source string ever reaches
 * dangerouslySetInnerHTML, so a report cannot inject markup into the dashboard.
 * Headings carry stable ids so a table of contents can link into the document.
 */

import { useMemo, type JSX, type ReactNode } from 'react'

export interface MarkdownHeading {
  id: string
  level: number
  text: string
}

type Align = 'left' | 'center' | 'right'

interface ListItem {
  text: string
  children: { ordered: boolean; items: string[] } | null
}

type Block =
  | { kind: 'heading'; level: number; id: string; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'list'; ordered: boolean; start: number; items: ListItem[] }
  | { kind: 'code'; lang: string; code: string }
  | { kind: 'quote'; blocks: Block[] }
  | { kind: 'rule' }
  | { kind: 'table'; head: string[]; align: Align[]; rows: string[][] }

interface Parsed {
  blocks: Block[]
  headings: MarkdownHeading[]
}

/* --- Block grammar -------------------------------------------------------- */

const RE_HEADING = /^ {0,3}(#{1,4})\s+(.*?)\s*#*\s*$/
const RE_RULE = /^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/
const RE_FENCE = /^ {0,3}(`{3,}|~{3,})\s*([^`\s]*).*$/
const RE_QUOTE = /^ {0,3}>\s?(.*)$/
const RE_ITEM = /^(\s*)(?:([-*+])|(\d{1,9})[.)])\s+(.*)$/
const RE_DELIMITER = /^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*$/

export function slugify(text: string): string {
  const base = text
    .toLowerCase()
    .replace(/[`*~[\]()#]/g, '')
    .replace(/[^a-z0-9\s-]/g, ' ')
    .trim()
    .replace(/\s+/g, '-')
  return base || 'section'
}

function splitRow(line: string): string[] {
  const cells: string[] = []
  let current = ''
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index]
    if (char === '\\' && line[index + 1] === '|') {
      current += '|'
      index += 1
      continue
    }
    if (char === '|') {
      cells.push(current)
      current = ''
      continue
    }
    current += char
  }
  cells.push(current)
  // A leading or trailing pipe produces an empty edge cell that carries no data.
  if (cells.length && cells[0].trim() === '') cells.shift()
  if (cells.length && cells[cells.length - 1].trim() === '') cells.pop()
  return cells.map((cell) => cell.trim())
}

function alignmentOf(cell: string): Align {
  const text = cell.trim()
  const left = text.startsWith(':')
  const right = text.endsWith(':')
  if (left && right) return 'center'
  if (right) return 'right'
  return 'left'
}

function startsBlock(line: string): boolean {
  return (
    RE_HEADING.test(line) ||
    RE_RULE.test(line) ||
    RE_FENCE.test(line) ||
    RE_QUOTE.test(line) ||
    RE_ITEM.test(line)
  )
}

function parseBlocks(lines: string[], claimId: (text: string) => string): Block[] {
  const blocks: Block[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      index += 1
      continue
    }

    const fence = RE_FENCE.exec(line)
    if (fence) {
      const marker = fence[1][0]
      const body: string[] = []
      index += 1
      while (index < lines.length) {
        const candidate = lines[index]
        if (new RegExp(`^ {0,3}${marker === '`' ? '`' : '~'}{3,}\\s*$`).test(candidate)) {
          index += 1
          break
        }
        body.push(candidate)
        index += 1
      }
      blocks.push({ kind: 'code', lang: fence[2] ?? '', code: body.join('\n') })
      continue
    }

    const heading = RE_HEADING.exec(line)
    if (heading) {
      const text = heading[2]
      blocks.push({ kind: 'heading', level: heading[1].length, id: claimId(text), text })
      index += 1
      continue
    }

    if (RE_RULE.test(line)) {
      blocks.push({ kind: 'rule' })
      index += 1
      continue
    }

    if (RE_QUOTE.test(line)) {
      const inner: string[] = []
      while (index < lines.length) {
        const quoted = RE_QUOTE.exec(lines[index])
        if (!quoted) {
          if (!lines[index].trim()) break
          inner.push(lines[index])
        } else {
          inner.push(quoted[1])
        }
        index += 1
      }
      blocks.push({ kind: 'quote', blocks: parseBlocks(inner, claimId) })
      continue
    }

    // A table is a header row plus an alignment row; without the second the
    // pipes are just characters in a paragraph.
    if (line.includes('|') && index + 1 < lines.length && RE_DELIMITER.test(lines[index + 1])) {
      const head = splitRow(line)
      const align = splitRow(lines[index + 1]).map(alignmentOf)
      index += 2
      const rows: string[][] = []
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        rows.push(splitRow(lines[index]))
        index += 1
      }
      blocks.push({ kind: 'table', head, align, rows })
      continue
    }

    const item = RE_ITEM.exec(line)
    if (item) {
      const ordered = item[2] === undefined
      const start = ordered ? Number(item[3]) : 1
      const items: ListItem[] = []
      let baseIndent = item[1].length

      while (index < lines.length) {
        const current = lines[index]
        if (!current.trim()) {
          const next = lines[index + 1] ?? ''
          if (!RE_ITEM.test(next)) break
          index += 1
          continue
        }
        const parsedItem = RE_ITEM.exec(current)
        if (!parsedItem) {
          if (startsBlock(current)) break
          // A continuation line belongs to the item above it.
          if (items.length) {
            items[items.length - 1].text = `${items[items.length - 1].text} ${current.trim()}`.trim()
            index += 1
            continue
          }
          break
        }
        const indent = parsedItem[1].length
        const isOrdered = parsedItem[2] === undefined
        if (indent <= baseIndent) {
          if (isOrdered !== ordered && indent === baseIndent && items.length) break
          baseIndent = Math.min(baseIndent, indent)
          items.push({ text: parsedItem[4], children: null })
        } else if (items.length) {
          const parent = items[items.length - 1]
          if (!parent.children) parent.children = { ordered: isOrdered, items: [] }
          parent.children.items.push(parsedItem[4])
        } else {
          items.push({ text: parsedItem[4], children: null })
        }
        index += 1
      }
      blocks.push({ kind: 'list', ordered, start: Number.isFinite(start) ? start : 1, items })
      continue
    }

    const paragraph: string[] = []
    while (index < lines.length && lines[index].trim() && !startsBlock(lines[index])) {
      const opensTable =
        paragraph.length > 0 &&
        lines[index].includes('|') &&
        index + 1 < lines.length &&
        RE_DELIMITER.test(lines[index + 1])
      if (opensTable) break
      paragraph.push(lines[index].trim())
      index += 1
    }
    if (paragraph.length) blocks.push({ kind: 'paragraph', text: paragraph.join(' ') })
    else index += 1
  }

  return blocks
}

export function parseMarkdown(source: string): Parsed {
  const headings: MarkdownHeading[] = []
  const used = new Map<string, number>()

  const claimId = (text: string): string => {
    const base = slugify(stripInline(text))
    const seen = used.get(base) ?? 0
    used.set(base, seen + 1)
    return seen === 0 ? base : `${base}-${seen}`
  }

  const lines = String(source ?? '')
    .replace(/\r\n?/g, '\n')
    .split('\n')
  const blocks = parseBlocks(lines, claimId)

  const collect = (list: Block[]) => {
    for (const block of list) {
      if (block.kind === 'heading') {
        headings.push({ id: block.id, level: block.level, text: stripInline(block.text) })
      } else if (block.kind === 'quote') {
        collect(block.blocks)
      }
    }
  }
  collect(blocks)

  return { blocks, headings }
}

/** The heading outline of a document, for a table of contents. */
export function extractHeadings(source: string): MarkdownHeading[] {
  return parseMarkdown(source).headings
}

/** Markdown punctuation removed, for titles and anchors. */
export function stripInline(text: string): string {
  return String(text ?? '')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/\*\*([\s\S]*?)\*\*/g, '$1')
    .replace(/\*([\s\S]*?)\*/g, '$1')
    .replace(/~~([\s\S]*?)~~/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .trim()
}

/* --- Inline grammar ------------------------------------------------------- */

const RE_INLINE = new RegExp(
  [
    '(`+)([\\s\\S]*?[^`])\\1(?!`)', // 1 ticks, 2 code
    '(\\*\\*|__)([\\s\\S]+?)\\3', // 3 delimiter, 4 strong
    '(\\*|_)([^\\s][\\s\\S]*?)\\5', // 5 delimiter, 6 emphasis
    '(~~)([\\s\\S]+?)\\7', // 7 delimiter, 8 strike
    '!?\\[([^\\]]*)\\]\\(((?:[^()\\s]|\\([^()\\s]*\\))*)(?:\\s+"[^"]*")?\\)', // 9 label, 10 href
    '<((?:https?://|mailto:)[^>\\s]+)>', // 11 autolink
  ].join('|'),
  'g',
)

/** Only schemes that cannot execute script survive into an href. */
function safeHref(raw: string): string | null {
  const href = String(raw ?? '').trim()
  if (!href) return null
  if (href.startsWith('#') || href.startsWith('/') || href.startsWith('./') || href.startsWith('../')) {
    return href
  }
  if (/^(https?:|mailto:)/i.test(href)) return href
  return null
}

function link(href: string, children: ReactNode, key: string): ReactNode {
  const safe = safeHref(href)
  if (!safe) return <span key={key}>{children}</span>
  const external = /^https?:/i.test(safe)
  return (
    <a key={key} href={safe} target={external ? '_blank' : undefined} rel={external ? 'noreferrer noopener' : undefined}>
      {children}
    </a>
  )
}

const isWordChar = (char: string | undefined) => Boolean(char) && /[\w]/.test(char as string)

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = []
  const source = String(text ?? '')
  let last = 0
  let counter = 0
  RE_INLINE.lastIndex = 0

  let match: RegExpExecArray | null
  while ((match = RE_INLINE.exec(source)) !== null) {
    const start = match.index
    const delimiter = match[3] ?? match[5]

    // Underscores inside identifiers such as missing_feature_count are not
    // emphasis. GitHub applies the same intraword rule.
    if (delimiter === '_' || delimiter === '__') {
      const before = start > 0 ? source[start - 1] : undefined
      const after = source[start + match[0].length]
      if (isWordChar(before) || isWordChar(after)) {
        RE_INLINE.lastIndex = start + delimiter.length
        continue
      }
    }

    if (start > last) out.push(source.slice(last, start))
    const key = `${keyPrefix}-${counter}`
    counter += 1

    if (match[1] !== undefined) {
      out.push(
        <code key={key} className="md__code">
          {match[2]}
        </code>,
      )
    } else if (match[3] !== undefined) {
      out.push(<strong key={key}>{renderInline(match[4], key)}</strong>)
    } else if (match[5] !== undefined) {
      out.push(<em key={key}>{renderInline(match[6], key)}</em>)
    } else if (match[7] !== undefined) {
      out.push(<s key={key}>{renderInline(match[8], key)}</s>)
    } else if (match[9] !== undefined) {
      out.push(link(match[10], renderInline(match[9], key), key))
    } else if (match[11] !== undefined) {
      out.push(link(match[11], match[11], key))
    }

    last = start + match[0].length
    RE_INLINE.lastIndex = last
  }

  if (last < source.length) out.push(source.slice(last))
  return out
}

/* --- Rendering ------------------------------------------------------------ */

function renderBlock(block: Block, key: string): ReactNode {
  switch (block.kind) {
    case 'heading': {
      const inline = renderInline(block.text, key)
      if (block.level === 1) return <h1 key={key} id={block.id} className="md__h1">{inline}</h1>
      if (block.level === 2) return <h2 key={key} id={block.id} className="md__h2">{inline}</h2>
      if (block.level === 3) return <h3 key={key} id={block.id} className="md__h3">{inline}</h3>
      return <h4 key={key} id={block.id} className="md__h4">{inline}</h4>
    }
    case 'paragraph':
      return (
        <p key={key} className="md__p">
          {renderInline(block.text, key)}
        </p>
      )
    case 'rule':
      return <hr key={key} className="md__rule" />
    case 'code':
      return (
        <pre key={key} className="md__pre" data-lang={block.lang || undefined}>
          <code>{block.code}</code>
        </pre>
      )
    case 'quote':
      return (
        <blockquote key={key} className="md__quote">
          {block.blocks.map((child, index) => renderBlock(child, `${key}-${index}`))}
        </blockquote>
      )
    case 'list': {
      const items = block.items.map((item, index) => (
        <li key={`${key}-${index}`} className="md__li">
          <span>{renderInline(item.text, `${key}-${index}`)}</span>
          {item.children && item.children.items.length > 0 &&
            (item.children.ordered ? (
              <ol className="md__list md__list--ordered md__list--nested">
                {item.children.items.map((child, childIndex) => (
                  <li key={`${key}-${index}-${childIndex}`} className="md__li">
                    <span>{renderInline(child, `${key}-${index}-${childIndex}`)}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <ul className="md__list md__list--bullet md__list--nested">
                {item.children.items.map((child, childIndex) => (
                  <li key={`${key}-${index}-${childIndex}`} className="md__li">
                    <span>{renderInline(child, `${key}-${index}-${childIndex}`)}</span>
                  </li>
                ))}
              </ul>
            ))}
        </li>
      ))
      return block.ordered ? (
        <ol key={key} start={block.start} className="md__list md__list--ordered">
          {items}
        </ol>
      ) : (
        <ul key={key} className="md__list md__list--bullet">
          {items}
        </ul>
      )
    }
    case 'table':
      return (
        <div key={key} className="md__table-wrap">
          <table className="md__table">
            <thead>
              <tr>
                {block.head.map((cell, index) => (
                  <th key={`${key}-h-${index}`} style={{ textAlign: block.align[index] ?? 'left' }}>
                    {renderInline(cell, `${key}-h-${index}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={`${key}-r-${rowIndex}`}>
                  {block.head.map((_, cellIndex) => (
                    <td key={`${key}-r-${rowIndex}-${cellIndex}`} style={{ textAlign: block.align[cellIndex] ?? 'left' }}>
                      {renderInline(row[cellIndex] ?? '', `${key}-r-${rowIndex}-${cellIndex}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    default:
      return null
  }
}

export function Markdown({ source, className }: { source: string; className?: string }): JSX.Element {
  const parsed = useMemo(() => parseMarkdown(source), [source])
  return (
    <div className={className ? `md ${className}` : 'md'}>
      {parsed.blocks.map((block, index) => renderBlock(block, `b${index}`))}
    </div>
  )
}

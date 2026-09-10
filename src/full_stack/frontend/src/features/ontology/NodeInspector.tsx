/** Right rail: everything known about the selected node. */

import { useMemo } from 'react'
import { MousePointerClick } from 'lucide-react'
import { Badge, Card, CopyButton, EmptyState } from '@/components/ui/primitives'
import { number, percent, signed } from '@/lib/format'
import type { AggregateStats, Band } from '@/lib/types'
import { BandDot, BoxGlyph, Breadcrumb, DeviationBar, MagnitudeBar, ScoreChip, crumbsOf } from './atoms'
import {
  BAND_LABEL,
  MEMBERSHIP_LABEL,
  bandColor,
  chainOf,
  collectAtoms,
  featureColumns,
  histogram,
  membershipOf,
  prettyLabel,
  spreadOf,
  type Membership,
  type OntologyIndex,
} from './model'
import { useMeasure } from './useOntologyIndex'

export interface NodeInspectorProps {
  index: OntologyIndex
  nodeKey: string | null
  feature: string | null
  onSelect: (key: string, feature: string | null) => void
}

export function NodeInspector({ index, nodeKey, feature, onSelect }: NodeInspectorProps) {
  const flat = nodeKey ? index.byKey.get(nodeKey) : null

  const values = useMemo(
    () => (flat ? collectAtoms(index, flat.key) : []),
    [index, flat],
  )
  const bins = useMemo(() => histogram(values, index.clamp), [values, index.clamp])

  if (!flat) {
    return (
      <Card title="Inspector" className="onto-rail__card">
        <EmptyState
          icon={<MousePointerClick size={20} />}
          title="Nothing selected"
          body="Pick a node in any view. Its evidence, its place in the hierarchy and, in a cohort, its spread all land here."
        />
      </Card>
    )
  }

  const node = flat.node
  const aggregate = node.aggregate ?? null
  const membership = membershipOf(node)
  const chain = chainOf(index, flat.key)
  const isLeaf = flat.children.length === 0
  const coverage = node.leaf_count > 0 ? node.present_leaves / node.leaf_count : null
  // A leaf shows the parent feature it restates; a branch shows only the
  // features no child already accounts for.
  const features = flat.attached ? [flat.attached, ...flat.ownFeatures] : flat.ownFeatures
  const columns = featureColumns(features)

  const children = flat.children
    .map((key) => index.byKey.get(key))
    .filter((child): child is NonNullable<typeof child> => Boolean(child))
    .sort((a, b) => (b.magnitude ?? -1) - (a.magnitude ?? -1))
  const childMax = children.reduce((max, child) => Math.max(max, child.magnitude ?? 0), 0)

  return (
    <Card
      title="Inspector"
      subtitle={isLeaf ? 'Measured leaf' : `${number(node.leaf_count)} leaves beneath`}
      actions={<CopyButton text={node.path.join(' / ')} label="Copy path" />}
      className="onto-rail__card"
    >
      <div className="stack gap-4">
        <Breadcrumb chain={crumbsOf(chain)} onSelect={(key) => onSelect(key, null)} />

        <div className="onto-focus">
          <span
            className="onto-focus__value tabular"
            style={{ color: bandColor(node.band) }}
            title={node.score === null ? 'Not measured' : String(node.score)}
            aria-label="Deviation of the selected node"
          >
            {isLeaf
              ? signed(node.score)
              : node.mean_abs_score === null
                ? '-'
                : node.mean_abs_score.toFixed(2)}
          </span>
          <div className="stack gap-1 grow" style={{ minWidth: 0 }}>
            <span className="row gap-2">
              <BandDot band={node.band} />
              <span className="t-small semibold">{BAND_LABEL[node.band]}</span>
            </span>
            <span className="t-tiny muted">
              {isLeaf ? 'signed deviation' : 'mean absolute deviation of the subtree'}
            </span>
            <DeviationBar
              score={isLeaf ? node.score : flat.signedMean}
              clamp={index.clamp}
              width={168}
              height={9}
            />
          </div>
        </div>

        {aggregate && <CohortBlock aggregate={aggregate} membership={membership} clamp={index.clamp} />}

        <dl className="kv onto-kv">
          <dt>Depth</dt>
          <dd className="tabular">{node.depth + 1}</dd>
          <dt>Leaves</dt>
          <dd className="tabular">{number(node.leaf_count)}</dd>
          <dt>Present leaves</dt>
          <dd className="tabular">{number(node.present_leaves)}</dd>
          <dt>Coverage</dt>
          <dd className="tabular">{coverage === null ? '-' : percent(coverage, 0)}</dd>
          <dt>Subtree features</dt>
          <dd className="tabular">{number(node.feature_count)}</dd>
          <dt>Signed mean</dt>
          <dd className="tabular">{signed(flat.signedMean)}</dd>
        </dl>

        {bins.total > 0 && (
          <section className="stack gap-2">
            <span className="onto-section">
              Subtree distribution
              <span className="t-tiny muted"> {number(bins.total)} values</span>
            </span>
            <Histogram bins={bins.bins} max={bins.max} clamp={index.clamp} />
          </section>
        )}

        {features.length > 0 && (
          <section className="stack gap-2">
            <span className="onto-section">Features</span>
            <div className="onto-scrollbox">
              <table className="table onto-features">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th className="num">z</th>
                    {columns.refRange && <th>Ref</th>}
                    {columns.significance && <th>Note</th>}
                    {columns.percentile && <th>Pct</th>}
                  </tr>
                </thead>
                <tbody>
                  {features.map((row) => (
                    <tr
                      key={row.feature}
                      data-selected={feature === row.feature || undefined}
                      onClick={() => onSelect(flat.key, row.feature)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <span className="row gap-2" style={{ minWidth: 0 }}>
                          <BandDot band={row.band} size={6} />
                          <span className="truncate" title={row.feature}>
                            {prettyLabel(row.feature)}
                          </span>
                        </span>
                      </td>
                      <td className="truncate">{String(row.value ?? '-')}</td>
                      <td className="num tabular">{row.z_score === null ? '-' : signed(row.z_score)}</td>
                      {columns.refRange && <td className="t-tiny">{row.ref_range ?? '-'}</td>}
                      {columns.significance && <td className="t-tiny">{row.significance ?? '-'}</td>}
                      {columns.percentile && <td className="t-tiny">{row.percentile ?? '-'}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {children.length > 0 && (
          <section className="stack gap-2">
            <span className="onto-section">
              Children
              <span className="t-tiny muted"> ranked by deviation</span>
            </span>
            <div className="onto-scrollbox stack" style={{ gap: 2 }}>
              {children.map((child) => (
                <button
                  key={child.key}
                  type="button"
                  className="onto-child"
                  onClick={() => onSelect(child.key, null)}
                >
                  <span className="onto-child__label truncate" title={child.node.label}>
                    {prettyLabel(child.node.label)}
                  </span>
                  <ScoreChip
                    band={child.node.band}
                    value={child.children.length === 0 ? child.node.score : child.magnitude}
                    aggregate={child.children.length > 0}
                  />
                  <MagnitudeBar
                    value={child.magnitude ?? 0}
                    max={Math.max(childMax, 0.001)}
                    band={child.node.band}
                  />
                  <span className="t-micro muted tabular">{number(child.node.leaf_count)}</span>
                </button>
              ))}
            </div>
          </section>
        )}

        {isLeaf && features.length === 0 && (
          <Badge tone="neutral" outline>
            No feature payload for this leaf
          </Badge>
        )}
      </div>
    </Card>
  )
}

/**
 * What the cohort says about one node.
 *
 * Membership first, because a mean over three of nine participants means
 * something different from a mean over all nine, and the reader has to see
 * which one they are looking at before they read the number.
 */
function CohortBlock({
  aggregate,
  membership,
  clamp,
}: {
  aggregate: AggregateStats
  membership: Membership | null
  clamp: number
}) {
  const spread = spreadOf(aggregate)
  const tone: Record<Membership, 'positive' | 'info' | 'caution' | 'accent'> = {
    shared: 'positive',
    common: 'info',
    partial: 'caution',
    unique: 'accent',
  }
  return (
    <section className="stack gap-2 onto-agg">
      <div className="row between gap-2 wrap">
        {membership && <Badge tone={tone[membership]}>{MEMBERSHIP_LABEL[membership]}</Badge>}
        <span className="t-tiny muted tabular">
          {number(aggregate.present_in)} / {number(aggregate.participant_count)} participants
        </span>
      </div>
      {spread && spread.n > 1 ? (
        <>
          <BoxGlyph spread={spread} clamp={clamp} width={196} height={16} />
          <dl className="kv onto-kv">
            <dt>Median</dt>
            <dd className="tabular">{signed(aggregate.median)}</dd>
            <dt>Quartiles</dt>
            <dd className="tabular">
              {signed(aggregate.q1)} to {signed(aggregate.q3)}
            </dd>
            <dt>Range</dt>
            <dd className="tabular">
              {signed(aggregate.min)} to {signed(aggregate.max)}
            </dd>
            <dt>Mean, sd</dt>
            <dd className="tabular">
              {signed(aggregate.mean)}
              {aggregate.sd === null ? '' : ` ± ${aggregate.sd.toFixed(2)}`}
            </dd>
            <dt>Measured</dt>
            <dd className="tabular">n {number(aggregate.n)}</dd>
          </dl>
        </>
      ) : (
        <span className="t-tiny muted">
          {aggregate.n === 1
            ? 'One participant carries this, so there is no spread to show.'
            : 'No measured value in the selection.'}
        </span>
      )}
    </section>
  )
}

function Histogram({
  bins,
  max,
  clamp,
}: {
  bins: { from: number; to: number; count: number; band: Band }[]
  max: number
  clamp: number
}) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const width = Math.max(120, size.width || 260)
  const height = 68
  const barWidth = width / bins.length

  return (
    <div ref={ref} className="onto-hist">
      <svg width={width} height={height} role="img" aria-label="Distribution of subtree z scores">
        {bins.map((bin, at) => {
          const value = max > 0 ? bin.count / max : 0
          const barHeight = bin.count > 0 ? Math.max(2, value * (height - 14)) : 0
          return (
            <rect
              key={bin.from}
              x={at * barWidth + 0.5}
              y={height - 12 - barHeight}
              width={Math.max(1, barWidth - 1)}
              height={barHeight}
              rx={1.5}
              style={{ fill: bandColor(bin.band) }}
            >
              <title>{`${bin.from.toFixed(1)} to ${bin.to.toFixed(1)}: ${bin.count}`}</title>
            </rect>
          )
        })}
        <line className="onto-hist__axis" x1={0} x2={width} y1={height - 12} y2={height - 12} />
        <line className="onto-hist__zero" x1={width / 2} x2={width / 2} y1={0} y2={height - 8} />
        <text className="onto-hist__tick" x={2} y={height - 2}>
          {-clamp}
        </text>
        <text className="onto-hist__tick" x={width / 2} y={height - 2} textAnchor="middle">
          0
        </text>
        <text className="onto-hist__tick" x={width - 2} y={height - 2} textAnchor="end">
          {clamp}
        </text>
      </svg>
    </div>
  )
}

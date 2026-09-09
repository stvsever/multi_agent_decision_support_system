/** Application chrome: navigation rail, top bar, and the content frame. */

import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  CircleDashed,
  Compass,
  FileText,
  Layers,
  LayoutGrid,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Sparkles,
  X,
} from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'
import type { ReactNode } from 'react'
import { useAccount, useRuns } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { usd } from '@/lib/format'
import { Button, Tooltip } from '@/components/ui/primitives'
import './shell.css'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutGrid, end: true, hint: 'Where things stand' },
  { to: '/studio', label: 'Configure', icon: Sparkles, hint: 'Compose a run' },
  { to: '/runs', label: 'Runs', icon: Activity, hint: 'Live and past runs' },
  { to: '/ontology', label: 'Ontology', icon: Network, hint: 'Explore the evidence taxonomy' },
  { to: '/reports', label: 'Reports', icon: FileText, hint: 'Deep phenotype output' },
  { to: '/batch', label: 'Batch', icon: Layers, hint: 'Run a cohort' },
]

export function AppShell({ children }: { children: ReactNode }) {
  const { navCollapsed, toggleNav, openSettings } = useApp()
  const location = useLocation()

  return (
    <div className={clsx('shell', navCollapsed && 'shell--collapsed')}>
      <nav className="shell__nav" aria-label="Primary">
        <div className="shell__brand">
          <span className="shell__mark" aria-hidden>
            <Compass size={17} />
          </span>
          {!navCollapsed && (
            <span className="stack" style={{ gap: 0, minWidth: 0 }}>
              <span className="shell__wordmark">COMPASS</span>
              <span className="t-micro muted truncate">Decision support engine</span>
            </span>
          )}
        </div>

        <ul className="shell__links">
          {NAV.map((item) => (
            <li key={item.to}>
              <Tooltip content={navCollapsed ? item.label : ''} side="right">
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => clsx('shell__link', isActive && 'shell__link--active')}
                  data-tour={`nav-${item.label.toLowerCase()}`}
                >
                  <item.icon size={16} className="shell__link-icon" />
                  {!navCollapsed && (
                    <span className="stack grow" style={{ gap: 0, minWidth: 0 }}>
                      <span className="truncate">{item.label}</span>
                    </span>
                  )}
                </NavLink>
              </Tooltip>
            </li>
          ))}
        </ul>

        <div className="shell__nav-footer">
          <ActiveRunsBadge collapsed={navCollapsed} />
          <button
            type="button"
            className="shell__collapse"
            onClick={toggleNav}
            aria-label={navCollapsed ? 'Expand navigation' : 'Collapse navigation'}
          >
            {navCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
            {!navCollapsed && <span className="t-tiny">Collapse</span>}
          </button>
        </div>
      </nav>

      <div className="shell__main">
        <header className="shell__topbar">
          <Breadcrumb path={location.pathname} />
          <div className="grow" />
          <ConnectionPill />
          <Tooltip content="Settings">
            <button
              type="button"
              className="shell__gear"
              onClick={() => openSettings()}
              aria-label="Open settings"
              data-tour="settings-button"
            >
              <Settings size={16} />
            </button>
          </Tooltip>
        </header>

        <main className="shell__content">{children}</main>
      </div>
      <ToastStack />
    </div>
  )
}

const TITLES: Record<string, string> = {
  '/': 'Overview',
  '/studio': 'Configure a run',
  '/runs': 'Runs',
  '/ontology': 'Ontology explorer',
  '/reports': 'Reports',
  '/batch': 'Batch',
}

function Breadcrumb({ path }: { path: string }) {
  const base = `/${path.split('/')[1] ?? ''}` || '/'
  const title = TITLES[base] ?? TITLES[path] ?? 'COMPASS'
  const sub = path.split('/').slice(2).filter(Boolean)
  return (
    <div className="row gap-2" style={{ minWidth: 0 }}>
      <span className="t-body semibold truncate">{title}</span>
      {sub.length > 0 && (
        <>
          <ChevronLeft size={13} className="faint" style={{ transform: 'rotate(180deg)' }} />
          <span className="t-small muted mono truncate">{sub.join(' / ')}</span>
        </>
      )}
    </div>
  )
}

function ActiveRunsBadge({ collapsed }: { collapsed: boolean }) {
  const { data } = useRuns(true)
  const active = (data?.runs ?? []).filter((r) => r.status === 'running' || r.status === 'queued')
  if (active.length === 0) return null
  return (
    <NavLink to="/runs" className="shell__active">
      <CircleDashed size={13} className="spin" />
      {!collapsed && <span className="t-tiny">{active.length} running</span>}
    </NavLink>
  )
}

function ConnectionPill() {
  const { data, isLoading } = useAccount()
  const openSettings = useApp((s) => s.openSettings)

  const tone = isLoading ? 'idle' : data?.valid ? 'ok' : 'warn'
  const label = isLoading
    ? 'Checking'
    : data?.valid
      ? data.remaining_usd != null
        ? `${usd(data.remaining_usd)} left`
        : 'Connected'
      : 'Not connected'

  return (
    <Tooltip
      content={
        isLoading
          ? 'Verifying the provider connection'
          : data?.valid
            ? `OpenRouter connected${data.usage_usd != null ? `. ${usd(data.usage_usd)} used to date.` : ''}`
            : (data?.reason ?? 'Add an OpenRouter key to run the engine.')
      }
    >
      <button type="button" className={clsx('shell__pill', `shell__pill--${tone}`)} onClick={() => openSettings('connection')}>
        {tone === 'ok' ? <CheckCircle2 size={13} /> : tone === 'warn' ? <AlertTriangle size={13} /> : <CircleDashed size={13} className="spin" />}
        <span className="t-tiny">{label}</span>
      </button>
    </Tooltip>
  )
}

function ToastStack() {
  const { toasts, dismissToast } = useApp()
  return (
    <div className="toasts">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            layout
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 24, transition: { duration: 0.15 } }}
            className={clsx('toast', `toast--${toast.tone}`)}
          >
            <div className="grow stack gap-1">
              <span className="semibold">{toast.title}</span>
              {toast.body && <span className="t-tiny muted">{toast.body}</span>}
            </div>
            <Button variant="ghost" size="sm" iconOnly icon={<X size={13} />} onClick={() => dismissToast(toast.id)} aria-label="Dismiss" />
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

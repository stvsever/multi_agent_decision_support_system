/** Application chrome: navigation rail, top bar, and the content frame. */

import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  ChevronLeft,
  CircleDashed,
  Compass,
  FileText,
  LayoutGrid,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Sparkles,
  WifiOff,
  X,
} from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'
import { useEffect, useState, useSyncExternalStore } from 'react'
import type { ReactNode } from 'react'
import { useConnectivity, useRuns } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { Button, Tooltip } from '@/components/ui/primitives'
import { openWelcome } from '@/components/welcome/welcomeOpen'
import './shell.css'

/* Batch is reachable from Configure rather than from the rail: it is a mode of
   composing a run, not a section of its own. */
const NAV = [
  { to: '/', label: 'Overview', icon: LayoutGrid, end: true },
  { to: '/studio', label: 'Configure', icon: Sparkles },
  { to: '/runs', label: 'Runs', icon: Activity },
  { to: '/ontology', label: 'Ontology', icon: Network },
  { to: '/reports', label: 'Reports', icon: FileText },
]

/* A window this narrow has no room for a labelled rail. The breakpoint is read
   here rather than acted on in CSS so that the one class carries both the width
   and the centring: a media query that only set the width left every icon
   sitting off the rail's centre line. */
const NARROW = '(max-width: 900px)'

function useNarrowWindow(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      // Both signals, because a browser that misses one still resizes: the
      // query is the answer, the events are only the prompt to re-read it.
      const query = window.matchMedia(NARROW)
      query.addEventListener('change', onChange)
      window.addEventListener('resize', onChange)
      window.addEventListener('orientationchange', onChange)
      return () => {
        query.removeEventListener('change', onChange)
        window.removeEventListener('resize', onChange)
        window.removeEventListener('orientationchange', onChange)
      }
    },
    () => window.matchMedia(NARROW).matches,
    () => false,
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  const { navCollapsed, toggleNav, openSettings } = useApp()
  const location = useLocation()
  const narrow = useNarrowWindow()
  /* What the rail actually is, as opposed to what the user asked for. */
  const collapsed = navCollapsed || narrow

  return (
    <div className={clsx('shell', collapsed && 'shell--collapsed')}>
      <nav className="shell__nav" aria-label="Primary">
        <div className="shell__brand">
          <Tooltip content="About COMPASS" side={collapsed ? 'right' : 'bottom'}>
            <button type="button" className="shell__mark" onClick={openWelcome} aria-label="About COMPASS">
              <Compass size={17} />
            </button>
          </Tooltip>
          {!collapsed && (
            <span className="stack" style={{ gap: 0, minWidth: 0 }}>
              <span className="shell__wordmark">COMPASS</span>
              <span className="t-micro muted truncate">Decision support engine</span>
            </span>
          )}
        </div>

        <ul className="shell__links">
          {NAV.map((item) => (
            <li key={item.to}>
              <Tooltip content={collapsed ? item.label : ''} side="right">
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => clsx('shell__link', isActive && 'shell__link--active')}
                  data-tour={`nav-${item.label.toLowerCase()}`}
                  aria-label={item.label}
                >
                  <item.icon size={16} className="shell__link-icon" />
                  {!collapsed && (
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
          <ActiveRunsBadge collapsed={collapsed} />
          <button
            type="button"
            className="shell__collapse"
            onClick={toggleNav}
            disabled={narrow}
            aria-label={
              narrow
                ? 'Navigation stays collapsed at this window width'
                : collapsed
                  ? 'Expand navigation'
                  : 'Collapse navigation'
            }
          >
            {collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
            {!collapsed && <span className="t-tiny">Collapse</span>}
          </button>
        </div>
      </nav>

      <div className="shell__main">
        <header className="shell__topbar">
          <Breadcrumb path={location.pathname} />
          <div className="grow" />
          <OfflineChip />
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

/* Keyed by the first path segment. /batch keeps a title because the route is
   still reachable from Configure even though it left the rail. */
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
    <NavLink to="/runs" className="shell__active" aria-label={`${active.length} running`}>
      <CircleDashed size={13} className="spin" />
      {!collapsed && <span className="t-tiny">{active.length} running</span>}
    </NavLink>
  )
}

/**
 * Reachability, and only when it fails. A healthy connection is the normal
 * case and gets no screen space; the chip appears only when something is
 * actually broken and offers the one control that can fix it.
 */
function OfflineChip() {
  const { data, isError, refetch } = useConnectivity()
  const openSettings = useApp((s) => s.openSettings)
  const [deviceOnline, setDeviceOnline] = useState(() => navigator.onLine)

  // The poll is slow, so a dropped local network would otherwise take up to
  // half a minute to surface. The browser tells us immediately.
  useEffect(() => {
    const up = () => {
      setDeviceOnline(true)
      void refetch()
    }
    const down = () => setDeviceOnline(false)
    window.addEventListener('online', up)
    window.addEventListener('offline', down)
    return () => {
      window.removeEventListener('online', up)
      window.removeEventListener('offline', down)
    }
  }, [refetch])

  const broken = !deviceOnline || isError || data?.online === false || data?.provider_reachable === false
  if (!broken) return null

  const reason = !deviceOnline
    ? 'This device is offline.'
    : isError
      ? 'The dashboard service did not answer.'
      : data?.reason
        ? data.reason
        : 'The model provider is not reachable.'

  return (
    <Tooltip content={`${reason} Open connection settings.`}>
      <button type="button" className="shell__offline" onClick={() => openSettings('connection')}>
        <WifiOff size={12} />
        <span className="t-micro">No connection</span>
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

import { useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { TourLayer } from '@/components/tour/TourLayer'
import { WelcomeDialog } from '@/components/welcome/WelcomeDialog'
import { BatchPage } from '@/features/batch/BatchPage'
import { OntologyPage } from '@/features/ontology/OntologyPage'
import { OverviewPage } from '@/features/overview/OverviewPage'
import { ReportsPage } from '@/features/reports/ReportsPage'
import { RunDetailPage } from '@/features/run/RunDetailPage'
import { RunsPage } from '@/features/run/RunsPage'
import { SettingsSheet } from '@/features/settings/SettingsSheet'
import { StudioPage } from '@/features/studio/StudioPage'
import { setLocale } from '@/lib/format'
import { useSettings } from '@/lib/hooks'
import { applyAppearance, useApp } from '@/lib/store'

export default function App() {
  const appearance = useApp((s) => s.appearance)
  const setAppearance = useApp((s) => s.setAppearance)
  const { data } = useSettings()

  // Apply the local mirror straight away so there is no unstyled flash, then
  // reconcile with whatever the service has stored.
  useEffect(() => {
    applyAppearance(appearance)
    setLocale(appearance.numericLocale)
  }, [appearance])

  useEffect(() => {
    const remote = data?.config.appearance
    if (!remote) return
    setAppearance({
      theme: remote.theme,
      accent: remote.accent,
      density: remote.density,
      fontScale: remote.font_scale,
      reducedMotion: remote.reduced_motion,
      numericLocale: remote.numeric_locale,
    })
  }, [data?.config.appearance, setAppearance])

  return (
    <>
      <AppShell>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/studio" element={<StudioPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/ontology" element={<OntologyPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/reports/:participant" element={<ReportsPage />} />
          <Route path="/batch" element={<BatchPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
      <SettingsSheet />
      <WelcomeDialog />
      <TourLayer />
    </>
  )
}

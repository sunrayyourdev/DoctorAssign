'use client'

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DoctorsStats } from "./components/doctors-stats"
import { PracticePerformance } from "./components/practice-performance"
import { MedicalInsights } from "./components/medical-insights"
import { PatientFeedback } from "./components/patient-feedback"

export default function Dashboard() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Admin's Dashboard</h1>
      <Tabs defaultValue="patient-stats" className="space-y-4">
        <TabsList>
          <TabsTrigger value="doctors-stats">Doctors Stats</TabsTrigger>
          <TabsTrigger value="practice-performance">Practice Performance</TabsTrigger>
          <TabsTrigger value="medical-insights">Medical Insights</TabsTrigger>
          <TabsTrigger value="patient-feedback">Patient Feedback</TabsTrigger>
        </TabsList>
        <TabsContent value="doctors-stats">
          <DoctorsStats />
        </TabsContent>
        <TabsContent value="practice-performance">
          <PracticePerformance />
        </TabsContent>
        <TabsContent value="medical-insights">
          <MedicalInsights />
        </TabsContent>
        <TabsContent value="patient-feedback">
          <PatientFeedback />
        </TabsContent>
      </Tabs>
    </div>
  )
}


import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export function MedicalInsights() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>Most Common Diagnosis</CardTitle>
          <CardDescription>Last 30 days</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold">Hypertension</p>
          <p className="text-sm text-muted-foreground">22% of all diagnoses</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Average Consultation</CardTitle>
          <CardDescription>Time spent with patients</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold">18 minutes</p>
          <p className="text-sm text-muted-foreground">+2 minutes from last month</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Prescription Rate</CardTitle>
          <CardDescription>Medications per visit</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold">1.8</p>
          <p className="text-sm text-muted-foreground">-0.2 from last quarter</p>
        </CardContent>
      </Card>
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Clinical Insights</CardTitle>
          <CardDescription>Important observations and trends</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="list-disc pl-5 space-y-2">
            <li>Seasonal allergies have increased by 15% compared to the same period last year.</li>
            <li>Patients who follow the recommended exercise plan show 30% better outcomes for Type 2 Diabetes.</li>
            <li>The new medication protocol for hypertension has reduced hospital readmissions by 22%.</li>
            <li>Patients who use the telehealth follow-up option have 18% better medication adherence.</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}


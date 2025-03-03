import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

const data = [
  { name: "Jan", waitTime: 22, satisfaction: 92 },
  { name: "Feb", waitTime: 18, satisfaction: 94 },
  { name: "Mar", waitTime: 15, satisfaction: 95 },
  { name: "Apr", waitTime: 17, satisfaction: 93 },
  { name: "May", waitTime: 14, satisfaction: 96 },
  { name: "Jun", waitTime: 12, satisfaction: 97 },
]

export function PracticePerformance() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>Average Wait Time</CardTitle>
          <CardDescription>Minutes before appointment</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis
                stroke="#888888"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value}m`}
              />
              <Tooltip />
              <Line type="monotone" dataKey="waitTime" stroke="#8884d8" activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Patient Satisfaction</CardTitle>
          <CardDescription>Overall score (%)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis
                stroke="#888888"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip />
              <Line type="monotone" dataKey="satisfaction" stroke="#82ca9d" activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>No-Show Rate</CardTitle>
          <CardDescription>Last 30 days</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-4xl font-bold">3.2%</div>
          <p className="text-xs text-muted-foreground">-0.8% from last month</p>
        </CardContent>
      </Card>
    </div>
  )
}


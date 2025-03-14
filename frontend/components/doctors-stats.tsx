import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { name: "Jan", total: 85 },
  { name: "Feb", total: 92 },
  { name: "Mar", total: 105 },
  { name: "Apr", total: 120 },
  { name: "May", total: 115 },
  { name: "Jun", total: 130 },
]

export function DoctorsStats() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Doctors</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">1,245</div>
          <p className="text-xs text-muted-foreground">+15 from last month</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Appointments Today</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">24</div>
          <p className="text-xs text-muted-foreground">4 remaining</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Doctor Growth & Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <div className="text-2xl font-bold">18</div>
            <p className="text-xs text-muted-foreground">New doctors this week</p>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data}>
              <XAxis 
                dataKey="name" 
                stroke="#888888" 
                fontSize={12} 
                tickLine={false} 
                axisLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                stroke="#888888"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `${value}`}
                width={40}
              />
              <Bar 
                dataKey="total" 
                fill="#4cc9f0" 
                radius={[4, 4, 0, 0]}
                label={{ position: 'top', fill: '#888888', fontSize: 12 }}
              />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}


"use client"

import { useState } from "react"
import { ChevronDown, ChevronUp, Mail, Phone } from "lucide-react"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// Sample data for doctors and their patients
const doctors = [
  {
    id: 1,
    name: "Dr. Sarah Johnson",
    specialty: "Cardiology",
    image: "/placeholder.svg?height=100&width=100",
    email: "sarah.johnson@example.com",
    phone: "(555) 123-4567",
    patients: [
      { id: 101, name: "James Wilson", age: 62, condition: "Hypertension" },
      { id: 102, name: "Emma Davis", age: 54, condition: "Arrhythmia" },
      { id: 103, name: "Michael Brown", age: 70, condition: "Heart Failure" },
    ],
  },
  {
    id: 2,
    name: "Dr. Robert Chen",
    specialty: "Neurology",
    image: "/placeholder.svg?height=100&width=100",
    email: "robert.chen@example.com",
    phone: "(555) 234-5678",
    patients: [
      { id: 201, name: "Olivia Martinez", age: 45, condition: "Migraine" },
      { id: 202, name: "William Taylor", age: 67, condition: "Parkinson's" },
      { id: 203, name: "Sophia Anderson", age: 38, condition: "Multiple Sclerosis" },
    ],
  },
  {
    id: 3,
    name: "Dr. Lisa Patel",
    specialty: "Pediatrics",
    image: "/placeholder.svg?height=100&width=100",
    email: "lisa.patel@example.com",
    phone: "(555) 345-6789",
    patients: [
      { id: 301, name: "Noah Garcia", age: 7, condition: "Asthma" },
      { id: 302, name: "Ava Robinson", age: 3, condition: "Ear Infection" },
      { id: 303, name: "Liam Johnson", age: 10, condition: "Allergies" },
    ],
  },
  {
    id: 4,
    name: "Dr. Marcus Williams",
    specialty: "Orthopedics",
    image: "/placeholder.svg?height=100&width=100",
    email: "marcus.williams@example.com",
    phone: "(555) 456-7890",
    patients: [
      { id: 401, name: "Charlotte Lee", age: 28, condition: "Fractured Wrist" },
      { id: 402, name: "Benjamin Clark", age: 45, condition: "Knee Replacement" },
      { id: 403, name: "Amelia Walker", age: 67, condition: "Osteoarthritis" },
      { id: 404, name: "Lucas Hall", age: 34, condition: "Sports Injury" },
    ],
  },
]

export default function DoctorList() {
  const [expandedDoctors, setExpandedDoctors] = useState<number[]>([])

  const toggleExpanded = (doctorId: number) => {
    console.log("Toggling doctor:", doctorId) // Debug log
    setExpandedDoctors((current) =>
      current.includes(doctorId) ? current.filter((id) => id !== doctorId) : [...current, doctorId],
    )
  }

  console.log("Expanded doctors:", expandedDoctors) // Debug log

  return (
    <div className="container mx-auto py-6">
      <h2 className="text-2xl font-bold mb-6">Medical Staff</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {doctors.map((doctor) => (
          <Card key={doctor.id} className="overflow-hidden">
            <CardHeader className="pb-2">
              <div className="flex items-start gap-4">
                <Avatar className="h-12 w-12">
                  <AvatarImage src={doctor.image} alt={doctor.name} />
                  <AvatarFallback>
                    {doctor.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")}
                  </AvatarFallback>
                </Avatar>
                <div className="space-y-1">
                  <CardTitle>{doctor.name}</CardTitle>
                  <Badge variant="outline">{doctor.specialty}</Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-sm space-y-2 mb-4">
                <div className="flex items-center gap-2">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  <span>{doctor.email}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Phone className="h-4 w-4 text-muted-foreground" />
                  <span>{doctor.phone}</span>
                </div>
              </div>
              <div className="border-t pt-4">
                <Button
                  variant="ghost"
                  className="flex w-full justify-between"
                  onClick={() => toggleExpanded(doctor.id)}
                >
                  <span>Patients ({doctor.patients.length})</span>
                  {expandedDoctors.includes(doctor.id) ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </Button>
                {expandedDoctors.includes(doctor.id) && (
                  <div className="mt-2 space-y-2">
                    {doctor.patients.map((patient) => (
                      <div key={patient.id} className="p-2 bg-muted rounded-md">
                        <div className="font-medium">{patient.name}</div>
                        <div className="text-sm text-muted-foreground">
                          Age: {patient.age} • {patient.condition}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}


"use client"

import { useState } from "react"
import { Pencil, Check, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import Layout from "./layout1"
import Link from "next/link";

export default function Page() {
    const [appointments, setAppointments] = useState([
        {
            id: 1,
            status: "pending",
            title: "Annual checkup - John Smith",
            date: "2022-02-18 14:09:54",
            patientName: "John Smith",
            reason: "Annual physical examination",
            requestedTime: "10:00 AM - 11:00 AM",
        },
        {
            id: 2,
            status: "pending",
            title: "Follow-up consultation - Sarah Johnson",
            date: "2022-02-18 12:47:08",
            patientName: "Sarah Johnson",
            reason: "Follow-up on medication",
            requestedTime: "1:30 PM - 2:00 PM",
        },
        {
            id: 3,
            status: "pending",
            title: "New patient consultation - Michael Brown",
            date: "2022-02-18 11:27:54",
            patientName: "Michael Brown",
            reason: "Persistent headaches",
            requestedTime: "3:15 PM - 4:00 PM",
        },
    ])

    const [sortOption, setSortOption] = useState("date-desc")

    const handleSortChange = (value: string) => {
        setSortOption(value)
    }

    const getSortedAppointments = () => {
        return [...appointments].sort((a, b) => {
            if (sortOption === "date-desc") {
                return new Date(b.date).getTime() - new Date(a.date).getTime()
            } else if (sortOption === "date-asc") {
                return new Date(a.date).getTime() - new Date(b.date).getTime()
            } else if (sortOption === "alpha-asc") {
                return a.patientName.localeCompare(b.patientName)
            } else if (sortOption === "alpha-desc") {
                return b.patientName.localeCompare(a.patientName)
            }
            return 0
        })
    }

    const handleApprove = (id: number) => {
        setAppointments(appointments.map((app) => (app.id === id ? { ...app, status: "approved" } : app)))
    }

    const handleReject = (id: number) => {
        setAppointments(appointments.map((app) => (app.id === id ? { ...app, status: "rejected" } : app)))
    }

    return (
        <Layout>
            <div className="grid gap-6">
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base font-medium">Doctor Profile</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-600">Name</span>
                                <span>Dr. Robert Williams</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">Specialty</span>
                                <span>Cardiology</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">License Number</span>
                                <span>MD6490101</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">Status</span>
                                <span>Active</span>
                            </div>
                            <div className="flex justify-between col-span-2">
                                <span className="text-gray-600">Office Address</span>
                                <span>123 Medical Center Drive, Suite 101</span>
                            </div>
                            <div className="flex justify-between col-span-2">
                                <span className="text-gray-600">Education</span>
                                <span>Harvard Medical School, Residency at Mayo Clinic</span>
                            </div>
                            <div className="flex justify-between col-span-2">
                                <span className="text-gray-600">Contact</span>
                                <span>office: (555) 123-4567 / mobile: (555) 987-6543</span>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-3">
                        <CardTitle className="text-base font-medium">Patient Appointment Requests</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-medium">Sort by:</span>
                                <Select defaultValue="date-desc" onValueChange={handleSortChange}>
                                    <SelectTrigger className="w-[200px]">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="date-desc">Date (newest first)</SelectItem>
                                        <SelectItem value="date-asc">Date (oldest first)</SelectItem>
                                        <SelectItem value="alpha-asc">Name (A-Z)</SelectItem>
                                        <SelectItem value="alpha-desc">Name (Z-A)</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="border rounded-lg">
                                {getSortedAppointments().map((appointment) => (
                                    <div
                                        key={appointment.id}
                                        className="grid grid-cols-[auto_1fr_auto] items-center gap-4 p-3 hover:bg-gray-50 border-t first:border-t-0"
                                    >
                                        <Badge
                                            variant={
                                                appointment.status === "pending"
                                                    ? "outline"
                                                    : appointment.status === "approved"
                                                        ? "default" // Change "success" to "default" or "secondary"
                                                        : "destructive"
                                            }
                                        >

                                            {appointment.status === "pending"
                                                ? "Pending"
                                                : appointment.status === "approved"
                                                    ? "Approved"
                                                    : "Rejected"}
                                        </Badge>
                                        <div>
                                            <div className="text-sm font-medium">{appointment.title}</div>
                                            <div className="text-xs text-gray-500">{appointment.date}</div>
                                            <div className="text-xs mt-1">
                                                <span className="font-semibold">Reason:</span> {appointment.reason}
                                            </div>
                                            <div className="text-xs">
                                                <span className="font-semibold">Requested time:</span> {appointment.requestedTime}
                                            </div>
                                        </div>
                                        <div className="flex gap-1">
                                            {appointment.status === "pending" && (
                                                <>
                                                    <Button
                                                        variant="outline"
                                                        size="icon"
                                                        className="h-8 w-8 text-green-600 border-green-600 hover:bg-green-50"
                                                        onClick={() => handleApprove(appointment.id)}
                                                    >
                                                        <Check className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="outline"
                                                        size="icon"
                                                        className="h-8 w-8 text-red-600 border-red-600 hover:bg-red-50"
                                                        onClick={() => handleReject(appointment.id)}
                                                    >
                                                        <X className="h-4 w-4" />
                                                    </Button>
                                                </>
                                            )}
                                            <Button variant="ghost" size="icon" className="h-8 w-8">
                                                <Pencil className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>
                <Card>
                <Link href="/">

                    <button className="px-4 py-2 bg-blue-500 text-white rounded-md">
                        Go to Dashboard
                    </button>

                </Link>
            </Card>
            </div>
            
        </Layout>
    )
}


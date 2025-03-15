"use client"

import { useState, useEffect } from "react"
import { Pencil, Check, X, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import Link from "next/link"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { doctorService, Doctor, PatientChat } from "../api"
import { ChatModal } from "./ChatModal"

// Fixed doctor ID
const DOCTOR_ID = 720;

export default function Page() {
    // State for doctor data
    const [doctor, setDoctor] = useState<Doctor | null>(null)
    const [loading, setLoading] = useState<boolean>(true)
    const [error, setError] = useState<string | null>(null)
    const [appointmentsLoading, setAppointmentsLoading] = useState<boolean>(true)
    const [appointmentsError, setAppointmentsError] = useState<string | null>(null)

    // State for appointments
    const [appointments, setAppointments] = useState<{
        id: number;
        status: string;
        title: string;
        date: string;
        patientName: string;
        reason: string;
        requestedTime: string;
        patientId?: number;
    }[]>([
        {
            id: 1,
            status: "pending",
            title: "Annual checkup - John Smith",
            date: "2022-02-18 14:09:54",
            patientName: "John Smith",
            reason: "Annual physical examination",
            requestedTime: "10:00 AM - 11:00 AM",
            patientId: 535,
        },
        {
            id: 2,
            status: "pending",
            title: "Follow-up consultation - Sarah Johnson",
            date: "2022-02-18 12:47:08",
            patientName: "Sarah Johnson",
            reason: "Follow-up on medication",
            requestedTime: "1:30 PM - 2:00 PM",
            patientId: 536,
        },
        {
            id: 3,
            status: "pending",
            title: "New patient consultation - Michael Brown",
            date: "2022-02-18 11:27:54",
            patientName: "Michael Brown",
            reason: "Persistent headaches",
            requestedTime: "3:15 PM - 4:00 PM",
            patientId: 537,
        },
    ])

    const [sortOption, setSortOption] = useState("date-desc")
    
    // Chat modal state
    const [isChatModalOpen, setIsChatModalOpen] = useState(false)
    const [selectedChat, setSelectedChat] = useState<{
        patientId: number;
        patientName: string;
        title: string;
    } | null>(null)

    // Function to fetch doctor data
    const fetchDoctorData = async () => {
        setLoading(true)
        setError(null)

        try {
            const doctorData = await doctorService.getDoctorById(DOCTOR_ID)
            if (doctorData) {
                setDoctor(doctorData)
                // After fetching doctor data, fetch their appointments
                fetchDoctorAppointments(DOCTOR_ID)
            } else {
                setError("Doctor not found")
            }
        } catch (err) {
            setError("Failed to fetch doctor data")
            console.error("Error fetching doctor:", err)
        } finally {
            setLoading(false)
        }
    }

    // Function to fetch doctor appointments
    const fetchDoctorAppointments = async (docId: number) => {
        setAppointmentsLoading(true)
        setAppointmentsError(null)

        try {
            const appointmentsData = await doctorService.getDoctorAppointments(docId)
            if (appointmentsData && appointmentsData.length > 0) {
                // Transform the appointments data to match our format
                const formattedAppointments = appointmentsData.map((chat) => ({
                    id: chat.chatId,
                    status: chat.status || "pending", // Default to pending if not specified
                    title: chat.title || `Chat ID: ${chat.chatId}`,
                    date: chat.timestamp || new Date().toISOString(),
                    patientName: chat.patientName || `Patient ID: ${chat.patientId}`,
                    reason: chat.reason || "No reason provided",
                    requestedTime: chat.requestedTime || "Time not specified",
                    patientId: chat.patientId,
                }))
                setAppointments(formattedAppointments)
            } else {
                setAppointmentsError("No appointments found for this doctor")
                // Keep the demo appointments as a fallback
            }
        } catch (err) {
            setAppointmentsError("Failed to fetch doctor appointments")
            console.error("Error fetching appointments:", err)
        } finally {
            setAppointmentsLoading(false)
        }
    }

    // Load data when component mounts
    useEffect(() => {
        fetchDoctorData();
    }, []);

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

    const handleApprove = async (id: number) => {
        try {
            // Call the API to update approval status
            await doctorService.updateDoctorApproval(id, 1);
            // Update the local state to reflect the change
            setAppointments(appointments.map((app) => (app.id === id ? { ...app, status: "approved" } : app)));
        } catch (error) {
            console.error("Error approving appointment:", error);
            alert("Failed to approve appointment. Please try again.");
        }
    }

    const handleReject = async (id: number) => {
        try {
            // Call the API to update approval status
            await doctorService.updateDoctorApproval(id, 0);
            // Update the local state to reflect the change
            setAppointments(appointments.map((app) => (app.id === id ? { ...app, status: "rejected" } : app)));
        } catch (error) {
            console.error("Error rejecting appointment:", error);
            alert("Failed to reject appointment. Please try again.");
        }
    }

    // Function to handle opening the chat modal
    const handleOpenChatModal = (appointment: {
        patientId?: number;
        patientName: string;
        title: string;
    }) => {
        if (!appointment.patientId) {
            alert("Patient ID not available. Cannot view chat history.");
            return;
        }
        
        setSelectedChat({
            patientId: appointment.patientId,
            patientName: appointment.patientName,
            title: appointment.title,
        });
        setIsChatModalOpen(true);
    }

    return (
        <div className="container mx-auto p-6">
            <div className="grid gap-6">
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base font-medium">Doctor Profile</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {error && (
                            <div className="text-red-500 text-sm mb-4">
                                {error}
                            </div>
                        )}

                        {loading ? (
                            <div className="text-center py-4">
                                <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2 text-blue-500" />
                                <p className="text-gray-600">Loading doctor information...</p>
                            </div>
                        ) : doctor ? (
                            <div>
                                <div className="flex items-center gap-4 mb-4">
                                    <div className="h-24 w-24 overflow-hidden rounded-md">
                                        <img 
                                            src="/doctorimage.jpg" 
                                            alt={doctor.name || "Doctor profile"} 
                                            className="h-full w-full object-cover"
                                        />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-semibold">{doctor.name}</h2>
                                        <p className="text-gray-600">{doctor.specialty}</p>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Experience</span>
                                        <span>{doctor.experience_years} years</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Doctor ID</span>
                                        <span>{doctor.doctorId}</span>
                                    </div>
                                    <div className="flex justify-between col-span-2">
                                        <span className="text-gray-600">Available Hours</span>
                                        <span>{doctor.available_hours}</span>
                                    </div>
                                    <div className="flex justify-between col-span-2">
                                        <span className="text-gray-600">Contact</span>
                                        <span>{doctor.doctorContact}</span>
                                    </div>
                                    <div className="flex justify-between col-span-2">
                                        <span className="text-gray-600">Description</span>
                                        <span>{doctor.description}</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-4 text-gray-500">
                                No doctor information available
                            </div>
                        )}
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

                            {appointmentsLoading && (
                                <div className="text-center py-8">
                                    <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2 text-blue-500" />
                                    <p className="text-gray-600">Loading appointments...</p>
                                </div>
                            )}

                            {appointmentsError && (
                                <div className="text-red-500 text-sm mb-4">
                                    {appointmentsError}
                                </div>
                            )}

                            <div className="border rounded-lg">
                                {getSortedAppointments().length > 0 ? (
                                    getSortedAppointments().map((appointment) => (
                                        <div
                                            key={appointment.id}
                                            className="grid grid-cols-[auto_1fr_auto] items-center gap-4 p-3 hover:bg-gray-50 border-t first:border-t-0"
                                        >
                                            <Badge
                                                variant={
                                                    appointment.status === "pending"
                                                        ? "outline"
                                                        : appointment.status === "approved"
                                                            ? "default" 
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
                                                <Button 
                                                    variant="ghost" 
                                                    size="icon" 
                                                    className="h-8 w-8"
                                                    onClick={() => handleOpenChatModal(appointment)}
                                                >
                                                    <Pencil className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-center py-8 text-gray-500">
                                        {appointmentsLoading ? "Loading..." : "No appointments found for this doctor"}
                                    </div>
                                )}
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Chat Modal */}
            {selectedChat && (
                <ChatModal
                    open={isChatModalOpen}
                    onOpenChange={setIsChatModalOpen}
                    doctorId={DOCTOR_ID}
                    patientId={selectedChat.patientId}
                    patientName={selectedChat.patientName}
                    appointmentTitle={selectedChat.title}
                />
            )}
        </div>
    )
}


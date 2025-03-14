"use client"

import { useState } from "react"
import { Search, Phone, Video, MessageSquare, MoreHorizontal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Layout from "../layout1"

type PatientChat = {
  id: string
  patientName: string
  patientInitials: string
  patientImage?: string
  lastMessage: string
  lastMessageTime: string
  unreadCount: number
  hasVideoHistory: boolean
  status: "active" | "completed" | "scheduled"
}

export default function PatientsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [patientChats, setPatientChats] = useState<PatientChat[]>([
    {
      id: "1",
      patientName: "John Smith",
      patientInitials: "JS",
      patientImage: "/placeholder.svg?height=40&width=40",
      lastMessage: "Thank you doctor, I'll follow your advice on the medication schedule.",
      lastMessageTime: "10:32 AM",
      unreadCount: 3,
      hasVideoHistory: true,
      status: "active",
    },
    {
      id: "2",
      patientName: "Sarah Johnson",
      patientInitials: "SJ",
      patientImage: "/placeholder.svg?height=40&width=40",
      lastMessage: "When should I schedule my next follow-up appointment?",
      lastMessageTime: "Yesterday",
      unreadCount: 0,
      hasVideoHistory: true,
      status: "active",
    },
    {
      id: "3",
      patientName: "Michael Brown",
      patientInitials: "MB",
      lastMessage: "The new prescription is working much better for my headaches.",
      lastMessageTime: "2 days ago",
      unreadCount: 0,
      hasVideoHistory: false,
      status: "active",
    },
    {
      id: "4",
      patientName: "Emily Davis",
      patientInitials: "ED",
      patientImage: "/placeholder.svg?height=40&width=40",
      lastMessage: "Video consultation completed. Follow-up in 2 weeks.",
      lastMessageTime: "Mar 15",
      unreadCount: 0,
      hasVideoHistory: true,
      status: "completed",
    },
    {
      id: "5",
      patientName: "Robert Wilson",
      patientInitials: "RW",
      lastMessage: "Video consultation scheduled for March 25, 2:00 PM",
      lastMessageTime: "Mar 14",
      unreadCount: 1,
      hasVideoHistory: false,
      status: "scheduled",
    },
    {
      id: "6",
      patientName: "Jennifer Martinez",
      patientInitials: "JM",
      patientImage: "/placeholder.svg?height=40&width=40",
      lastMessage: "I've uploaded my blood pressure readings for the past week.",
      lastMessageTime: "Mar 12",
      unreadCount: 0,
      hasVideoHistory: true,
      status: "active",
    },
  ])

  const filteredChats = patientChats.filter(
    (chat) =>
      chat.patientName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      chat.lastMessage.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-green-500"
      case "completed":
        return "bg-gray-500"
      case "scheduled":
        return "bg-blue-500"
      default:
        return "bg-gray-500"
    }
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Patient Communications</h1>
          <Button>
            <Phone className="mr-2 h-4 w-4" />
            New Consultation
          </Button>
        </div>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-medium">Patient Chats & Video Calls</CardTitle>
              <div className="relative w-64">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search patients..."
                  className="pl-8"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="all">
              <TabsList className="mb-4">
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="active">Active</TabsTrigger>
                <TabsTrigger value="completed">Completed</TabsTrigger>
                <TabsTrigger value="scheduled">Scheduled</TabsTrigger>
              </TabsList>

              <TabsContent value="all" className="space-y-0 border rounded-md">
                {filteredChats.length > 0 ? (
                  filteredChats.map((chat) => (
                    <div
                      key={chat.id}
                      className="flex items-start gap-4 p-4 hover:bg-gray-50 border-b last:border-b-0 cursor-pointer"
                    >
                      <div className="relative">
                        <Avatar className="h-10 w-10">
                          <AvatarImage src={chat.patientImage} alt={chat.patientName} />
                          <AvatarFallback>{chat.patientInitials}</AvatarFallback>
                        </Avatar>
                        <span
                          className={`absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-white ${getStatusColor(chat.status)}`}
                        />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div className="font-medium">{chat.patientName}</div>
                          <div className="text-xs text-gray-500">{chat.lastMessageTime}</div>
                        </div>
                        <div className="text-sm text-gray-600 truncate">{chat.lastMessage}</div>
                        <div className="flex items-center gap-2 mt-1">
                          {chat.hasVideoHistory && (
                            <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                              <Video className="h-3 w-3 mr-1" />
                              Video
                            </Badge>
                          )}
                          <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                            <MessageSquare className="h-3 w-3 mr-1" />
                            Chat
                          </Badge>
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        {chat.unreadCount > 0 && (
                          <Badge className="h-5 w-5 flex items-center justify-center p-0 rounded-full">
                            {chat.unreadCount}
                          </Badge>
                        )}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>
                              <MessageSquare className="mr-2 h-4 w-4" />
                              <span>Send message</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Video className="mr-2 h-4 w-4" />
                              <span>Start video call</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Phone className="mr-2 h-4 w-4" />
                              <span>Start audio call</span>
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="p-8 text-center text-gray-500">No patients found matching your search.</div>
                )}
              </TabsContent>

              <TabsContent value="active" className="space-y-0 border rounded-md">
                {filteredChats
                  .filter((chat) => chat.status === "active")
                  .map((chat) => (
                    /* Same chat item structure as above */
                    <div
                      key={chat.id}
                      className="flex items-start gap-4 p-4 hover:bg-gray-50 border-b last:border-b-0 cursor-pointer"
                    >
                      <div className="relative">
                        <Avatar className="h-10 w-10">
                          <AvatarImage src={chat.patientImage} alt={chat.patientName} />
                          <AvatarFallback>{chat.patientInitials}</AvatarFallback>
                        </Avatar>
                        <span
                          className={`absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-white ${getStatusColor(chat.status)}`}
                        />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div className="font-medium">{chat.patientName}</div>
                          <div className="text-xs text-gray-500">{chat.lastMessageTime}</div>
                        </div>
                        <div className="text-sm text-gray-600 truncate">{chat.lastMessage}</div>
                        <div className="flex items-center gap-2 mt-1">
                          {chat.hasVideoHistory && (
                            <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                              <Video className="h-3 w-3 mr-1" />
                              Video
                            </Badge>
                          )}
                          <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                            <MessageSquare className="h-3 w-3 mr-1" />
                            Chat
                          </Badge>
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        {chat.unreadCount > 0 && (
                          <Badge className="h-5 w-5 flex items-center justify-center p-0 rounded-full">
                            {chat.unreadCount}
                          </Badge>
                        )}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>
                              <MessageSquare className="mr-2 h-4 w-4" />
                              <span>Send message</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Video className="mr-2 h-4 w-4" />
                              <span>Start video call</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Phone className="mr-2 h-4 w-4" />
                              <span>Start audio call</span>
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))}
              </TabsContent>

              <TabsContent value="completed" className="space-y-0 border rounded-md">
                {filteredChats
                  .filter((chat) => chat.status === "completed")
                  .map((chat) => (
                    /* Same chat item structure as above */
                    <div
                      key={chat.id}
                      className="flex items-start gap-4 p-4 hover:bg-gray-50 border-b last:border-b-0 cursor-pointer"
                    >
                      <div className="relative">
                        <Avatar className="h-10 w-10">
                          <AvatarImage src={chat.patientImage} alt={chat.patientName} />
                          <AvatarFallback>{chat.patientInitials}</AvatarFallback>
                        </Avatar>
                        <span
                          className={`absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-white ${getStatusColor(chat.status)}`}
                        />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div className="font-medium">{chat.patientName}</div>
                          <div className="text-xs text-gray-500">{chat.lastMessageTime}</div>
                        </div>
                        <div className="text-sm text-gray-600 truncate">{chat.lastMessage}</div>
                        <div className="flex items-center gap-2 mt-1">
                          {chat.hasVideoHistory && (
                            <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                              <Video className="h-3 w-3 mr-1" />
                              Video
                            </Badge>
                          )}
                          <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                            <MessageSquare className="h-3 w-3 mr-1" />
                            Chat
                          </Badge>
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        {chat.unreadCount > 0 && (
                          <Badge className="h-5 w-5 flex items-center justify-center p-0 rounded-full">
                            {chat.unreadCount}
                          </Badge>
                        )}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>
                              <MessageSquare className="mr-2 h-4 w-4" />
                              <span>Send message</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Video className="mr-2 h-4 w-4" />
                              <span>Start video call</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Phone className="mr-2 h-4 w-4" />
                              <span>Start audio call</span>
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))}
              </TabsContent>

              <TabsContent value="scheduled" className="space-y-0 border rounded-md">
                {filteredChats
                  .filter((chat) => chat.status === "scheduled")
                  .map((chat) => (
                    /* Same chat item structure as above */
                    <div
                      key={chat.id}
                      className="flex items-start gap-4 p-4 hover:bg-gray-50 border-b last:border-b-0 cursor-pointer"
                    >
                      <div className="relative">
                        <Avatar className="h-10 w-10">
                          <AvatarImage src={chat.patientImage} alt={chat.patientName} />
                          <AvatarFallback>{chat.patientInitials}</AvatarFallback>
                        </Avatar>
                        <span
                          className={`absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-white ${getStatusColor(chat.status)}`}
                        />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div className="font-medium">{chat.patientName}</div>
                          <div className="text-xs text-gray-500">{chat.lastMessageTime}</div>
                        </div>
                        <div className="text-sm text-gray-600 truncate">{chat.lastMessage}</div>
                        <div className="flex items-center gap-2 mt-1">
                          {chat.hasVideoHistory && (
                            <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                              <Video className="h-3 w-3 mr-1" />
                              Video
                            </Badge>
                          )}
                          <Badge variant="outline" className="text-xs px-1 py-0 h-5 bg-gray-100">
                            <MessageSquare className="h-3 w-3 mr-1" />
                            Chat
                          </Badge>
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        {chat.unreadCount > 0 && (
                          <Badge className="h-5 w-5 flex items-center justify-center p-0 rounded-full">
                            {chat.unreadCount}
                          </Badge>
                        )}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>
                              <MessageSquare className="mr-2 h-4 w-4" />
                              <span>Send message</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Video className="mr-2 h-4 w-4" />
                              <span>Start video call</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Phone className="mr-2 h-4 w-4" />
                              <span>Start audio call</span>
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </Layout>
  )
}


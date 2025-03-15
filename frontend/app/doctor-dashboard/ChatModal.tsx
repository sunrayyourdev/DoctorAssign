"use client"

import { useState, useEffect, useRef } from "react"
import { X, User, Stethoscope, RefreshCw, Send, AlertCircle, FileText } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Avatar } from "@/components/ui/avatar"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { chatService } from "../api"

interface MessageType {
  id: string
  content: string
  sender: "user" | "system"
}

interface ChatData {
  chatId: number
  messages: any[]
  responses: any[]
  extracted_symptoms: string[]
}

interface ChatModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  doctorId: number
  patientId: number
  patientName: string
  appointmentTitle: string
}

export function ChatModal({
  open,
  onOpenChange,
  doctorId,
  patientId,
  patientName,
  appointmentTitle,
}: ChatModalProps) {
  const [chatData, setChatData] = useState<ChatData | null>(null)
  const [messages, setMessages] = useState<MessageType[]>([])
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState<string>("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Fetch chat history when modal opens
  useEffect(() => {
    if (open) {
      fetchChatHistory()
    }
  }, [open, doctorId, patientId])

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const fetchChatHistory = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const data = await chatService.getPatientChat(doctorId, patientId)
      
      if (data.error) {
        setError(data.error)
        return
      }

      // Store the complete chat data
      setChatData({
        chatId: data.chatId,
        messages: data.messages || [],
        responses: data.responses || [],
        extracted_symptoms: data.extracted_symptoms || []
      })

      // Convert the messages and responses to our message format
      const formattedMessages: MessageType[] = []
      
      // Process patient messages
      if (data.messages && Array.isArray(data.messages)) {
        data.messages.forEach((msg: any, index: number) => {
          formattedMessages.push({
            id: `user-${index}`,
            content: typeof msg === 'object' ? msg.content : msg,
            sender: "user"
          })

          // If there's a corresponding response, add it too
          if (data.responses && data.responses[index]) {
            formattedMessages.push({
              id: `system-${index}`,
              content: typeof data.responses[index] === 'object' 
                ? data.responses[index].content 
                : data.responses[index],
              sender: "system"
            })
          }
        })
      }

      setMessages(formattedMessages)
    } catch (err) {
      console.error("Error fetching chat history:", err)
      setError("Failed to load chat history. Please try again later.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!inputValue.trim()) return
    
    // Add message to chat (in a real app, you'd send this to an API)
    const newMessage: MessageType = {
      id: `doctor-${Date.now()}`,
      content: inputValue,
      sender: "system" // Doctor messages are sent as "system" in this UI
    }
    
    setMessages([...messages, newMessage])
    setInputValue("")
  }

  // Loading and error states
  const renderLoadingOrError = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="flex flex-col items-center">
            <RefreshCw className="h-8 w-8 animate-spin text-primary mb-2" />
            <p className="text-sm text-muted-foreground">Loading chat history...</p>
          </div>
        </div>
      )
    }
    
    if (error) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="bg-red-50 text-red-700 p-4 rounded-md max-w-md text-sm">
            <p className="font-medium mb-1">Error loading chat</p>
            <p>{error}</p>
            <Button
              variant="outline" 
              size="sm" 
              className="mt-2"
              onClick={fetchChatHistory}
            >
              Try Again
            </Button>
          </div>
        </div>
      )
    }
    
    return null
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px] md:max-w-[700px] lg:max-w-[800px] h-[600px] p-0 flex flex-col">
        <DialogHeader className="p-4 border-b">
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="text-lg font-semibold">{patientName}</DialogTitle>
              <p className="text-sm text-muted-foreground">{appointmentTitle}</p>
              {chatData && (
                <Badge variant="outline" className="mt-1">
                  Chat ID: {chatData.chatId}
                </Badge>
              )}
            </div>
            <DialogClose asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full">
                <X className="h-4 w-4" />
              </Button>
            </DialogClose>
          </div>
        </DialogHeader>

        {/* Main content area with tabs */}
        <Tabs defaultValue="chat" className="flex-1 flex flex-col">
          <div className="border-b px-4">
            <TabsList>
              <TabsTrigger value="chat" className="data-[state=active]:bg-primary/10">
                Chat History
              </TabsTrigger>
              <TabsTrigger value="symptoms" className="data-[state=active]:bg-primary/10">
                Extracted Symptoms
              </TabsTrigger>
              <TabsTrigger value="raw" className="data-[state=active]:bg-primary/10">
                Raw Data
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Chat tab */}
          <TabsContent value="chat" className="flex-1 p-0 m-0 data-[state=active]:flex data-[state=active]:flex-col">
            <ScrollArea className="flex-1 p-4">
              {isLoading || error ? (
                renderLoadingOrError()
              ) : messages.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground text-sm">No messages found for this patient.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message) => (
                    <div key={message.id} className={`flex ${message.sender === "user" ? "justify-start" : "justify-end"}`}>
                      <div className={`flex items-start gap-2 max-w-[80%] ${message.sender === "user" ? "flex-row" : "flex-row-reverse"}`}>
                        <Avatar className={`mt-1 h-8 w-8 ${message.sender === "user" ? "bg-gray-200" : "bg-primary"}`}>
                          {message.sender === "user" ? (
                            <User className="h-4 w-4 text-gray-600" />
                          ) : (
                            <Stethoscope className="h-4 w-4 text-white" />
                          )}
                        </Avatar>

                        <div className={`rounded-lg p-3 text-sm ${
                          message.sender === "user" 
                            ? "bg-gray-100" 
                            : "bg-primary text-white"
                        }`}>
                          {message.content}
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </ScrollArea>

            {/* Input area */}
            <div className="p-4 border-t mt-auto">
              <form onSubmit={handleSendMessage} className="flex gap-2">
                <Input
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Type your response..."
                  className="flex-1"
                />
                <Button type="submit" size="icon">
                  <Send className="h-4 w-4" />
                </Button>
              </form>
            </div>
          </TabsContent>

          {/* Symptoms tab */}
          <TabsContent value="symptoms" className="flex-1 p-4 m-0 overflow-auto">
            {isLoading || error ? (
              renderLoadingOrError()
            ) : !chatData || !chatData.extracted_symptoms || chatData.extracted_symptoms.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full">
                <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
                <p className="text-muted-foreground text-sm">No symptoms were extracted from this conversation.</p>
              </div>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-amber-500" />
                    Extracted Symptoms
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {chatData.extracted_symptoms.map((symptom, index) => (
                      <Badge key={index} variant="secondary" className="px-3 py-1.5 text-sm">
                        {symptom}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Raw Data tab */}
          <TabsContent value="raw" className="flex-1 p-4 m-0 overflow-auto">
            {isLoading || error ? (
              renderLoadingOrError()
            ) : !chatData ? (
              <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground text-sm">No data available.</p>
              </div>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Raw API Response
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="bg-gray-50 p-4 rounded-md text-xs overflow-auto max-h-[400px]">
                    {JSON.stringify(chatData, null, 2)}
                  </pre>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
} 
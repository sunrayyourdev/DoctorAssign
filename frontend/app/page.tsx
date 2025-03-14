"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, User, Stethoscope } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar } from "@/components/ui/avatar"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

// Types for our messages and doctor
type MessageType = {
  id: string
  content: string
  sender: "user" | "system"
  isDoctor?: boolean
  doctor?: DoctorType
}

type DoctorType = {
  id: string
  name: string
  specialty: string
  experience: string
  rating: number
  image: string
}

// Mock doctor data
const doctors: DoctorType[] = [
  {
    id: "1",
    name: "Dr. Sarah Johnson",
    specialty: "Cardiologist",
    experience: "12 years",
    rating: 4.9,
    image: "/placeholder.svg?height=80&width=80",
  },
  {
    id: "2",
    name: "Dr. Michael Chen",
    specialty: "Neurologist",
    experience: "15 years",
    rating: 4.8,
    image: "/placeholder.svg?height=80&width=80",
  },
  {
    id: "3",
    name: "Dr. Emily Rodriguez",
    specialty: "Dermatologist",
    experience: "8 years",
    rating: 4.7,
    image: "/placeholder.svg?height=80&width=80",
  },
]

// Simple symptom to specialty matching
const matchDoctorBySymptoms = (symptoms: string): DoctorType => {
  const symptomLower = symptoms.toLowerCase()

  if (
    symptomLower.includes("heart") ||
    symptomLower.includes("chest pain") ||
    symptomLower.includes("blood pressure")
  ) {
    return doctors[0]
  } else if (symptomLower.includes("head") || symptomLower.includes("migraine") || symptomLower.includes("dizzy")) {
    return doctors[1]
  } else {
    return doctors[2]
  }
}

export default function DoctorMatchingChat() {
  const [messages, setMessages] = useState<MessageType[]>([
    {
      id: "1",
      content: "Hello! Please describe your symptoms so I can match you with the right doctor.",
      sender: "system",
    },
  ])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim()) return

    // Add user message
    const userMessage: MessageType = {
      id: Date.now().toString(),
      content: input,
      sender: "user",
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsTyping(true)

    // Simulate processing time
    setTimeout(() => {
      setIsTyping(false)

      // Match doctor based on symptoms
      const matchedDoctor = matchDoctorBySymptoms(input)

      // Add system response with doctor card
      const systemResponse: MessageType = {
        id: (Date.now() + 1).toString(),
        content: `Based on your symptoms, I've found a specialist who can help you:`,
        sender: "system",
        isDoctor: true,
        doctor: matchedDoctor,
      }

      setMessages((prev) => [...prev, systemResponse])
    }, 1500)
  }

  return (
    <div className="flex flex-col mx-auto p-2 sm:p-4 h-[500px] sm:h-[600px] w-full max-w-3xl">
      <div className="bg-white rounded-lg shadow-lg flex flex-col h-full">
        {/* Chat header */}
        <div className="p-2 sm:p-3 border-b">
          <h1 className="text-base sm:text-lg font-semibold text-primary">DoctorAssign</h1>
          <p className="text-xs text-muted-foreground">How are you feeling today? Describe your symptoms and we'll find the best doctor for you.</p>
        </div>

        {/* Messages container */}
        <div className="flex-1 overflow-y-auto p-2 sm:p-3 space-y-2 sm:space-y-3">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`flex items-start gap-1 sm:gap-2 max-w-[90%] sm:max-w-[80%] ${message.sender === "user" ? "flex-row-reverse" : "flex-row"}`}
              >
                <Avatar className="mt-1 h-6 w-6 sm:h-8 sm:w-8">
                  {message.sender === "user" ? (
                    <User className="h-4 w-4 sm:h-5 sm:w-5" />
                  ) : (
                    <Stethoscope className="h-4 w-4 sm:h-5 sm:w-5" />
                  )}
                </Avatar>

                <div className="w-full">
                  {/* Regular message bubble */}
                  <div
                    className={`rounded-lg p-2 text-sm ${
                      message.sender === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                    }`}
                  >
                    {message.content}
                  </div>

                  {/* Doctor card (if applicable) */}
                  {message.isDoctor && message.doctor && (
                    <Card className="mt-2 p-2 sm:p-3 border border-blue-100 shadow-sm">
                      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-3">
                        <Avatar className="h-10 w-10 sm:h-12 sm:w-12 rounded-md">
                          <img src={message.doctor.image || "/placeholder.svg"} alt={message.doctor.name} />
                        </Avatar>

                        <div className="flex-1">
                          <h3 className="font-semibold text-sm sm:text-base">{message.doctor.name}</h3>
                          <p className="text-xs text-muted-foreground">{message.doctor.specialty}</p>
                          <div className="flex items-center gap-1 mt-0.5">
                            <div className="flex">
                              {[...Array(5)].map((_, i) => (
                                <span
                                  key={i}
                                  className={`text-xs ${i < Math.floor(message.doctor.rating) ? "text-yellow-500" : "text-gray-300"}`}
                                >
                                  ★
                                </span>
                              ))}
                            </div>
                            <span className="text-xs font-medium">{message.doctor.rating}</span>
                            <span className="text-xs text-muted-foreground">• {message.doctor.experience}</span>
                          </div>
                        </div>

                        <Button size="sm" className="bg-green-600 hover:bg-green-700 mt-1 sm:mt-0 w-full sm:w-auto">
                          Contact
                        </Button>
                      </div>
                    </Card>
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {isTyping && (
            <div className="flex justify-start">
              <div className="flex items-start gap-1 sm:gap-2">
                <Avatar className="mt-1 h-6 w-6 sm:h-8 sm:w-8">
                  <Stethoscope className="h-4 w-4 sm:h-5 sm:w-5" />
                </Avatar>
                <div className="bg-muted rounded-lg p-2">
                  <div className="flex space-x-1">
                    <div className="h-1.5 w-1.5 sm:h-2 sm:w-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="h-1.5 w-1.5 sm:h-2 sm:w-2 bg-gray-400 rounded-full animate-bounce delay-75"></div>
                    <div className="h-1.5 w-1.5 sm:h-2 sm:w-2 bg-gray-400 rounded-full animate-bounce delay-150"></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="p-2 sm:p-3 border-t">
          <form onSubmit={handleSendMessage} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe your symptoms..."
              className="flex-1 text-sm"
            />
            <Button type="submit" size="sm" className="px-2 sm:px-3">
              <Send className="h-3 w-3 sm:h-4 sm:w-4" />
            </Button>
          </form>
        </div>
      </div>
    </div>
  )
}


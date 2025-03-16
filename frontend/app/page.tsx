"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Send, User, Stethoscope, Activity, AlertCircle, RotateCw, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar } from "@/components/ui/avatar"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import Link from "next/link"
import ErrorBoundary from "@/components/error-boundary"

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

// API response type
type ApiResponse = {
  response: string;
  chat_history: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
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

// API URL
const API_URL = "http://127.0.0.1:5010/chatbot_response";
const RECOMMEND_DOCTOR_URL = "http://127.0.0.1:5010/recommend_doctor";

// Default patient ID - this is the only ID guaranteed to work with the API
const DEFAULT_PATIENT_ID = 535;

// Tab types
type TabType = "chat" | "doctor";

export default function DoctorMatchingChat() {
  const [messages, setMessages] = useState<MessageType[]>([
    {
      id: "1",
      content: "Hello! Please describe your symptoms in detail so I can match you with the right doctor. Be specific about what you're experiencing, when it started, and how severe it is.",
      sender: "system",
    },
    {
      id: "2",
      content: "For example, instead of just saying 'headache', you might say 'I've been experiencing throbbing headaches on the right side for about a week, especially in the morning. Over-the-counter pain relievers don't seem to help much.'",
      sender: "system",
    }
  ])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [patientId, setPatientId] = useState<number>(DEFAULT_PATIENT_ID)
  const [showSettings, setShowSettings] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<TabType>("chat")
  const [isRecommending, setIsRecommending] = useState<boolean>(false)
  const [recommendationError, setRecommendationError] = useState<string | null>(null)
  const [recommendedDoctor, setRecommendedDoctor] = useState<{
    doctor_details: any;
    chatgpt_analysis: string;
  } | null>(null)
  const [consultationRequested, setConsultationRequested] = useState<boolean>(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Function to reset doctor recommendation and consultation state
  const handleReset = useCallback(() => {
    setRecommendedDoctor(null);
    setConsultationRequested(false);
  }, []);

  // Function to convert chat_history array to our MessageType format
  const convertChatHistoryToMessages = (chatHistory: Array<{role: string, content: string}>): MessageType[] => {
    if (!chatHistory || !Array.isArray(chatHistory)) return [];
    
    return chatHistory.map((msg, index) => ({
      id: `history-${index}`,
      content: msg.content,
      sender: msg.role === "user" ? "user" : "system",
    }));
  }

  const handleSendMessage = async (e: React.FormEvent) => {
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
    setApiError(null) // Reset any previous errors

    try {
      // Call the backend API with the correct format
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          patientId: patientId, 
          content: input 
        }),
      });

      if (!response.ok) {
        // Check if it might be a patient ID issue
        if (response.status === 400 || response.status === 404) {
          if (patientId !== DEFAULT_PATIENT_ID) {
            throw new Error(`API Error: Patient ID ${patientId} may not be valid. Try using ID ${DEFAULT_PATIENT_ID} instead.`);
          }
        }
        throw new Error(`API Error: ${response.status}`);
      }

      const data: ApiResponse = await response.json();

      // Option 1: Just add the latest response
      const systemResponse: MessageType = {
        id: Date.now().toString(),
        content: data.response,
        sender: "system",
      }
      
      setIsTyping(false)
      setMessages((prev) => [...prev, systemResponse])
      
      // Option 2 (alternative): Replace all messages with the complete chat history
      // Use this approach if you want the frontend to exactly match the backend's chat history
      // Uncomment the line below and comment out the 3 lines above if you prefer this approach
      // setMessages(convertChatHistoryToMessages(data.chat_history));
    } catch (error) {
      console.error("Error calling API:", error);
      setIsTyping(false)
      
      // Show error message
      const errorMessage: MessageType = {
        id: Date.now().toString(),
        content: error instanceof Error ? error.message : "Sorry, I'm having trouble connecting to our system. Please try again later.",
        sender: "system",
      }
      
      setMessages((prev) => [...prev, errorMessage])
      
      if (patientId !== DEFAULT_PATIENT_ID && error instanceof Error && (error.message.includes("Patient ID") || error.message.includes("400") || error.message.includes("404"))) {
        setApiError(`The patient ID ${patientId} may not be valid. Please use the Settings button to change it to ${DEFAULT_PATIENT_ID}.`);
      } else {
        setApiError("Failed to connect to the API server. Please check that the backend is running.");
      }
    }
  }

  const handleRecommendDoctor = async () => {
    if (messages.length <= 1) {
      setRecommendationError("Please chat about your symptoms first before requesting a doctor recommendation.");
      return;
    }

    setIsRecommending(true);
    setRecommendationError(null);
    setRecommendedDoctor(null);

    try {
      const response = await fetch(RECOMMEND_DOCTOR_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patientId: patientId
        }),
      });

      // Get raw response text first to inspect the message
      const responseText = await response.text();
      
      // Try to parse as JSON
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        console.error("Failed to parse JSON response:", responseText);
        throw new Error("Invalid response from server");
      }
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Not enough information provided. Please describe your symptoms in more detail.");
        }
        throw new Error(data.error || `Error: ${response.status}`);
      }
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setRecommendedDoctor(data);
    } catch (error) {
      console.error("Error recommending doctor:", error);
      
      // Handle the specific "not enough data" error
      const errorMessage = error instanceof Error ? error.message : "Failed to recommend a doctor";
      
      if (errorMessage.includes("Not enough data reported") || 
          errorMessage.includes("not enough information")) {
        setRecommendationError("We need more specific details about your symptoms to make a confident doctor recommendation. Please return to the chat and describe your symptoms more thoroughly.");
      } else {
        setRecommendationError(errorMessage);
      }
    } finally {
      setIsRecommending(false);
    }
  };

  return (
    <ErrorBoundary>
      <div className="flex flex-col mx-auto p-2 sm:p-4 h-[500px] sm:h-[600px] w-full max-w-3xl">
        {/* Header with API Connection Test Link and Tabs */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveTab("chat")}
              className={`px-4 py-2 rounded-t-lg text-sm font-medium ${
                activeTab === "chat"
                  ? "bg-white text-primary border-b-2 border-primary"
                  : "bg-gray-100 text-gray-600 hover:text-gray-800"
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => setActiveTab("doctor")}
              className={`px-4 py-2 rounded-t-lg text-sm font-medium ${
                activeTab === "doctor"
                  ? "bg-white text-primary border-b-2 border-primary"
                  : "bg-gray-100 text-gray-600 hover:text-gray-800"
              }`}
            >
              Doctor Recommendation
            </button>
          </div>
          
          <div className="flex space-x-2">
            <Button 
              variant="outline" 
              size="sm" 
              className="flex items-center gap-2"
              onClick={() => setShowSettings(!showSettings)}
            >
              <AlertCircle className="h-4 w-4" />
              <span>Settings</span>
            </Button>
            {/* <Link href="/api-test" passHref>
              <Button variant="outline" size="sm" className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                <span>Test API</span>
              </Button>
            </Link> */}
          </div>
        </div>
        
        {/* Settings Panel */}
        {showSettings && (
          <div className="mb-4 p-3 border rounded-md bg-gray-50 w-full">
            <h3 className="text-sm font-medium mb-2">Connection Settings</h3>
            
            {patientId !== DEFAULT_PATIENT_ID && (
              <div className="mb-3 p-2 bg-yellow-50 border border-yellow-100 rounded text-xs text-yellow-800">
                <p className="font-medium">⚠️ Warning: Using Non-Default Patient ID</p>
                <p>Only patient ID {DEFAULT_PATIENT_ID} is guaranteed to work with this API.</p>
              </div>
            )}
            
            <div className="flex flex-col gap-2">
              <div>
                <label htmlFor="patientId" className="block text-xs text-gray-600 mb-1">
                  Patient ID
                </label>
                <div className="flex gap-2">
                  <input
                    id="patientId"
                    type="number"
                    value={patientId}
                    onChange={(e) => setPatientId(Number(e.target.value) || DEFAULT_PATIENT_ID)}
                    className="flex-1 rounded border border-gray-300 px-2 py-1 text-sm"
                    min="1"
                  />
                  <Button 
                    size="sm" 
                    variant="outline" 
                    onClick={() => setPatientId(DEFAULT_PATIENT_ID)}
                    className="text-xs"
                  >
                    Reset
                  </Button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Used to identify the patient in API requests. Default: {DEFAULT_PATIENT_ID} (recommended)
                </p>
              </div>
            </div>
          </div>
        )}
        
        {/* API Error Alert */}
        {apiError && (
          <div className="mb-2 p-2 bg-red-100 border border-red-300 text-red-800 rounded-md text-xs">
            <p className="font-medium">Connection Error</p>
            <p>{apiError}</p>
          </div>
        )}
        
        <div className="bg-white rounded-lg shadow-lg flex flex-col h-full">
          {/* Chat header */}
          <div className="p-2 sm:p-3 border-b">
            <h1 className="text-base sm:text-lg font-semibold text-primary">DoctorAssign</h1>
            <p className="text-xs text-muted-foreground">
              {activeTab === "chat" 
                ? "How are you feeling today? Describe your symptoms so we can find the best doctor for you."
                : "Based on your symptoms, we'll recommend the most suitable doctor for you."}
            </p>
          </div>

          {activeTab === "chat" ? (
            <>
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
                                <img src={"/doctorimage.jpg"} alt={message.doctor.name} />
                              </Avatar>

                              <div className="flex-1">
                                <h3 className="font-semibold text-sm sm:text-base">{message.doctor.name}</h3>
                                <p className="text-xs text-muted-foreground">{message.doctor.specialty}</p>
                                <div className="flex items-center gap-1 mt-0.5">
                                  <div className="flex">
                                    {[...Array(5)].map((_, i) => (
                                      <span
                                        key={i}
                                        className={`text-xs ${i < Math.floor(message.doctor?.rating || 0) ? "text-yellow-500" : "text-gray-300"}`}
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
                <form onSubmit={handleSendMessage} className="flex flex-col gap-2">
                  <div className="text-xs text-gray-500 mb-1 px-1">
                    Describe your symptoms in detail for better doctor matching
                  </div>
                  <div className="flex gap-2">
                    <Input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Be specific about your symptoms, their severity, and duration..."
                      className="flex-1 text-sm"
                    />
                    <Button type="submit" size="sm" className="px-2 sm:px-3">
                      <Send className="h-3 w-3 sm:h-4 sm:w-4" />
                    </Button>
                  </div>
                </form>
              </div>
            </>
          ) : (
            /* Doctor Recommendation Tab */
            <div className="flex-1 overflow-y-auto p-4 sm:p-6">
              {recommendationError && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md text-sm">
                  <div className="flex items-start">
                    <AlertCircle className="h-5 w-5 text-red-500 mr-2 mt-0.5" />
                    <div>
                      <p className="font-medium text-red-800 mb-1">We couldn't generate a recommendation</p>
                      <p className="text-red-700">{recommendationError}</p>
                      
                      {recommendationError.includes("more specific details") && (
                        <div className="mt-3 p-3 bg-blue-50 border border-blue-100 rounded-md">
                          <p className="font-medium text-blue-800 mb-1">Try mentioning:</p>
                          <ul className="list-disc pl-5 text-sm text-blue-800">
                            <li>Specific symptoms (e.g., "sharp chest pain" instead of just "pain")</li>
                            <li>When the symptoms started and how they've changed</li>
                            <li>What makes symptoms better or worse</li>
                            <li>Any previous medical conditions or treatments</li>
                          </ul>
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="mt-2"
                            onClick={() => setActiveTab("chat")}
                          >
                            Return to Chat
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              {!recommendedDoctor && !isRecommending && (
                <div className="flex flex-col items-center justify-center space-y-4 h-full">
                  <div className="text-center max-w-md">
                    <Stethoscope className="h-12 w-12 mx-auto mb-4 text-primary" />
                    <h2 className="text-xl font-semibold mb-2">Get Doctor Recommendation</h2>
                    <p className="text-sm text-gray-600 mb-4">
                      Based on the symptoms you've shared in the chat, we can recommend the most suitable doctor for your needs.
                    </p>
                    
                    {/* Guidance message */}
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-md text-sm text-blue-800 mb-4">
                      <p className="font-medium mb-1">For best results, please ensure you've:</p>
                      <ul className="list-disc pl-5 text-left text-sm">
                        <li>Described your symptoms in detail</li>
                        <li>Mentioned how long you've experienced them</li>
                        <li>Shared any related medical history</li>
                        <li>Explained the severity of your symptoms</li>
                      </ul>
                    </div>
                    
                    {messages.length <= 1 ? (
                      <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-md text-sm text-yellow-800 mb-4">
                        <p>Please chat about your symptoms first before requesting a doctor recommendation.</p>
                      </div>
                    ) : null}
                    
                    <div className="flex flex-col gap-3">
                      <Button
                        onClick={() => setActiveTab("chat")}
                        variant="outline"
                        className="w-full"
                      >
                        Return to Chat
                      </Button>
                      <Button
                        onClick={handleRecommendDoctor}
                        disabled={isRecommending || messages.length <= 1}
                        className="w-full"
                      >
                        Find Me a Doctor
                      </Button>
                    </div>
                  </div>
                </div>
              )}
              
              {isRecommending && (
                <div className="flex flex-col items-center justify-center h-full space-y-4">
                  <div className="animate-spin">
                    <RotateCw className="h-12 w-12 text-primary" />
                  </div>
                  <p className="text-muted-foreground">Analyzing your symptoms and finding the best doctor match...</p>
                </div>
              )}
              
              {recommendedDoctor && (
                <div className="space-y-6">
                  <div className="text-center mb-6">
                    <h2 className="text-xl font-semibold">Your Recommended Doctor</h2>
                    <p className="text-sm text-gray-600">Based on your conversation, we've found the perfect specialist for you.</p>
                  </div>
                  
                  <Card className="p-4 sm:p-6 border border-blue-100 shadow-md">
                    <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center mb-4">
                      <Avatar className="h-16 w-16 sm:h-20 sm:w-20 rounded-md">
                        <img src="/doctorimage.jpg" alt={recommendedDoctor.doctor_details.name} />
                      </Avatar>
                      
                      <div className="flex-1">
                        <h3 className="text-lg sm:text-xl font-bold">{recommendedDoctor.doctor_details.name}</h3>
                        <p className="text-primary font-medium">{recommendedDoctor.doctor_details.specialty}</p>
                        
                        <div className="flex flex-col sm:flex-row sm:gap-4 mt-2">
                          <div className="flex items-center gap-1 text-sm">
                            <span className="text-gray-600">Experience:</span>
                            <span className="font-medium">{recommendedDoctor.doctor_details.experience_years} years</span>
                          </div>
                          
                          {recommendedDoctor.doctor_details.doctorContact && (
                            <div className="flex items-center gap-1 text-sm">
                              <span className="text-gray-600">Contact:</span>
                              <span className="font-medium">{recommendedDoctor.doctor_details.doctorContact}</span>
                            </div>
                          )}
                        </div>
                        
                        {recommendedDoctor.doctor_details.available_hours && (
                          <div className="text-sm mt-1">
                            <span className="text-gray-600">Available Hours:</span>
                            <span className="font-medium ml-1">{recommendedDoctor.doctor_details.available_hours}</span>
                          </div>
                        )}
                      </div>
                      
                      <Button 
                        className={`${consultationRequested ? 'bg-blue-600 hover:bg-blue-700' : 'bg-green-600 hover:bg-green-700'} w-full sm:w-auto`}
                        onClick={() => setConsultationRequested(true)}
                        disabled={consultationRequested}
                      >
                        {consultationRequested ? (
                          <div className="flex items-center gap-1">
                            <Check className="h-4 w-4" />
                            <span>Consultation Requested</span>
                          </div>
                        ) : (
                          "Request Consultation"
                        )}
                      </Button>
                    </div>
                    
                    {recommendedDoctor.doctor_details.description && (
                      <div className="mt-4 p-3 bg-gray-50 rounded-md text-sm">
                        <p className="font-medium mb-1">About the Doctor:</p>
                        <p>{recommendedDoctor.doctor_details.description}</p>
                      </div>
                    )}
                    
                    {/* AI Analysis */}
                    <div className="mt-6 border-t pt-4">
                      <h4 className="font-semibold mb-3">AI Analysis of Your Symptoms</h4>
                      <div className="prose prose-sm max-w-none">
                        <div className="whitespace-pre-line text-sm">
                          {recommendedDoctor.chatgpt_analysis}
                        </div>
                      </div>
                    </div>
                  </Card>
                  
                  <div className="flex justify-center">
                    <Button 
                      variant="outline" 
                      onClick={handleReset}
                      className="mr-2"
                    >
                      Reset
                    </Button>
                    <Button 
                      className={`${consultationRequested ? 'bg-blue-600 hover:bg-blue-700' : 'bg-green-600 hover:bg-green-700'}`}
                      onClick={() => setConsultationRequested(true)}
                      disabled={consultationRequested}
                    >
                      {consultationRequested ? (
                        <div className="flex items-center gap-1">
                          <Check className="h-4 w-4" />
                          <span>Consultation Requested</span>
                        </div>
                      ) : (
                        "Schedule Appointment"
                      )}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  )
}


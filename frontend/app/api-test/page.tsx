"use client";

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { AlertCircle, CheckCircle, ArrowLeft, RotateCw } from "lucide-react"
import Link from "next/link"

// API URL - must match the one in the main page
const API_URL = "http://127.0.0.1:5010/chatbot_response"

// Sample patient IDs to test with
const PATIENT_IDS = [535, 1, 2, 3, 100, 200, "custom"];

export default function ApiTestPage() {
  type StatusType = "idle" | "loading" | "success" | "error";
  const [status, setStatus] = useState<StatusType>("idle")
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [responseTime, setResponseTime] = useState<number | null>(null)
  const [responseData, setResponseData] = useState<any>(null)
  const [selectedPatientId, setSelectedPatientId] = useState<number | string>(535)
  const [customPatientId, setCustomPatientId] = useState<string>("")

  const testApiConnection = async () => {
    setStatus("loading")
    setErrorMessage(null)
    setResponseData(null)
    
    const startTime = performance.now()
    const patientId = selectedPatientId === "custom" ? 
      parseInt(customPatientId, 10) || 1 : selectedPatientId;
    
    try {
      // Send a test message to the API with the correct format
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          patientId: patientId,
          content: "API test message" 
        }),
        // Set a reasonable timeout
        signal: AbortSignal.timeout(5000),
      })
      
      const endTime = performance.now()
      setResponseTime(Math.round(endTime - startTime))
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`)
      }
      
      const data = await response.json()
      setResponseData(data)
      setStatus("success")
    } catch (error) {
      setStatus("error")
      if (error instanceof Error) {
        if (error.name === "AbortError") {
          setErrorMessage("Request timed out. The API server might be down or responding too slowly.")
        } else if (error.name === "TypeError" && error.message.includes("Failed to fetch")) {
          setErrorMessage("Could not connect to the API server. Make sure it's running and accessible.")
        } else {
          setErrorMessage(error.message)
        }
      } else {
        setErrorMessage("An unknown error occurred")
      }
      console.error("API test error:", error)
    }
  }

  // Run the test automatically when the page loads
  useEffect(() => {
    testApiConnection()
  }, [])

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-50">
      <Card className="w-full max-w-md p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">API Connection Test</h1>
          <Link href="/" passHref>
            <Button variant="ghost" size="sm" className="flex items-center gap-1">
              <ArrowLeft className="h-4 w-4" />
              <span>Back</span>
            </Button>
          </Link>
        </div>
        
        {/* Patient ID Selection */}
        <div className="w-full">
          <label htmlFor="patientId" className="block text-sm font-medium text-gray-700 mb-1">
            Patient ID
          </label>
          <div className="flex gap-2">
            <select
              id="patientId"
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="flex-1 rounded-md border border-gray-300 shadow-sm px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary"
            >
              {PATIENT_IDS.map((id) => (
                <option key={id} value={id}>
                  {id === "custom" ? "Custom ID..." : id}
                </option>
              ))}
            </select>
            <Button 
              onClick={testApiConnection} 
              disabled={status === "loading"}
              size="sm"
            >
              Test
            </Button>
          </div>
        </div>
        
        {selectedPatientId === "custom" && (
          <div className="mt-2">
            <input
              type="number"
              placeholder="Enter custom patient ID"
              value={customPatientId}
              onChange={(e) => setCustomPatientId(e.target.value)}
              className="w-full rounded-md border border-gray-300 shadow-sm px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary"
            />
          </div>
        )}
        
        <div className="flex flex-col items-center justify-center py-8 space-y-4">
          {status === "loading" && (
            <div className="flex flex-col items-center space-y-3">
              <div className="animate-spin">
                <RotateCw className="h-10 w-10 text-primary" />
              </div>
              <p className="text-muted-foreground">Testing connection to API...</p>
            </div>
          )}
          
          {status === "success" && (
            <div className="flex flex-col items-center space-y-3 w-full">
              <div className="text-green-500">
                <CheckCircle className="h-12 w-12" />
              </div>
              <p className="font-medium text-lg">Connection Successful!</p>
              <p className="text-sm text-muted-foreground">
                The API responded in {responseTime}ms
              </p>
              
              {responseData && (
                <div className="w-full mt-4 p-3 bg-gray-50 border border-gray-200 rounded-md text-sm">
                  <p className="font-medium mb-2">Response Preview:</p>
                  <div className="bg-white p-2 rounded border border-gray-200 font-mono text-xs overflow-auto max-h-36">
                    <pre>{JSON.stringify(responseData, null, 2)}</pre>
                  </div>
                  
                  <div className="mt-3 p-2 bg-blue-50 border border-blue-100 rounded text-xs text-blue-800">
                    <p className="font-medium">Expected Format:</p>
                    <ul className="list-disc list-inside mt-1 space-y-1">
                      <li><span className="font-mono">response</span>: Text response from chatbot</li>
                      <li><span className="font-mono">chat_history</span>: Array of messages with role ("user" or "assistant") and content</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {status === "error" && (
            <div className="flex flex-col items-center space-y-3">
              <div className="text-red-500">
                <AlertCircle className="h-12 w-12" />
              </div>
              <p className="font-medium text-lg">Connection Failed</p>
              <div className="bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-800 w-full">
                <p className="font-medium">Error Details:</p>
                <p>{errorMessage}</p>
              </div>
              <div className="bg-amber-50 border border-amber-200 rounded-md p-3 text-sm text-amber-800 w-full">
                <p className="font-medium">Troubleshooting Tips:</p>
                <ul className="list-disc list-inside space-y-1 mt-1">
                  <li>Check if your backend server is running at {API_URL}</li>
                  <li>Verify there are no CORS issues preventing the connection</li>
                  <li>Make sure your API accepts POST requests with JSON payloads</li>
                </ul>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex justify-center">
          <Button 
            onClick={testApiConnection} 
            disabled={status === "loading"}
            className="w-full max-w-xs"
          >
            {status === "loading" ? "Testing..." : "Test Connection Again"}
          </Button>
        </div>
      </Card>
    </div>
  )
} 
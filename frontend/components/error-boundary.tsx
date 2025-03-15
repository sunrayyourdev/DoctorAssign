"use client"

import React, { Component, ErrorInfo, ReactNode } from "react"
import { AlertCircle, RotateCw } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = {
      hasError: false,
      error: null
    }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("Error caught by boundary:", error)
    console.error("Component stack:", errorInfo.componentStack)
  }

  handleRetry = (): void => {
    this.setState({
      hasError: false,
      error: null
    })
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="flex flex-col items-center justify-center min-h-[400px] p-4 border rounded-lg bg-gray-50">
          <div className="flex flex-col items-center space-y-4 max-w-md text-center">
            <div className="text-red-500">
              <AlertCircle className="h-12 w-12" />
            </div>
            <h2 className="text-xl font-semibold">Something went wrong</h2>
            <div className="bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-800 w-full text-left">
              <p className="font-medium">Error Details:</p>
              <p className="break-words">{this.state.error?.message || "An unknown error occurred"}</p>
            </div>
            <Button 
              onClick={this.handleRetry}
              className="flex items-center gap-2"
            >
              <RotateCw className="h-4 w-4" />
              <span>Try Again</span>
            </Button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary 
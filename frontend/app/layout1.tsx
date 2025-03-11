import type React from "react" 
import LayoutClient from "./layout1-client"

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return <LayoutClient>{children}</LayoutClient>
}

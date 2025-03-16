"use client"

import type React from "react"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { User } from "lucide-react"

import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"

export default function Navbar() {
  const pathname = usePathname()

  return (
    <nav className="sticky top-0 z-10 flex h-16 w-full items-center justify-between bg-[#f5f5dc] px-4 shadow-md md:px-6">
      {/* Logo */}
      <div className="flex items-center">
        <Link href="/" className="flex items-center gap-2 font-bold text-gray-800">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <span className="text-lg font-bold">D</span>
          </div>
          <span className="hidden md:inline-block">DoctorAssign</span>
        </Link>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-1 md:gap-2">
        <NavItem href="/doctor-dashboard" active={pathname === "/doctor-dashboard"}>
          Doctor Dashboard
        </NavItem>
        <NavItem href="/api-test" active={pathname === "/api-test"}>
          API Test
        </NavItem>
      </div> 

      {/* User Profile */}
      <div className="flex items-center">
      <NavItem href="/profile" active={pathname === "/profile"}>
        <Avatar className="h-8 w-8 cursor-pointer transition-transform hover:scale-110">
          <AvatarFallback className="bg-primary text-primary-foreground">
            <User className="h-4 w-4" />
          </AvatarFallback>
        </Avatar>
      </NavItem>
      </div>
    </nav>
  )
}

interface NavItemProps {
  href: string
  active?: boolean
  children: React.ReactNode
}

function NavItem({ href, active, children }: NavItemProps) {
  return (
    <Link
      href={href}
      className={cn(
        "relative px-3 py-2 text-sm font-medium text-gray-800 transition-colors",
        "hover:text-primary",
        "after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-full after:origin-bottom-right after:scale-x-0 after:bg-primary after:transition-transform after:duration-300 after:ease-in-out",
        "hover:after:origin-bottom-left hover:after:scale-x-100",
        active && "text-primary after:origin-bottom-left after:scale-x-100",
      )}
    >
      {children}
    </Link>
  )
}


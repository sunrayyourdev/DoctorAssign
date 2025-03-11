import Dashboard from "@/dashboard"
import Link from "next/link";

export default function Page() {
  return (
    <main className="min-h-screen p-8">
        <Dashboard />
        <div className="mt-4">
            <h1>Admin Dashboard</h1>
      </div>
    </main>
  )
}
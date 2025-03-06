import Dashboard from "@/dashboard"
import Link from "next/link";

export default function Page() {
  return (
    <main className="min-h-screen p-8">
        <Dashboard />
        <div className="mt-4">
        <Link href="/profile">
          <button className="px-4 py-2 bg-blue-500 text-white rounded-md">
            Go to Profile
          </button>
        </Link>
      </div>
    </main>
  )
}
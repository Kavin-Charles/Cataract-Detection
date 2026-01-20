import { Eye } from 'lucide-react';

export function Header() {
  return (
    <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/20 overflow-hidden">
            <img 
              src="https://images.unsplash.com/photo-1500042450384-8b7947f4eec2?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxleWUlMjBtZWRpY2FsJTIwbG9nb3xlbnwxfHx8fDE3Njg4OTcwMzV8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
              alt="CataractScan Logo"
              className="size-6 object-cover rounded"
            />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-slate-800">CataractScan</h1>
            <p className="text-xs text-slate-500">Advanced Eye Health Analysis</p>
          </div>
        </div>
      </div>
    </header>
  );
}
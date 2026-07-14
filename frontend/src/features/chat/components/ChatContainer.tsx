import React, { useRef, useEffect, useState } from "react";
import { 
  Send, 
  Paperclip, 
  HelpCircle, 
  Sparkles, 
  Loader2, 
  ArrowRight,
  Info
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useChatStore } from "@/stores/useChatStore";
import { useUIStore } from "@/stores/useUIStore";
import { useChat } from "@/hooks/useChat";
import { MarkdownRenderer } from "./MarkdownRenderer";

export const ChatContainer: React.FC = () => {
  const { messages, isGenerating } = useChatStore();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const { sendMessage } = useChat();
  const [inputValue, setInputValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const suggestionPrompts = [
    "Bankın minimum nizamnamə kapitalı nə qədərdir?",
    "Kredit təşkilatlarında risklərin idarə olunması qaydaları hansılardır?",
    "Mərkəzi Bankın funksiyaları və kapitalı barədə məlumat verin."
  ];

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSend = () => {
    if (!inputValue.trim() || isGenerating) return;
    sendMessage(inputValue.trim());
    setInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionClick = (prompt: string) => {
    if (isGenerating) return;
    sendMessage(prompt);
  };

  return (
    <div className="flex-1 flex flex-col h-full min-w-0 bg-background">
      
      {/* Chat Area Header */}
      <header className="h-16 border-b flex items-center justify-between px-4 sm:px-6 shrink-0 bg-card">
        <div className="flex items-center gap-2">
          {!sidebarOpen && (
            <Button 
              variant="ghost" 
              size="sm" 
              className="md:hidden text-muted-foreground mr-1"
              onClick={toggleSidebar}
            >
              ☰
            </Button>
          )}
          <div className="text-left">
            <h2 className="text-sm font-bold flex items-center gap-1.5 text-foreground">
              <span>Tənzimləyici Köməkçi</span>
              <span className="text-[9px] bg-gold-100 text-gold-800 dark:bg-gold-950/30 dark:text-gold-400 font-bold px-1 py-0.5 rounded">AI</span>
            </h2>
            <p className="text-[10px] text-muted-foreground font-light">Azərbaycan Respublikası Mərkəzi Bankının tənzimləmələri</p>
          </div>
        </div>
      </header>

      {/* Messages / Suggestions */}
      <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        {messages.length === 0 ? (
          /* Empty State / Suggested Prompts */
          <div className="max-w-2xl mx-auto h-full flex flex-col justify-center items-center text-center space-y-6">
            <div className="p-4 bg-gold-100/50 dark:bg-navy-900 rounded-2xl border border-gold-200/20">
              <Sparkles className="h-8 w-8 text-gold-500 animate-pulse" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-bold">Necə kömək edə bilərəm?</h3>
              <p className="text-xs text-muted-foreground font-light max-w-md">
                Mərkəzi Bankın normativ aktları, prudensial tələbləri, risk limitləri və daxili audit normaları barədə sual verin.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-3 w-full max-w-lg pt-4">
              {suggestionPrompts.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSuggestionClick(prompt)}
                  className="flex items-center justify-between p-3.5 border rounded-xl bg-card hover:bg-secondary text-xs text-left font-medium transition-colors group cursor-pointer"
                >
                  <span className="text-foreground/90">{prompt}</span>
                  <ArrowRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-gold-500 transition-transform group-hover:translate-x-1" />
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Chat History Thread */
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                
                {/* Assistant icon */}
                {message.role === "assistant" && (
                  <div className="h-8 w-8 rounded-full border bg-navy-900 text-white dark:bg-gold-500 dark:text-navy-950 flex items-center justify-center text-xs font-bold shrink-0 shadow-sm">
                    R
                  </div>
                )}

                {/* Message Bubble Container */}
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm border text-left ${
                    message.role === "user"
                      ? "bg-navy-900 text-white dark:bg-white dark:text-navy-900 border-navy-950 dark:border-gray-200"
                      : "bg-card border-border/80 border-l-4 border-l-gold-500"
                  }`}
                >
                  {message.role === "user" ? (
                    <p className="text-sm whitespace-pre-wrap font-light">{message.content}</p>
                  ) : (
                    <div className="space-y-2">
                      {/* Check if generating empty token message */}
                      {message.content === "" && message.isStreaming ? (
                        <div className="flex items-center gap-2 py-2">
                          <Loader2 className="h-4 w-4 animate-spin text-gold-500" />
                          <span className="text-xs text-muted-foreground font-light">Araşdırılır...</span>
                        </div>
                      ) : (
                        <MarkdownRenderer content={message.content} sources={message.sources} />
                      )}

                      {/* Display Metrics metadata */}
                      {message.metrics && !message.isStreaming && (
                        <div className="pt-2 mt-2 border-t border-border/40 flex items-center gap-1.5 text-[9px] text-muted-foreground font-light font-mono">
                          <Info className="h-3 w-3 text-gold-500" />
                          <span>Axtarış: {message.metrics.retrieval_time.toFixed(3)}s</span>
                          <span>•</span>
                          <span>Generasiya: {message.metrics.generation_time.toFixed(3)}s</span>
                          <span>•</span>
                          <span>Cəmi: {message.metrics.total_time.toFixed(3)}s</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* User avatar initial */}
                {message.role === "user" && (
                  <div className="h-8 w-8 rounded-full bg-gold-500 text-white font-bold flex items-center justify-center text-xs shrink-0 shadow-sm">
                    U
                  </div>
                )}

              </div>
            ))}
            
            {/* Scroll bottom placeholder */}
            <div ref={scrollRef} />
          </div>
        )}
      </div>

      {/* Input box section */}
      <div className="p-4 sm:p-6 bg-gradient-to-t from-background via-background/95 to-transparent shrink-0">
        <div className="max-w-3xl mx-auto space-y-2">
          
          <div className="relative border rounded-2xl bg-card shadow-sm focus-within:ring-1 focus-within:ring-gold-500 focus-within:border-gold-500 transition-shadow">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isGenerating}
              rows={1}
              placeholder="Bank normativ aktları barədə sual verin..."
              className="w-full pl-4 pr-24 py-3 bg-transparent text-foreground text-sm focus:outline-none resize-none min-h-[48px] max-h-[160px] font-light placeholder:text-muted-foreground/60"
              style={{ height: "auto" }}
            />
            
            <div className="absolute right-2 bottom-2.5 flex items-center gap-1.5">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground hover:text-foreground opacity-50 cursor-not-allowed"
                title="Sənəd yükləyin (Tezliklə)"
                disabled
              >
                <Paperclip className="h-4 w-4" />
              </Button>
              <Button
                onClick={handleSend}
                disabled={!inputValue.trim() || isGenerating}
                size="icon"
                className="h-8 w-8 bg-navy-900 text-white hover:bg-navy-800 dark:bg-gold-500 dark:text-navy-950 dark:hover:bg-gold-600 rounded-xl"
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          
          {/* Disclaimer copy */}
          <div className="flex items-center justify-center gap-1.5 text-[10px] text-muted-foreground font-light text-center">
            <HelpCircle className="h-3.5 w-3.5 text-gold-500 shrink-0" />
            <span>Süni intellekt xətalara yol verə bilər. Qərarların qəbulu üçün normativ aktların rəsmi mətninə istinad edin.</span>
          </div>

        </div>
      </div>

    </div>
  );
};

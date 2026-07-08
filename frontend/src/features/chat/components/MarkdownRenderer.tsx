import React from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useUIStore } from "@/stores/useUIStore";
import { useChatStore } from "@/stores/useChatStore";

interface MarkdownRendererProps {
  content: string;
  sources?: any[];
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, sources }) => {
  const { setSelectedCitation, setActiveDocument } = useUIStore();
  const { activeSessionId } = useChatStore();

  const handleCitationClick = (citationNumber: number) => {
    if (!sources) return;
    
    // Find the source corresponding to this citation number
    const source = sources.find((s) => s.citation === citationNumber);
    if (source) {
      // 1. Dispatch citation click selection state
      setSelectedCitation({
        citationId: citationNumber,
        chunkId: source.chunk_id,
        documentId: source.document_id || "",
      });

      // 2. Set active document state in the Document Viewer panel
      setActiveDocument({
        documentId: source.document_id || "",
        activePage: source.page || 1,
        highlightText: source.chunk_preview,
      });
    }
  };

  // Utility to scan text and replace [N] citation markers with interactive buttons
  const renderTextWithCitations = (text: string) => {
    const regex = /\[(\d+)\]/g;
    const parts = text.split(regex);
    if (parts.length === 1) return text;

    return parts.map((part, idx) => {
      // Odd indices contain the capture group (digits)
      if (idx % 2 === 1) {
        const num = parseInt(part, 10);
        return (
          <button
            key={idx}
            onClick={() => handleCitationClick(num)}
            className="inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 text-[10px] font-bold border border-gold-400 bg-gold-50 text-gold-800 dark:bg-gold-950/20 dark:text-gold-300 rounded hover:bg-gold-100 dark:hover:bg-gold-900 transition-colors font-sans focus:outline-none focus:ring-1 focus:ring-gold-500 cursor-pointer"
            title={sources?.find(s => s.citation === num)?.document_name || `İstinad [${num}]`}
          >
            [{num}]
          </button>
        );
      }
      return part;
    });
  };

  // Helper to map children recursively and inject citations into text leaves
  const injectCitations = (children: React.ReactNode): React.ReactNode => {
    return React.Children.map(children, (child) => {
      if (typeof child === "string") {
        return renderTextWithCitations(child);
      }
      if (React.isValidElement(child) && child.props.children) {
        return React.cloneElement(child, {
          ...child.props,
          children: injectCitations(child.props.children),
        });
      }
      return child;
    });
  };

  return (
    <div className="prose prose-navy dark:prose-invert max-w-none text-sm leading-relaxed font-sans space-y-3">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Render paragraph with citation highlights
          p: ({ children }) => (
            <p className="text-foreground/90 font-light mb-3">
              {injectCitations(children)}
            </p>
          ),
          // Styled lists
          ul: ({ children }) => (
            <ul className="list-disc pl-5 space-y-1 my-2 text-foreground/90 font-light">
              {injectCitations(children)}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-5 space-y-1 my-2 text-foreground/90 font-light">
              {injectCitations(children)}
            </ol>
          ),
          li: ({ children }) => (
            <li className="text-sm">
              {injectCitations(children)}
            </li>
          ),
          // Styled tables
          table: ({ children }) => (
            <div className="overflow-x-auto my-4 rounded-xl border border-border">
              <table className="min-w-full divide-y divide-border text-xs text-left bg-card">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-secondary/40 font-semibold text-foreground uppercase tracking-wider text-[10px]">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-border">
              {children}
            </tbody>
          ),
          tr: ({ children }) => <tr>{children}</tr>,
          th: ({ children }) => <th className="px-4 py-2 border-b">{injectCitations(children)}</th>,
          td: ({ children }) => <td className="px-4 py-2 border-b text-foreground/80">{injectCitations(children)}</td>,
          
          // Bold / italic custom colors
          strong: ({ children }) => <strong className="font-bold text-navy-950 dark:text-white">{children}</strong>,
          em: ({ children }) => <em className="italic text-foreground/90">{children}</em>,
          
          // Inline and block code blocks
          code: ({ className, children }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="px-1.5 py-0.5 rounded bg-secondary text-navy-900 dark:bg-navy-900/60 dark:text-gold-300 text-xs font-mono">
                  {children}
                </code>
              );
            }
            return (
              <pre className="p-4 rounded-xl bg-navy-900 text-white font-mono text-xs overflow-x-auto my-3 select-all">
                <code>{children}</code>
              </pre>
            );
          },
          // Links styling
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gold-500 hover:text-gold-600 underline font-medium transition-colors cursor-pointer"
            >
              {children}
            </a>
          )
        }}
      >
        {content}
      </Markdown>
    </div>
  );
};

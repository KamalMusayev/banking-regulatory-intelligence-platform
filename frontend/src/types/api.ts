export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface SourceDocument {
  citation: number;
  chunk_id: string;
  document_id: string | null;
  document_name: string;
  category: string;
  chapter?: string | null;
  article?: string | null;
  page?: number | null;
  chunk_preview: string;
  rerank_score?: number | null;
  rrf_score?: number | null;
  semantic_rank?: number | null;
  bm25_rank?: number | null;
}

export interface MetricsResponse {
  retrieval_time: number;
  generation_time: number;
  total_time: number;
}

export interface ChatRequest {
  question: string;
  session_id?: string | null;
}

export interface ChatResponse {
  session_id: string;
  question: string;
  answer: string;
  sources: SourceDocument[];
  metrics: MetricsResponse;
}

export interface DocumentMetadataResponse {
  document_id: string;
  title: string;
  category: string;
  total_pages?: number | null;
  total_chunks?: number | null;
  language?: string | null;
  parser?: string | null;
  publication_date?: string | null;
  status: string; // "active" | "archived"
  related_articles: string[];
  document_metadata?: Record<string, any>;
}

export interface ArticleInfo {
  chapter?: string | null;
  article?: string | null;
  section?: string | null;
  chunk_id: string;
}

export interface DocumentPageResponse {
  document_id: string;
  page_number: number;
  page_content: string;
  article_information: ArticleInfo[];
  metadata?: Record<string, any>;
}

export interface DocumentHighlightResponse {
  document_id: string;
  page: number;
  article?: string | null;
  chunk_id: string;
  chunk_start?: number | null;
  chunk_end?: number | null;
  highlighted_text: string;
  offset_status: string; // "supported" | "future_enhancement"
}

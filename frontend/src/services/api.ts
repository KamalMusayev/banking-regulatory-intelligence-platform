import { mockService } from "./mockService";
import { 
  ChatResponse, 
  DocumentMetadataResponse, 
  DocumentPageResponse, 
  DocumentHighlightResponse 
} from "../types/api";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Helper to determine if we should use mock API
const useMockApi = (): boolean => {
  const envMockSetting = import.meta.env.VITE_USE_MOCK_API;
  
  // Allow toggling in localStorage so the user can change it dynamically in Settings
  const localOverride = localStorage.getItem("reguaz-use-mock-api");
  if (localOverride !== null) {
    return localOverride === "true";
  }

  return envMockSetting === "true" || envMockSetting === undefined;
};

// Response helper
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const message = errorData.error?.message || `API error: ${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const apiService = {
  // Toggle check
  isMockEnabled: () => useMockApi(),
  
  // Set toggle dynamically
  setMockEnabled: (enabled: boolean) => {
    localStorage.setItem("reguaz-use-mock-api", enabled ? "true" : "false");
  },

  // 1. GET /health
  getHealth: async (): Promise<{ status: string; app: string }> => {
    if (useMockApi()) {
      return mockService.getHealth();
    }
    
    try {
      const res = await fetch(`${API_URL}/health`);
      return handleResponse<{ status: string; app: string }>(res);
    } catch (error) {
      console.warn("Failed to connect to backend, falling back to Mock API:", error);
      return mockService.getHealth();
    }
  },

  // 2. POST /chat
  postChat: async (question: string, sessionId?: string | null): Promise<ChatResponse> => {
    if (useMockApi()) {
      return mockService.postChat(question, sessionId);
    }

    const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    return handleResponse<ChatResponse>(res);
  },

  // 3. GET /documents
  getDocuments: async (): Promise<DocumentMetadataResponse[]> => {
    if (useMockApi()) {
      return mockService.getDocuments();
    }

    try {
      const res = await fetch(`${API_URL}/documents`);
      return handleResponse<DocumentMetadataResponse[]>(res);
    } catch (e) {
      // If endpoint is not found or not connected, return mock adapter list
      console.warn("Real /documents endpoint failed, using mock data adapter:", e);
      return mockService.getDocuments();
    }
  },

  // 4. GET /documents/:id
  getDocumentMetadata: async (documentId: string): Promise<DocumentMetadataResponse | null> => {
    if (useMockApi()) {
      return mockService.getDocumentMetadata(documentId);
    }

    const res = await fetch(`${API_URL}/documents/${encodeURIComponent(documentId)}`);
    return handleResponse<DocumentMetadataResponse>(res);
  },

  // 5. GET /documents/:id/page/:page_number
  getDocumentPage: async (documentId: string, pageNumber: number): Promise<DocumentPageResponse | null> => {
    if (useMockApi()) {
      return mockService.getDocumentPage(documentId, pageNumber);
    }

    const res = await fetch(`${API_URL}/documents/${encodeURIComponent(documentId)}/page/${pageNumber}`);
    return handleResponse<DocumentPageResponse>(res);
  },

  // 6. GET /documents/highlight
  getHighlight: async (documentId: string, chunkId: string): Promise<DocumentHighlightResponse | null> => {
    if (useMockApi()) {
      return mockService.getHighlight(documentId, chunkId);
    }

    const res = await fetch(
      `${API_URL}/documents/highlight?document_id=${encodeURIComponent(documentId)}&chunk_id=${encodeURIComponent(chunkId)}`
    );
    return handleResponse<DocumentHighlightResponse>(res);
  },

  // 7. Conversation History Hooks placeholder (GET, POST, DELETE /history)
  getHistory: async (): Promise<any> => {
    // History is managed in Zustand and local mock storage, representing Auth / session states
    await new Promise(r => setTimeout(r, 100));
    return [];
  },

  postHistory: async (session: any): Promise<any> => {
    await new Promise(r => setTimeout(r, 100));
    return session;
  },

  deleteHistory: async (sessionId: string): Promise<boolean> => {
    await new Promise(r => setTimeout(r, 100));
    return true;
  }
};

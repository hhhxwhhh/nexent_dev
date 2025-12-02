/**
 * Authentication utilities
 */

import { fetchWithErrorHandling, ApiError } from "@/services/api";
import { STORAGE_KEYS, STATUS_CODES } from "@/const/auth";
import { Session } from "@/types/auth";
import { generateAvatarUrl as generateAvatar } from "@/lib/avatar";
import log from "@/lib/logger";

// Get color corresponding to user role
export function getRoleColor(role: string): string {
  switch (role) {
    case "admin":
      return "purple"
    case "user":
    default:
      return "geekblue"
  }
}

// Generate avatar based on email (re-export from avatar.tsx for backward compatibility)
export function generateAvatarUrl(email: string): string {
  return generateAvatar(email);
}

/**
 * Request with authorization headers
 * Checks token expiration before sending request to prevent sending expired tokens
 */
export const fetchWithAuth = async (url: string, options: RequestInit = {}) => {
  const session = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEYS.SESSION) : null;
  const sessionObj = session ? JSON.parse(session) : null;

  // Check if token is expired before sending request
  if (sessionObj?.access_token) {
    const now = Date.now();
    const expiresAt = sessionObj.expires_at ? sessionObj.expires_at * 1000 : 0;
    
    // If token is expired, clear session and throw error
    if (expiresAt > 0 && expiresAt <= now) {
      log.warn("Token expired, clearing session before request");
      removeSessionFromStorage();
      
      // Dispatch session expired event
      if (typeof window !== "undefined" && window.dispatchEvent) {
        window.dispatchEvent(new CustomEvent('session-expired', {
          detail: { message: "Login expired, please login again" }
        }));
      }
      
      throw new ApiError(STATUS_CODES.TOKEN_EXPIRED, "Login expired, please login again");
    }
  }

  const isFormData = options.body instanceof FormData;
  const headers = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(sessionObj?.access_token && { "Authorization": `Bearer ${sessionObj.access_token}` }),
    ...options.headers,
  };

  // Use request interceptor with error handling
  return fetchWithErrorHandling(url, {
    ...options,
    headers,
  });
};

/**
 * Save session to local storage
 */
export const saveSessionToStorage = (session: Session) => {
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEYS.SESSION, JSON.stringify(session));
  }
};

/**
 * Remove session from local storage
 */
export const removeSessionFromStorage = () => {
  if (typeof window !== "undefined") {
    localStorage.removeItem(STORAGE_KEYS.SESSION);
  }
};

/**
 * Get session from local storage
 */
export const getSessionFromStorage = (): Session | null => {
  try {
    const storedSession = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEYS.SESSION) : null;
    if (!storedSession) return null;

    return JSON.parse(storedSession);
  } catch (error) {
    log.error("Failed to parse session info:", error);
    return null;
  }
};

/**
 * Get the authorization header information for API requests
 * @returns HTTP headers object containing authentication and content type information
 */
export const getAuthHeaders = () => {
  const session = typeof window !== "undefined" ? localStorage.getItem("session") : null;
  const sessionObj = session ? JSON.parse(session) : null;

  return {
    'Content-Type': 'application/json',
    'User-Agent': 'AgentFrontEnd/1.0',
    ...(sessionObj?.access_token && { "Authorization": `Bearer ${sessionObj.access_token}` }),
  };
};
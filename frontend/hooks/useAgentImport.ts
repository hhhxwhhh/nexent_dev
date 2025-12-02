import { useState } from "react";
import { importAgent } from "@/services/agentConfigService";
import log from "@/lib/logger";

export interface ImportAgentData {
  agent_id: number;
  agent_info: Record<string, any>;
  mcp_info?: Array<{
    mcp_server_name: string;
    mcp_url: string;
  }>;
}

export interface UseAgentImportOptions {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
  forceImport?: boolean;
}

export interface UseAgentImportResult {
  isImporting: boolean;
  importFromFile: (file: File) => Promise<void>;
  importFromData: (data: ImportAgentData) => Promise<void>;
  error: Error | null;
}

/**
 * Unified agent import hook
 * Handles agent import from both file upload and direct data
 * Used in:
 * - Agent development (SubAgentPool)
 * - Agent space (SpaceContent)
 * - Agent market (MarketContent)
 */
export function useAgentImport(
  options: UseAgentImportOptions = {}
): UseAgentImportResult {
  const { onSuccess, onError, forceImport = false } = options;

  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  /**
   * Import agent from uploaded file
   */
  const importFromFile = async (file: File): Promise<void> => {
    setIsImporting(true);
    setError(null);

    try {
      // Read file content
      const fileContent = await readFileAsText(file);
      
      // Parse JSON
      let agentData: ImportAgentData;
      try {
        agentData = JSON.parse(fileContent);
      } catch (parseError) {
        throw new Error("Invalid JSON file format");
      }

      // Validate structure
      if (!agentData.agent_id || !agentData.agent_info) {
        throw new Error("Invalid agent data structure");
      }

      // Import using unified logic
      await importAgentData(agentData);
      
      onSuccess?.();
    } catch (err) {
      const error = err instanceof Error ? err : new Error("Unknown error");
      log.error("Failed to import agent from file:", error);
      setError(error);
      onError?.(error);
      throw error;
    } finally {
      setIsImporting(false);
    }
  };

  /**
   * Import agent from data object (e.g., from market)
   */
  const importFromData = async (data: ImportAgentData): Promise<void> => {
    setIsImporting(true);
    setError(null);

    try {
      // Validate structure
      if (!data.agent_id || !data.agent_info) {
        throw new Error("Invalid agent data structure");
      }

      // Import using unified logic
      await importAgentData(data);
      
      onSuccess?.();
    } catch (err) {
      const error = err instanceof Error ? err : new Error("Unknown error");
      log.error("Failed to import agent from data:", error);
      setError(error);
      onError?.(error);
      throw error;
    } finally {
      setIsImporting(false);
    }
  };

  /**
   * Core import logic - calls backend API
   */
  const importAgentData = async (data: ImportAgentData): Promise<void> => {
    const result = await importAgent(data, { forceImport });
    
    if (!result.success) {
      throw new Error(result.message || "Failed to import agent");
    }
  };

  /**
   * Helper: Read file as text
   */
  const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        const content = e.target?.result;
        if (typeof content === "string") {
          resolve(content);
        } else {
          reject(new Error("Failed to read file content"));
        }
      };
      
      reader.onerror = () => {
        reject(new Error("Failed to read file"));
      };
      
      reader.readAsText(file);
    });
  };

  return {
    isImporting,
    importFromFile,
    importFromData,
    error,
  };
}


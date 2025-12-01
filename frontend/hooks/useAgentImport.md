# useAgentImport Hook

Unified agent import hook for handling agent imports across the application.

## Overview

This hook provides a consistent interface for importing agents from different sources:
- File upload (used in Agent Development and Agent Space)
- Direct data (used in Agent Market)

All import operations ultimately call the same backend `/agent/import` endpoint.

## Usage

### Basic Import

```typescript
import { useAgentImport } from "@/hooks/useAgentImport";

function MyComponent() {
  const { isImporting, importFromFile, importFromData, error } = useAgentImport({
    onSuccess: () => {
      console.log("Import successful!");
    },
    onError: (error) => {
      console.error("Import failed:", error);
    },
  });

  // ...
}
```

### Import from File (SubAgentPool, SpaceContent)

```typescript
const handleFileImport = async (file: File) => {
  try {
    await importFromFile(file);
    // Success handled by onSuccess callback
  } catch (error) {
    // Error handled by onError callback
  }
};

// In file input handler
<input
  type="file"
  accept=".json"
  onChange={(e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileImport(file);
    }
  }}
/>
```

### Import from Data (Market)

```typescript
const handleMarketImport = async (agentDetails: MarketAgentDetail) => {
  // Prepare import data from agent details
  const importData = {
    agent_id: agentDetails.agent_id,
    agent_info: agentDetails.agent_json.agent_info,
    mcp_info: agentDetails.agent_json.mcp_info,
  };

  try {
    await importFromData(importData);
    // Success handled by onSuccess callback
  } catch (error) {
    // Error handled by onError callback
  }
};
```

## Integration Examples

### 1. SubAgentPool Component

```typescript
// In SubAgentPool.tsx
import { useAgentImport } from "@/hooks/useAgentImport";

export default function SubAgentPool({ onImportSuccess }: Props) {
  const { isImporting, importFromFile } = useAgentImport({
    onSuccess: () => {
      message.success(t("agent.import.success"));
      onImportSuccess?.();
    },
    onError: (error) => {
      message.error(error.message);
    },
  });

  const handleImportClick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        await importFromFile(file);
      }
    };
    input.click();
  };

  return (
    <button 
      onClick={handleImportClick}
      disabled={isImporting}
    >
      {isImporting ? t("importing") : t("import")}
    </button>
  );
}
```

### 2. SpaceContent Component

```typescript
// In SpaceContent.tsx
import { useAgentImport } from "@/hooks/useAgentImport";

export function SpaceContent({ onRefresh }: Props) {
  const { isImporting, importFromFile } = useAgentImport({
    onSuccess: () => {
      message.success(t("space.import.success"));
      onRefresh(); // Reload agent list
    },
  });

  const handleImportAgent = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        await importFromFile(file);
      }
    };
    input.click();
  };

  return (
    <button
      onClick={handleImportAgent}
      disabled={isImporting}
    >
      {isImporting ? "Importing..." : "Import Agent"}
    </button>
  );
}
```

### 3. AgentInstallModal (Market)

```typescript
// In AgentInstallModal.tsx
import { useAgentImport } from "@/hooks/useAgentImport";

export default function AgentInstallModal({ 
  agentDetails, 
  onComplete 
}: Props) {
  const { isImporting, importFromData } = useAgentImport({
    onSuccess: () => {
      message.success(t("market.install.success"));
      onComplete();
    },
  });

  const handleInstall = async () => {
    // Prepare configured data
    const importData = prepareImportData(agentDetails, userConfig);
    await importFromData(importData);
  };

  return (
    <Button
      onClick={handleInstall}
      loading={isImporting}
    >
      Install
    </Button>
  );
}
```

## API Reference

### Parameters

```typescript
interface UseAgentImportOptions {
  onSuccess?: () => void;       // Called on successful import
  onError?: (error: Error) => void;  // Called on import error
  forceImport?: boolean;         // Force import even if duplicate names exist
}
```

### Return Value

```typescript
interface UseAgentImportResult {
  isImporting: boolean;          // Import in progress
  importFromFile: (file: File) => Promise<void>;  // Import from file
  importFromData: (data: ImportAgentData) => Promise<void>;  // Import from data
  error: Error | null;           // Last error (if any)
}
```

### Data Structure

```typescript
interface ImportAgentData {
  agent_id: number;
  agent_info: Record<string, any>;
  mcp_info?: Array<{
    mcp_server_name: string;
    mcp_url: string;
  }>;
}
```

## Error Handling

The hook handles errors in two ways:

1. **Via onError callback** - Preferred method for user-facing error messages
2. **Via thrown exceptions** - For custom error handling in specific cases

Both approaches are supported to allow flexibility in different use cases.

## Implementation Notes

- File content is read as text and parsed as JSON
- Data structure validation is performed before calling the backend
- The backend `/agent/import` endpoint is called with the prepared data
- All logging uses the centralized `log` utility from `@/lib/logger`

